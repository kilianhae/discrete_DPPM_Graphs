import logging
import time
import os
import graph_tool.all as gt
from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation.stats import eval_torch_batch

from model.langevin_mc import LangevinMCSampler

from utils.arg_helper import edict2dict, parse_arguments, get_config, process_config, set_seed_and_logger, load_data
from utils.graph_utils import gen_list_of_data
from utils.loading_utils import get_mc_sampler, get_score_model, eval_sample_batch
from utils.visual_utils import plot_graphs_adj
from sample_edp_simple import sample_main_edp, sample_testing_edp
import wandb


# Gets data as list of tensors, each list corresponds to one sigmalevel, then we have tensordim matrices x i x j
def loss_func_bce(score_list, grad_log_q_noise_list, sigma_list):
    loss = 0.0
    for score, grad_log_q_noise, sigma in zip(score_list, grad_log_q_noise_list, sigma_list):
        BCE = torch.nn.BCEWithLogitsLoss()
        cur_loss = BCE(score, grad_log_q_noise)
        # Weight by 1-sigma, if sigma high then we have high noise so low weight
        loss = loss + cur_loss * (1-2*sigma+1/len(sigma_list))
    return loss


def fit(model, optimizer, mcmc_sampler, train_dl, max_node_number, max_epoch=20, config=None,
        save_interval=50,
        sample_interval=1,
        sigma_list=None,
        sample_from_sigma_delta=0.0,
        test_dl=None
        ):

    # Define the nr of noiselevels to use during training
    num_levels = [len(sigma_list)]

    # These parameters are set in order to do model selection based on the mmd and loss
    best_score = np.inf
    best_epoch = 0
    best_loss = np.inf
    best_epochloss = 0
    # Create a subdir for storing the selected models
    os.system(f"mkdir {config.model_save_dir}/best")
    os.system(f"mkdir {config.model_save_dir}/bestloss")
    os.system(f"mkdir {config.model_save_dir}/main")

    logging.info(f"{sigma_list}, {sample_from_sigma_delta}")

    # This is for storing the previous scores if we do not evaluate every epoch
    lastmmd = {}
    for noisenum in config.num_levels:
        lastmmd[noisenum] = {"degree": 0, "cluster": 0, "orbit": 0.0}

    # Set optimizer to zero
    optimizer.zero_grad()
    # Define schedular as ExpLR with th parameters given in config
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.train.lr_dacey)
    for epoch in range(max_epoch):  # range(max_epoch):
        train_losses = []
        train_loss_items = []
        test_losses = []
        test_loss_items = []
        t_start = time.time()
        model.train()

        for train_adj_b, train_x_b in train_dl:
            # train_adj_b is of size [batch_size, N, N]
            # train_x_b is of size [batch_size, N, F_i]
            train_adj_b = train_adj_b.to(config.dev)
            train_x_b = train_x_b.to(config.dev)

            train_node_flag_b = train_adj_b.sum(
                -1).gt(1e-5).to(dtype=torch.float32)
            if isinstance(sigma_list, float):
                sigma_list = [sigma_list]
            train_x_b, train_noise_adj_b, \
                train_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data(train_x_b, train_adj_b,
                                 train_node_flag_b, sigma_list, config)

            # train_noise_adj_b is of size [len(sigma_list) * batch_size, N, N]
            # train_x_b is of size [len(sigma_list) * batch_size, N, F_i]
            optimizer.zero_grad()
            score = model(x=train_x_b,
                          adjs=train_noise_adj_b,
                          node_flags=train_node_flag_b)

            loss = loss_func_bce(score.chunk(
                len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        scheduler.step(epoch)

        # The part that is commented out might not be working with the newest edp model
        model.eval()
        """
            for test_adj_b, test_x_b in test_dl:
                test_adj_b = test_adj_b.to(config.dev)
                test_x_b = test_x_b.to(config.dev)
                test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                test_x_b, test_noise_adj_b, test_node_flag_b, grad_log_q_noise_list = \
                    gen_list_of_data(test_x_b, test_adj_b,
                                    test_node_flag_b, sigma_list,config=config)
                with torch.no_grad():
                    score = model(x=test_x_b, adjs=test_noise_adj_b,
                                node_flags=test_node_flag_b)
                loss = loss_func_bce(score.chunk(len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
                #test_loss_items.append(loss_items)
                test_losses.append(loss.detach().cpu().item())"""

        try:
            mean_train_loss = np.mean(train_losses)
            mean_test_loss = np.mean(test_losses)
            mean_train_loss_item = np.mean(train_loss_items, axis=0)
            mean_train_loss_item_str = np.array2string(
                mean_train_loss_item, precision=2, separator="\t", prefix="\t")
            mean_test_loss_item = np.mean(test_loss_items, axis=0)
            mean_test_loss_item_str = np.array2string(
                mean_test_loss_item, precision=2, separator="\t", prefix="\t")
        except:
            mean_train_loss = np.mean(train_losses)
            mean_train_loss_item = np.mean(train_loss_items, axis=0)
            mean_train_loss_item_str = np.array2string(
                mean_train_loss_item, precision=2, separator="\t", prefix="\t")
            mean_test_loss_item = 0
            mean_test_loss_item_str = 0
            mean_test_loss = 0

        logging.info(f'epoch: {epoch:03d}| time: {time.time() - t_start:.1f}s| '
                     f'train loss: {mean_train_loss:+.3e} | '
                     f'test loss: {mean_test_loss:+.3e} | ')

        logging.info(f'epoch: {epoch:03d}| '
                     f'train loss i: {mean_train_loss_item_str} '
                     f'test loss i: {mean_test_loss_item_str} | ')

        if epoch % save_interval == save_interval - 1:
            to_save = {
                'model': model.state_dict(),
                'sigma_list': sigma_list,
                'config': edict2dict(config),
                'epoch': epoch,
                'train_loss': mean_train_loss,
                'test_loss': mean_test_loss,
                'train_loss_item': mean_train_loss_item,
                'test_loss_item': mean_test_loss_item,
            }
            torch.save(to_save, os.path.join(config.model_save_dir,
                                             f"{config.dataset.name}.pth"))

        if mean_train_loss < best_loss:
            best_epochloss = epoch
            best_score = mean_train_loss
            to_save = {
                'model': model.state_dict(),
                'sigma_list': sigma_list,
                'config': edict2dict(config),
                'epoch': epoch,
                'train_loss': best_score,
                'test_loss': mean_test_loss,
                'train_loss_item': mean_train_loss_item,
                'test_loss_item': mean_test_loss_item,
            }
            torch.save(to_save, os.path.join(config.model_save_dir,
                                             f"bestloss/{config.dataset.name}.pth"))

        # If conditions are met then evaluate the MMD score compared to the train set (in order to do model selection)
        if epoch % sample_interval == sample_interval - 1 and config.eval_from < epoch:
            with torch.no_grad():
                wandb_dict = {}
                for num_noiselevel in config.num_levels:
                    results = sample_testing_edp(
                        config, f"{config.model_save_dir}", epoch, num_noiselevel, train_dl)
                    wandb_dict.update({f"degree_mmd_{num_noiselevel}": results["degree"], f"cluster_mmd_{num_noiselevel}": results["cluster"],
                                      f"orbit_mmd_{num_noiselevel}": results["orbit"], f"trainloss": mean_train_loss, f"testloss": mean_test_loss})
                    lastmmd[num_noiselevel] = results
                wandb.log(wandb_dict)

            if sum([results[key] if "likelyhood" not in key else 1-results[key] for key in results.keys()]) < best_score:
                best_epoch = epoch
                best_score = sum(
                    [results[key] if "likelyhood" not in key else 1-results[key] for key in results.keys()])
                to_save = {
                    'model': model.state_dict(),
                    'sigma_list': sigma_list,
                    'config': edict2dict(config),
                    'epoch': epoch,
                    'train_loss': best_score,
                    'test_loss': mean_test_loss,
                    'train_loss_item': mean_train_loss_item,
                    'test_loss_item': mean_test_loss_item,
                }
                torch.save(to_save, os.path.join(config.model_save_dir,
                                                 f"best/{config.dataset.name}.pth"))
        else:
            wandb_dict = {}
            try:
                for num_noiselevel in config.num_levels:
                    wandb_dict.update({f"degree_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["degree"], f"cluster_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["cluster"], f"orbit_mmd_{num_noiselevel}": lastmmd[
                                      num_noiselevel]["orbit"], f"likelyhood_{num_noiselevel}": lastmmd[num_noiselevel]["likelyhood"], "trainloss": mean_train_loss, "testloss": mean_test_loss})
            except:
                for num_noiselevel in config.num_levels:
                    wandb_dict.update({f"degree_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["degree"], f"cluster_mmd_{num_noiselevel}": lastmmd[num_noiselevel][
                                      "cluster"], f"orbit_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["orbit"], "trainloss": mean_train_loss, "testloss": mean_test_loss})
            wandb.log(wandb_dict)

        # If conditions are met then evaluate the MMD score compared to the test set using the model selected based on th best mmd score
        try:
            if epoch % config.finalinterval == config.finalinterval-1 and config.eval_from < epoch:
                with torch.no_grad():
                    wandb_dict = {}
                    results = sample_main_edp(
                        config, f"{config.model_save_dir}/best", epoch, num_noiselevel)
                    wandb_dict.update({f"degree_mmd_{num_noiselevel}_best": results["degree"], f"cluster_mmd_{num_noiselevel}_best": results[
                                      "cluster"], f"orbit_mmd_{num_noiselevel}_best": results["orbit"], f"testloss_best": best_score})
                    wandb.log(wandb_dict)
        except:
            print("error")

        # If conditions are met then evaluate the MMD score compared to the test set using the model selected based on the best trainloss
        try:
            if epoch % config.finalinterval == config.finalinterval-1 and config.eval_from < epoch:
                with torch.no_grad():
                    wandb_dict = {}
                    results = sample_main_edp(
                        config, f"{config.model_save_dir}/bestloss", epoch, num_noiselevel)
                    wandb_dict.update({f"degree_mmd_{num_noiselevel}_bestloss": results["degree"], f"cluster_mmd_{num_noiselevel}_bestloss": results[
                                      "cluster"], f"orbit_mmd_{num_noiselevel}_bestloss": results["orbit"], f"testloss_bestloss": best_score})
                    wandb.log(wandb_dict)
        except:
            print("error")

        # If conditions are met then evaluate the MMD score compared to the test set using the current model
        try:
            if epoch % config.finalinterval == config.finalinterval-1:
                with torch.no_grad():
                    wandb_dict = {}
                    results = sample_main_edp(
                        config, f"{config.model_save_dir}", epoch, num_noiselevel)
                    wandb_dict.update({f"degree_mmd_{num_noiselevel}_main": results["degree"], f"cluster_mmd_{num_noiselevel}_main": results[
                                      "cluster"], f"orbit_mmd_{num_noiselevel}_main": results["orbit"], f"testloss": best_score})
                    logging.info(wandb_dict)
                    wandb.log(wandb_dict)
        except:
            print("error")


def train_main(config, args):
    config.train.sigmas = np.linspace(0, 0.5, config.num_levels[0]+1).tolist()
    set_seed_and_logger(config, args)
    train_dl, test_dl = load_data(config)
    # mc_sampler = get_mc_sampler(config)

    # Here, the `model` get `num_classes=len(sigma_list)`
    model = get_score_model(config)

    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) -
                             len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()}")
    param_string = '\n'.join(param_strings)
    logging.info(f"Parameters: \n{param_string}")
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel()
                                 for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Parameters Count: {total_params}, Trainable: {total_trainable_params}")
    optimizer = optim.Adam(model.parameters(),
                           lr=config.train.lr_init,
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.train.weight_decay)

    # Create the sigma_list which is just a list that defines the noiselevels to use
    sigma_tens = torch.linspace(0, 1/2, len(config.train.sigmas))
    sigma_list = sigma_tens.tolist()
    sigma_list.sort()

    wandb.login(key="c41e04df5bc64c8719064e73973311f58f030f3e")
    wandb.init(project="my-test-project", entity="kahefeli")

    fit(model, optimizer, None, train_dl,
        max_node_number=config.dataset.max_node_num,
        max_epoch=config.train.max_epoch,
        config=config,
        save_interval=config.train.save_interval,
        sample_interval=config.train.sample_interval,
        sigma_list=sigma_list,
        sample_from_sigma_delta=0.0,
        test_dl=test_dl
        )


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    args = parse_arguments('train_ego_small.yaml')
    ori_config_dict = get_config(args)
    config_dict = edict(ori_config_dict.copy())
    process_config(config_dict)
    config_dict.model.name = "edp-gnn"
    train_main(config_dict, args)
