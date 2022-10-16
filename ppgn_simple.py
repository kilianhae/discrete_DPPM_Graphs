import logging
import time
import os
import sys
import graph_tool.all as gt
from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation.stats import eval_torch_batch

from model.langevin_mc import LangevinMCSampler
from sample_ppgn_simple import sample_main, sample_testing

from utils.arg_helper import edict2dict, parse_arguments, get_config, process_config, set_seed_and_logger, load_data
from utils.graph_utils import gen_list_of_data_single, generate_mask
from utils.loading_utils import get_mc_sampler, get_score_model, eval_sample_batch
from utils.visual_utils import plot_graphs_adj
from model.ppgn import Powerful
from matplotlib import pyplot as plt
import wandb


def loss_func_bce(score_list, grad_log_q_noise_list, sigma_list, config, mask):
    loss=0.0
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_matrix = BCE(score_list,grad_log_q_noise_list)
    loss_matrix = loss_matrix * (1-2*torch.tensor(sigma_list).unsqueeze(-1).unsqueeze(-2).expand(grad_log_q_noise_list.size(0),grad_log_q_noise_list.size(1),grad_log_q_noise_list.size(2)).to(config.dev)+1.0/len(sigma_list))
    ##loss analogue to https://arxiv.org/pdf/2111.12701.pdf
    
    loss_matrix=(loss_matrix+torch.transpose(loss_matrix, -2, -1))/2
    loss_matrix=loss_matrix * mask
    #loss = loss_matrix.sum()
    loss = torch.mean(loss_matrix)
    return loss


def fit(model, optimizer, mcmc_sampler, train_dl, max_node_number, max_epoch=20, config=None,
        save_interval=50,
        sample_interval=1,
        sigma_length=None,
        sample_from_sigma_delta=0.0,
        test_dl=None
        ):
        
        best_score=np.inf
        best_score_loss=np.inf
        best_epoch=0
        best_epoch_loss=0
        os.system(f"mkdir {config.model_save_dir}/best")
        os.system(f"mkdir {config.model_save_dir}/bestloss")

        lastmmd={}
        for noisenum in config.num_levels:
            lastmmd[noisenum]={"degree": 0, "cluster": 0, "orbit": 0.0}
        resultlist=[]
        optimizer.zero_grad()
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
        for epoch in range(max_epoch):
            train_losses = []
            train_loss_items = []
            test_losses = []
            test_loss_items = []
            t_start = time.time()

            model.train()
            for train_adj_b, train_x_b in train_dl:
                ##here sample the noiselevels randomly from 0 to 0.5
                sigma_list=list(np.random.uniform(low=0.0, high=0.5, size=train_adj_b.size(0)))
                train_adj_b = train_adj_b.to(config.dev)
                train_x_b = train_x_b.to(config.dev)
                train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                if isinstance(sigma_list, float):
                    sigma_list = [sigma_list]
                train_x_b, train_noise_adj_b, \
                train_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data_single(train_x_b, train_adj_b,
                                 train_node_flag_b, sigma_list, config)
                ##now we have tensor of size B x N x N and grad_log_q_noise_list is list of B x Tensor( N x N )
                optimizer.zero_grad()
                train_noise_adj_b_chunked = train_noise_adj_b.chunk(len(sigma_list), dim=0)
                train_adj_b_chunked = train_adj_b.chunk(len(sigma_list), dim=0)
                train_node_flag_b = train_node_flag_b.chunk(len(sigma_list), dim=0)
                score=[]
                masks=[]
                for i, sigma in enumerate(sigma_list):
                    mask = generate_mask(train_node_flag_b[i])
                    score_batch=model(A=train_noise_adj_b_chunked[i].unsqueeze(0).to(config.dev),node_features=train_noise_adj_b_chunked[i].to(config.dev),mask=mask.to(config.dev),noiselevel=sigma).to(config.dev)
                    score.append(score_batch)
                    masks.append(mask)
                score=torch.cat(score,dim=0).squeeze(-1).to(config.dev)
                masktens=torch.cat(masks,dim=0).to(config.dev)
                loss = loss_func_bce(score, torch.stack(grad_log_q_noise_list), sigma_list,config,masktens)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().item())
            scheduler.step(epoch)

            model.eval()
            for test_adj_b, test_x_b in test_dl:
                test_adj_b = test_adj_b.to(config.dev)
                test_x_b = test_x_b.to(config.dev)
                test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                sigma_list=list(np.random.uniform(low=0.0, high=0.5, size=test_adj_b.size(0)))
                test_x_b, test_noise_adj_b, test_node_flag_b, grad_log_q_noise_list = \
                    gen_list_of_data_single(test_x_b, test_adj_b,
                                    test_node_flag_b, sigma_list,config=config)
                with torch.no_grad():
                    test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
                    test_adj_b_chunked = test_adj_b.chunk(len(sigma_list), dim=0)
                    test_node_flag_b = test_node_flag_b.chunk(len(sigma_list), dim=0)
                    score=[]
                    masks=[]
                    for i, sigma in enumerate(sigma_list):
                        mask = generate_mask(test_node_flag_b[i])
                        score_batch=model(A=test_noise_adj_b_chunked[i].unsqueeze(0).to(config.dev),node_features=test_noise_adj_b_chunked[i].to(config.dev),mask=mask.to(config.dev),noiselevel=sigma).to(config.dev)
                        masks.append(mask)
                        score.append(score_batch)
                    score=torch.cat(score,dim=0).squeeze(-1).to(config.dev)
                    masktens=torch.cat(masks,dim=0).to(config.dev)
                    ###here changed so that loss just gets tensor of size B x N x N
                    loss = loss_func_bce(score, torch.stack(grad_log_q_noise_list), sigma_list,config,masktens)
                test_losses.append(loss.detach().cpu().item())

            mean_train_loss = np.mean(train_losses)
            mean_test_loss = np.mean(test_losses)
            mean_train_loss_item = np.mean(train_loss_items, axis=0)
            mean_train_loss_item_str = np.array2string(mean_train_loss_item, precision=2, separator="\t", prefix="\t")
            mean_test_loss_item = np.mean(test_loss_items, axis=0)
            mean_test_loss_item_str = np.array2string(mean_test_loss_item, precision=2, separator="\t", prefix="\t")

            logging.info(f'epoch: {epoch:03d}| time: {time.time() - t_start:.1f}s| '
                        f'train loss: {mean_train_loss:+.3e} | '
                        f'test loss: {mean_test_loss:+.3e} | ')

            logging.info(f'epoch: {epoch:03d}| '
                        f'train loss i: {mean_train_loss_item_str} '
                        f'test loss i: {mean_test_loss_item_str} | ')
            
            ##save current model
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
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                torch.save(to_save, os.path.join(config.model_save_dir,
                                                f"{config.dataset.name}.pth"))

            ## model selection based on trainloss
            if mean_train_loss<best_score_loss:
                    best_epoch_loss = epoch
                    best_score_loss = mean_train_loss
                    to_save = {
                        'model': model.state_dict(),
                        'sigma_list': sigma_list,
                        'config': edict2dict(config),
                        'epoch': epoch,
                        'train_loss': best_score,
                        'test_loss': mean_test_loss,
                        'train_loss_item': mean_train_loss_item,
                        'test_loss_item': mean_test_loss_item,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    torch.save(to_save, os.path.join(config.model_save_dir,
                                                    f"bestloss/{config.dataset.name}.pth"))

            logging.info(f'epoch: {epoch:03d}| time: {time.time() - t_start:.1f}s| '
                        f'train loss: {mean_train_loss:+.3e} | '
                        f'test loss: {mean_test_loss:+.3e} | ')

            logging.info(f'epoch: {epoch:03d}| '
                        f'train loss i: {mean_train_loss_item_str} '
                        f'test loss i: {mean_test_loss_item_str} | ')

            ## generate graphs based on current model and do model selection based on the mmd score compared to the train set
            if epoch % sample_interval == sample_interval - 1 and config.eval_from<epoch:
                with torch.no_grad():
                    wandb_dict={}
                    if False:#"sbm" in config.dataset.name:
                        for num_noiselevel in config.num_levels:
                            print(f"{config.model_save_dir}")
                            if "density" in config.noisetype:
                                results=sample_testing_train_density(config,f"{config.model_save_dir}",epoch,num_noiselevel,train_dl)
                            else:
                                results=sample_testing_train(config,f"{config.model_save_dir}",epoch,num_noiselevel,train_dl)
                            wandb_dict.update({f"degree_mmd_{num_noiselevel}": results["degree"],f"cluster_mmd_{num_noiselevel}": results["cluster"],f"orbit_mmd_{num_noiselevel}": results["orbit"],f"likelyhood_{num_noiselevel}": results["likelyhood"],f"trainloss": mean_train_loss,f"testloss": mean_test_loss})
                            lastmmd[num_noiselevel]=results
                    else:

                        for num_noiselevel in config.num_levels:
                            if "density" in config.noisetype:
                                results=sample_testing_density(config,f"{config.model_save_dir}",epoch,num_noiselevel,train_dl)
                            else:

                                results=sample_testing(config,f"{config.model_save_dir}",epoch,num_noiselevel,train_dl)
                                print(results)
                                print("orbitresults")
                            wandb_dict.update({f"degree_mmd_{num_noiselevel}": results["degree"],f"cluster_mmd_{num_noiselevel}": results["cluster"],f"orbit_mmd_{num_noiselevel}": results["orbit"],f"trainloss": mean_train_loss,f"testloss": mean_test_loss})
                            lastmmd[num_noiselevel]=results
                    wandb.log(wandb_dict)

                ## model selection based on the found mmd scores
                if sum([results[key] if "likelyhood" not in key else 1-results[key] for key in results.keys()])<best_score:
                    best_epoch = epoch
                    best_score = sum([results[key] if "likelyhood" not in key else 1-results[key] for key in results.keys()])
                    to_save = {
                        'model': model.state_dict(),
                        'sigma_list': sigma_list,
                        'config': edict2dict(config),
                        'epoch': epoch,
                        'train_loss': best_score,
                        'test_loss': mean_test_loss,
                        'train_loss_item': mean_train_loss_item,
                        'test_loss_item': mean_test_loss_item,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    torch.save(to_save, os.path.join(config.model_save_dir,
                                                    f"best/{config.dataset.name}.pth"))
            else:
                wandb_dict={}
                try:
                    for num_noiselevel in config.num_levels:
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["degree"],f"cluster_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["cluster"],f"orbit_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["orbit"],f"likelyhood_{num_noiselevel}": lastmmd[num_noiselevel]["likelyhood"],"trainloss": mean_train_loss,"testloss": mean_test_loss})
                except: 
                    for num_noiselevel in config.num_levels:
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["degree"],f"cluster_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["cluster"],f"orbit_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["orbit"],"trainloss": mean_train_loss,"testloss": mean_test_loss})
                wandb.log(wandb_dict)


            ## test the selected model with best trainloss
            try:      
                if epoch%config.finalinterval==config.finalinterval-1 and config.eval_from<epoch:
                    with torch.no_grad():
                        wandb_dict={}
                        results=sample_main(config,f"{config.model_save_dir}/bestloss",epoch,num_noiselevel)
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}_bestloss": results["degree"],f"cluster_mmd_{num_noiselevel}_bestloss": results["cluster"],f"orbit_mmd_{num_noiselevel}_bestloss": results["orbit"],f"testloss_bestloss": best_score_loss})
                        wandb.log(wandb_dict)
            except Exception as e:
                print("error in besloss")
                print(e)
            
            ## test the model without modelselection
            try:      
                if epoch%config.finalinterval==config.finalinterval-1:
                    with torch.no_grad():
                        wandb_dict={}
                        results=sample_main(config,f"{config.model_save_dir}",epoch,num_noiselevel)
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}_main": results["degree"],f"cluster_mmd_{num_noiselevel}_main": results["cluster"],f"orbit_mmd_{num_noiselevel}_main": results["orbit"],f"testloss_bestloss": best_score_loss})
                        logging.info(wandb_dict)
                        wandb.log(wandb_dict)
            except Exception as e:
                print("error in main")
                print(e)
            
            ## test the selected model with best train-mmd score
            try:      
                if epoch%config.finalinterval==config.finalinterval-1 and config.eval_from<epoch:
                    with torch.no_grad():
                        wandb_dict={}
                        results=sample_main(config,f"{config.model_save_dir}/best",epoch,num_noiselevel)
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}_best": results["degree"],f"cluster_mmd_{num_noiselevel}_best": results["cluster"] })
                        wandb.log(wandb_dict)
            except Exception as e:
                print("error in best")
                print(e)

        
def train_main(config, args):
    config.train.sigmas=np.linspace(0,0.5,config.num_levels[0]+1).tolist()
    set_seed_and_logger(config, args)
    train_dl, test_dl = load_data(config)
    #mc_sampler = get_mc_sampler(config)
    # here, the `model` get `num_classes=len(sigma_list)`
    model = get_score_model(config)

    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()}")
    param_string = '\n'.join(param_strings)
    logging.info(f"Parameters: \n{param_string}")
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters Count: {total_params}, Trainable: {total_trainable_params}")
    optimizer = optim.Adam(model.parameters(),
                           lr=config.train.lr_init,
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.train.weight_decay)

    wandb.login(key="c41e04df5bc64c8719064e73973311f58f030f3e")
    wandb.init(project="train_ppgn_consec_gridsearch", entity="khaefeli",config=config)
    sigma_list = len(config.train.sigmas)
    
    fit(model, optimizer, None, train_dl,
        max_node_number=config.dataset.max_node_num,
        max_epoch=config.train.max_epoch,
        config=config,
        save_interval=config.train.save_interval,
        sample_interval=config.train.sample_interval,
        sigma_length=sigma_list,
        sample_from_sigma_delta=0.0,
        test_dl=test_dl
        )

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    args = parse_arguments('train_ego_small.yaml')
    ori_config_dict = get_config(args)
    config_dict = edict(ori_config_dict.copy())
    process_config(config_dict)
    config_dict.model.name = "ppgn"
    print(config_dict)
    train_main(config_dict, args)