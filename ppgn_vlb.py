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

from sample_ppgn_vlb import sample_main, sample_testing

from utils.arg_helper import edict2dict, parse_arguments, get_config, process_config, set_seed_and_logger, load_data, graphs_to_tensor
from utils.graph_utils import gen_list_of_data_single, generate_mask
from utils.loading_utils import get_mc_sampler, get_score_model, eval_sample_batch
from utils.visual_utils import plot_graphs_adj
from model.ppgn import Powerful
from matplotlib import pyplot as plt
import wandb


#  Calculate the value of p(x_t-1 | x_t, x_0)
def posterior(sigmatilde_t,sigma_t,sigmatilde_t1,x0,xt):
    if xt<0.01 and x0<0.01:
        return sigmatilde_t1 * sigma_t / (1-sigmatilde_t)
    elif xt>0.99 and x0<0.01:
        return sigmatilde_t1 * (1-sigma_t) / (sigmatilde_t)
    elif xt>0.99 and x0>0.99:
        return (1-sigmatilde_t1) * (1-sigma_t) / (1-sigmatilde_t)
    elif xt<0.01 and x0>0.99:
        return (1-sigmatilde_t1) * sigma_t / (sigmatilde_t)


#  Given the list of sigmas used, return a list of sigmas
def sigma_lin(sigma_list):
    sigmas=[]
    for g,sigma in enumerate(sigma_list):
        if sigma<0.0000000001:
            sigmas.append(0.0)
            continue
        sigmas.append(((1-sigma)-(1-sigma_list[g-1]))/(1-2*(1-sigma_list[g-1])))
    return sigmas


#  A function for getting the loss compared to the training set (only used for the model selection)
def eval_loss(eval_set, num_levels, config, model):
    sigma_ind_list=np.array(range(1,num_levels[0]+1))

    if not config.linear:
        sigma_ind_list=np.array(range(1,num_levels[0]+1))
        c=torch.tensor(range(0,num_levels[0]+1))
        c=c*(0.5*np.pi/num_levels[0])
        c=torch.cos(c)
        sigma_line = 0.5 - 0.5 * c
    else:
        sigma_line=torch.linspace(0,1/2,num_levels[0]+1).tolist()

    sigma_list = [sigma_line[i] for i in sigma_ind_list]
    sig_list = sigma_lin(sigma_line)
    sigma_nontild_list = [sig_list[i] for i in sigma_ind_list]

    # Eval set is list of size 32 x 2 x tensor(n x n)
    loss=0.0
    for eval_adj_b, eval_x_b in eval_set:
        adjs=eval_adj_b.repeat(num_levels[0],1,1).to(config.dev)
        xs=eval_x_b.repeat(num_levels[0],1,1).to(config.dev)
        flags = adjs.sum(-1).gt(1e-5).to(dtype=torch.float32).to(config.dev)
        # adjs now tensor of size num_levels x n x n
        eval_x_b, eval_noise_adj_b, \
                eval_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data_single(xs, adjs,
                                 flags, sigma_list, config)

        eval_noise_adj_b_chunked = eval_noise_adj_b.chunk(len(sigma_list), dim=0)
        eval_adj_b_chunked = adjs.clone().chunk(len(sigma_list), dim=0)
        eval_node_flag_b = flags.chunk(len(sigma_list), dim=0)
        score=[]
        masks=[]
        for i, sigma in enumerate(sigma_list):
            mask = generate_mask(eval_node_flag_b[i]).to(config.dev)
            score_batch=model(A=eval_noise_adj_b_chunked[i].unsqueeze(0).to(config.dev),node_features=eval_noise_adj_b_chunked[i].to(config.dev),mask=mask.to(config.dev),noiselevel=sigma).to(config.dev)
            score.append(score_batch)
            masks.append(mask)
        score=torch.cat(score,dim=0).squeeze(-1).to(config.dev)
        masktens=torch.cat(masks,dim=0).to(config.dev)
        loss += loss_func_kld(score, torch.stack(eval_noise_adj_b_chunked).to(config.dev), adjs.to(config.dev), torch.stack(grad_log_q_noise_list), sigma_list,sigma_ind_list,sigma_nontild_list, config, masktens)
    return loss



def loss_func_kld(score_list, train_noise_adj_b, train_adj_b, grad_log_q_noise_list, sigma_list, sigma_ind_list,sigma_nontild_list, config, mask):
    loss=0.0
    kl_loss = nn.KLDivLoss(reduction="none")

    # Need to compute wether switch would go to on or to off (since model just predicts if we switched and not in which direction)
    for i,adj in enumerate(train_noise_adj_b):
        sigmatilde_t=sigma_list[i]
        sigma_t=sigma_nontild_list[i]

        sigmatilde_t1=sigma_list[i]-sigma_list[i]/sigma_ind_list[i]
        # Compute q which is the posterior on each matrix element but with knowing x0 and knowing xt which means we need both as arguments
        mult1=torch.where(train_noise_adj_b[i]>1/2,(1-sigma_t),sigma_t)
        mult2=torch.where(train_adj_b[i]>1/2,1-sigmatilde_t1,sigmatilde_t1)
        xor=torch.logical_xor(train_noise_adj_b[i], train_adj_b[i])
        div=torch.where(xor>1/2,sigmatilde_t,1-sigmatilde_t)
        q=mult1*mult2/div
        

        # Change score list based on if xt is 0 or 1 
        score_i=torch.where(train_noise_adj_b[i]>1/2,1-torch.sigmoid(score_list[i]),torch.sigmoid(score_list[i]))
        # score list represents p(x0|xt)
        
        # Calculate posterior(sigmatilde_t,sigma_t,sigmatilde_t1,0,xt)
        mult1=torch.where(train_noise_adj_b[i]>1/2,(1-sigma_t),sigma_t)
        mult2=torch.where(torch.zeros_like(train_adj_b[i])>1/2,1-sigmatilde_t1,sigmatilde_t1)
        xor=torch.logical_xor(train_noise_adj_b[i], torch.zeros_like(train_adj_b[i]))
        div=torch.where(xor>1/2,sigmatilde_t,1-sigmatilde_t)
        p = ( 1 - score_i ) * mult1*mult2/div

        # Calculate posterior(sigmatilde_t,sigma_t,sigmatilde_t1,1,xt)
        mult1=torch.where(train_noise_adj_b[i]>1/2,(1-sigma_t),sigma_t)
        mult2=torch.where(torch.ones_like(train_adj_b[i])>1/2,1-sigmatilde_t1,sigmatilde_t1)
        xor=torch.logical_xor(train_noise_adj_b[i], torch.ones_like(train_adj_b[i]))
        div=torch.where(xor>1/2,sigmatilde_t,1-sigmatilde_t)
        p += ( score_i ) * mult1 * mult2/div

        # p stands for probablity p(x0=1|xt=xt) now
        score_list[i] = p
        # This q is q(x0=1|xt=xt,x0=x0)
        grad_log_q_noise_list[i] = q

        score_inv = 1-score_list[i]
        score_list_twoclass=torch.cat([score_list[i].unsqueeze(-1),score_inv.unsqueeze(-1)],-1)
        grad_inv=1-grad_log_q_noise_list[i]
        grad_log_q_noise_list_twoclass=torch.cat([grad_log_q_noise_list[i].unsqueeze(-1),grad_inv.unsqueeze(-1)],-1)
        loss_matrix=kl_loss(torch.log(score_list_twoclass),grad_log_q_noise_list_twoclass).to(config.dev)
        
        loss_matrix=loss_matrix.sum(-1)
        loss_matrix=(loss_matrix+torch.transpose(loss_matrix, -2, -1))/2
        loss_matrix=loss_matrix.to(config.dev) * mask[i].to(config.dev)
        # Exclude the diagonal elements
        loss_matrix = torch.triu(loss_matrix,diagonal=1) + torch.tril(loss_matrix,diagonal= -1)
        loss += loss_matrix.sum()
    return loss

def fit(model, optimizer, mcmc_sampler, train_dl, max_node_number, max_epoch=20, config=None,
        save_interval=50,
        sample_interval=1,
        sigma_length=None,
        sample_from_sigma_delta=0.0,
        test_dl=None,
        eval_set=None
        ):

        # These parameters are set in order to do model selection based on the mmd and loss
        best_score=np.inf
        best_score_loss=np.inf
        best_score_loss_eval=np.inf
        best_epoch=0
        best_epoch_loss=0
        best_epoch_loss_eval=0
        # Create a subdir for storing the selected models
        os.system(f"mkdir {config.model_save_dir}/best")
        os.system(f"mkdir {config.model_save_dir}/bestloss")
        os.system(f"mkdir {config.model_save_dir}/bestloss_eval")

        # This is for storing the previous scores if we do not evaluate every epoch
        lastmmd={}
        for noisenum in config.num_levels:
            lastmmd[noisenum]={"degree": 0, "cluster": 0, "orbit": 0.0}

        # Set optimizer to zero slope
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
                # Here sample the noiselevels randomly from th scheduled levels 
                # sigma_ind_list is a list of random indexes which defines which noiselevel to use for which graph
                sigma_ind_list = np.random.random_integers(low=1,high=config.num_levels[0],size=train_adj_b.size(0))
                # sigma_line represents the linear distributed noiselevels (in paper equivalent to the Beta_tildes or Beta_overlines), so the noise from x0 to xt
                sigma_line=torch.linspace(0,1/2,config.num_levels[0]+1).tolist()
                # sigma_list represents the randomly chosen Beta_overline for each graph 
                sigma_list = [sigma_line[i] for i in sigma_ind_list]
                # sig_list represents the corresponding Betas (NOT beta_overlines), so noise from xt-1 to xt
                sig_list = sigma_lin(sigma_line)
                sigma_nontild_list = [sig_list[i] for i in sigma_ind_list]

                train_adj_b = train_adj_b.to(config.dev)
                train_x_b = train_x_b.to(config.dev)
                train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)

                if isinstance(sigma_list, float):
                    sigma_list = [sigma_list]
                train_x_b, train_noise_adj_b, \
                train_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data_single(train_x_b, train_adj_b,
                                 train_node_flag_b, sigma_list, config)
                # Now we have tensor of size B x N x N and grad_log_q_noise_list is list of B x Tensor( N x N )

                optimizer.zero_grad()

                train_noise_adj_b_chunked = train_noise_adj_b.chunk(len(sigma_list), dim=0)
                train_adj_b_chunked = train_adj_b.clone().chunk(len(sigma_list), dim=0)
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

                # Compute loss for this epoch
                loss = loss_func_kld(score, torch.stack(train_noise_adj_b_chunked), train_adj_b, torch.stack(grad_log_q_noise_list), sigma_list,sigma_ind_list,sigma_nontild_list, config, masktens)

                # Take step on gradient
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().item())
            scheduler.step(epoch)
            
            # Do the same for test set to get testloss
            model.eval()
            for test_adj_b, test_x_b in test_dl:
                test_adj_b = test_adj_b.to(config.dev)
                test_x_b = test_x_b.to(config.dev)
                test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                sigma_ind_list = np.random.random_integers(low=1,high=config.num_levels[0],size=test_adj_b.size(0))
                sigma_line=np.linspace(0,1/2,config.num_levels[0]+1).tolist()
                sigma_list = [sigma_line[i] for i in sigma_ind_list]
                sig_list = sigma_lin(np.linspace(0,1/2,config.num_levels[0]+1).tolist())
                sigma_nontild_list = [sig_list[i] for i in sigma_ind_list]
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
                    loss = loss_func_kld(score, torch.stack(test_noise_adj_b_chunked), test_adj_b, torch.stack(grad_log_q_noise_list), sigma_list,sigma_ind_list,sigma_nontild_list, config, masktens)
                test_losses.append(loss.detach().cpu().item())

            try:
                mean_train_loss = np.mean(train_losses)
                mean_test_loss = np.mean(test_losses)
                mean_train_loss_item = np.mean(train_loss_items, axis=0)
                mean_train_loss_item_str = np.array2string(mean_train_loss_item, precision=2, separator="\t", prefix="\t")
                mean_test_loss_item = np.mean(test_loss_items, axis=0)
                mean_test_loss_item_str = np.array2string(mean_test_loss_item, precision=2, separator="\t", prefix="\t")
            except:
                mean_train_loss = np.mean(train_losses)
                mean_test_loss = 0.0
                mean_train_loss_item = np.mean(train_loss_items, axis=0)
                mean_train_loss_item_str = np.array2string(mean_train_loss_item, precision=2, separator="\t", prefix="\t")
                mean_test_loss_item = 0.0
                mean_test_loss_item_str = 0.0

            # Save the model at epoch eval_from
            if epoch == config.eval_from:
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

            # Store model in general every x iterations
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

            # If conditions are met then evaluate the trainloss
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
            
            # If conditions are met then evaluate the eval_loss (which is similar to trainloss however taken over a set of graphs where each noisestep applied to every graph instead of randomly sampling 1 noiselevel per graph)
            if epoch % sample_interval == sample_interval - 1 and config.eval_from<epoch:
                loss_eval=eval_loss(eval_set,config.num_levels,config,model)
                if loss_eval<best_score_loss_eval:
                        best_epoch_loss_eval = epoch
                        best_score_loss_eval = loss_eval
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
                                                        f"bestloss_eval/{config.dataset.name}.pth"))
            else:
                loss_eval=0


            logging.info(f'epoch: {epoch:03d}| time: {time.time() - t_start:.1f}s| '
                        f'train loss: {mean_train_loss:+.3e} | '
                        f'test loss: {mean_test_loss:+.3e} | ')

            logging.info(f'epoch: {epoch:03d}| '
                        f'train loss i: {mean_train_loss_item_str} '
                        f'test loss i: {mean_test_loss_item_str} | ')
            

            
            
            if epoch % sample_interval == sample_interval - 1 and config.eval_from<epoch:
                with torch.no_grad():
                    wandb_dict={}
                    results=sample_testing(config,f"{config.model_save_dir}",epoch,num_noiselevel,train_dl)
                    wandb_dict.update({f"degree_mmd_{num_noiselevel}": results["degree"],f"cluster_mmd_{num_noiselevel}": results["cluster"],f"orbit_mmd_{num_noiselevel}": results["orbit"],f"trainloss": mean_train_loss,f"testloss": mean_test_loss,f"evalloss": loss_eval})
                    lastmmd[num_noiselevel]=results
                    wandb.log(wandb_dict)

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
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["degree"],f"cluster_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["cluster"],f"likelyhood_{num_noiselevel}": lastmmd[num_noiselevel]["likelyhood"],"trainloss": mean_train_loss,"testloss": mean_test_loss,f"evalloss": loss_eval})
                except: 
                    for num_noiselevel in config.num_levels:
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["degree"],f"cluster_mmd_{num_noiselevel}": lastmmd[num_noiselevel]["cluster"],"trainloss": mean_train_loss,"testloss": mean_test_loss,f"evalloss": loss_eval})
                wandb.log(wandb_dict)


            try:      
                if epoch%config.finalinterval==config.finalinterval-1 and config.eval_from<epoch:
                    with torch.no_grad():
                        wandb_dict={}
                        results=sample_main(config,f"{config.model_save_dir}/best",epoch,num_noiselevel)
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}_best": results["degree"],f"cluster_mmd_{num_noiselevel}_best": results["cluster"],f"orbit_mmd_{num_noiselevel}_best": results["orbit"],f"testloss_best": mean_test_loss})
                        wandb.log(wandb_dict)
            except Exception as e:
                print(e)

            try:      
                if epoch%config.finalinterval==config.finalinterval-1:
                    with torch.no_grad():
                        wandb_dict={}
                        results=sample_main(config,f"{config.model_save_dir}",epoch,num_noiselevel)
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}_main": results["degree"],f"cluster_mmd_{num_noiselevel}_main": results["cluster"],f"orbit_mmd_{num_noiselevel}_main": results["orbit"],f"testloss_best": mean_test_loss})
                        logging.info(wandb_dict)
                        wandb.log(wandb_dict)
            except Exception as e:
                print(e)

            try:      
                if epoch%config.finalinterval==config.finalinterval-1 and config.eval_from<epoch:
                    with torch.no_grad():
                        wandb_dict={}
                        results=sample_main(config,f"{config.model_save_dir}/bestloss",epoch,num_noiselevel)
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}_bestloss": results["degree"],f"cluster_mmd_{num_noiselevel}_bestloss": results["cluster"],f"orbit_mmd_{num_noiselevel}_bestloss": results["orbit"],f"testloss_bestloss": best_score_loss})
                        wandb.log(wandb_dict)
            except Exception as e:
                print(e)

            try:      
                if epoch%config.finalinterval==config.finalinterval-1 and config.eval_from<epoch:
                    with torch.no_grad():
                        wandb_dict={}
                        results=sample_main(config,f"{config.model_save_dir}/bestloss_eval",epoch,num_noiselevel)
                        wandb_dict.update({f"degree_mmd_{num_noiselevel}_bestloss_eval": results["degree"],f"cluster_mmd_{num_noiselevel}_bestloss_eval": results["cluster"],f"orbit_mmd_{num_noiselevel}_bestloss_eval": results["orbit"],f"train_bestloss_eval": best_score_loss_eval})
                        wandb.log(wandb_dict)
            except Exception as e:
                print(e)


        

def train_main(config, args):
    config.train.sigmas=np.linspace(0,0.5,config.num_levels[0]+1).tolist()
    set_seed_and_logger(config, args)
    train_dl, test_dl = load_data(config)
    # mc_sampler = get_mc_sampler(config)
    # Here, the `model` get `num_classes=len(sigma_list)`
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

    wandb.login(key="")
    wandb.init(project="train_ppgn_consec_gridsearch", entity="khaefeli",config=config)
    sigma_list = len(config.train.sigmas)

    # Select 32 random Graphs from the dataloader 
    train_graph_list, test_graph_list = load_data(config, get_graph_list=True)
    # rand_idx = np.random.randint(0, len(train_graph_list), 32)
    rand_idx = np.random.randint(0, len(train_graph_list), 64)
    eval_graph_list = [train_graph_list[i] for i in rand_idx]
    eval_adjs, eval_x = graphs_to_tensor(config, eval_graph_list)
    eval_set = list(zip(eval_adjs, eval_x))

    fit(model, optimizer, None, train_dl,
        max_node_number=config.dataset.max_node_num,
        max_epoch=config.train.max_epoch,
        config=config,
        save_interval=config.train.save_interval,
        sample_interval=config.train.sample_interval,
        sigma_length=sigma_list,
        sample_from_sigma_delta=0.0,
        test_dl=test_dl,
        eval_set=eval_set
        )

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    args = parse_arguments('train_ego_small.yaml')
    ori_config_dict = get_config(args)
    config_dict = edict(ori_config_dict.copy())
    process_config(config_dict)
    config_dict.model.name = "ppgn"
    train_main(config_dict, args)