import logging
import pickle
import os

from easydict import EasyDict as edict
import numpy as np
import torch

from evaluation.stats import eval_torch_batch, adjs_to_graphs, eval_graph_list
from utils.arg_helper import mkdir, set_seed_and_logger, load_data, graphs_to_tensor, load_model, parse_arguments, \
    get_config
from utils.graph_utils import discretenoise
from utils.loading_utils import get_mc_sampler, eval_sample_batch, prepare_test_model, prepare_test_model_train
from utils.visual_utils import plot_graphs_list, plot_inter_graphs, plot_inter_graphs_list


def sample_main_edp(config, modellink,epoch,noise_num):
    config.train.sigmas=np.linspace(0,0.5,config.num_levels[0]+1).tolist()
    
    train_graph_list, test_graph_list = load_data(config, get_graph_list=True)
    #mcmc_sampler = get_mc_sampler(config)
    models = prepare_test_model_train(config,modellink)
    max_node_number = config.dataset.max_node_num
    test_batch_size = config.test.batch_size
    ###optional lines here only to be used if we want to logg the progress
    #mult_stages = []

    ##initial batch will be chosen from complete noise as it the forward noise process should in theory always arrive at a completely random distribution if sampled enough times
    def gen_init_data(batch_size):
        ##chose randomly among traindata on how many nodes should be used and on how they are ordered
        rand_idx = np.random.randint(0, len(train_graph_list), batch_size)
        graph_list = [train_graph_list[i] for i in rand_idx]
        base_adjs, base_x = graphs_to_tensor(config, graph_list)
        base_adjs, base_x = base_adjs.to(config.dev), base_x.to(config.dev)
        node_flags = base_adjs.sum(-1).gt(1e-5).to(dtype=torch.float32)

        #node_flags=torch.ones_like(node_flags)
        
        ##create a matrix with p=1/2 elements at all positions Aij where i and j not masked by node_flagij=0:
        bernoulli_adj = torch.zeros(batch_size, max_node_number, max_node_number).to(config.dev)
        for k, matrix in enumerate(base_adjs):
            for i,row in enumerate(matrix):
                    for j,col in enumerate(row):
                        if 1/2 < node_flags[k][i] and 1/2 < node_flags[k][j]:
                            bernoulli_adj[k,i,j] = 1/2

        noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
        noise_lower = noise_upper.transpose(-1, -2)
        initialmatrix = noise_lower + noise_upper

        ##this line is to simply assert that all attributes are initialized with zero
        #base_x = base_x.gt(1e-5).to(dtype=torch.float32)
        base_x=torch.zeros_like(base_x)
        return initialmatrix, base_x, node_flags


    ##load model and sigma configuration
    file, sigma_list, model_params = models[0]
    model = load_model(*model_params)
    sigma_tens = torch.linspace(0,1/2,len(config.train.sigmas))
    sigma_list = sigma_tens.tolist()
    sigma_list.sort()

    ## adds noise back onto prediction
    def add_stepwise_bernoulli(init_flags, init_adjs):
        init_adjs_list = list(init_adjs.chunk(len(sigma_list), dim=0))
        init_adjs = init_adjs.chunk(len(sigma_list), dim=0)
        init_flags_list = init_flags.chunk(len(sigma_list), dim=0)

        for i in range(len(sigma_list)):
            init_adjs_list[i], noise_added = discretenoise(list(init_adjs)[i], list(init_flags_list)[i], sigma_list[i], config)
        return init_adjs_list

    ## takes a single step and returns latent at xt-1
    def take_step(noise_func, init_flags, init_adjs):
        ##we have to add noise at each lambde of different magintude
        init_adjs_list = add_stepwise_bernoulli(init_flags, init_adjs)
        init_adjs = torch.cat(init_adjs_list,dim=0)
        noise_unnormal = noise_func(init_adjs, init_flags)
        ##apply sigmoid since we must normalize this first and then go to 0 or 1 abolute values
        noise_rel = torch.sigmoid(noise_unnormal)
        ##here we take the absolute value of the noise predictions
        noise = torch.bernoulli(noise_rel).to(config.dev)
        noise=torch.triu(noise,diagonal=1) + torch.triu(noise,diagonal=1).transpose(-2,-1)
        noise_list=noise.chunk(len(sigma_list))
        inter_adjs = torch.where(noise>1/2,init_adjs-1,init_adjs)
        new_adjs = torch.where(inter_adjs < -1/2 , inter_adjs+2 , inter_adjs)
        return new_adjs.chunk(len(sigma_list), dim=0)


    ##loop through each noise level from high noise to low level noise and at each step add 
    def run_sample(eval_len=1024, methods=None):
        warm_up_count = 0
        gen_graph_list = []
        init_adjs, init_x, node_flags = gen_init_data(batch_size = test_batch_size * len(sigma_list))
        sample_node_flags=node_flags
        sample_adjs=init_adjs
        sample_x=init_x

        ##mult stages serves only to save the graphs at each noisestep
        mult_stages = [adjs_to_graphs(sample_adjs.chunk(len(sigma_list),dim=0)[-1].detach().cpu().numpy())]

        while len(gen_graph_list) < eval_len:
            with torch.no_grad():
                sample_adjs_list = take_step(lambda x, y: model(sample_x, x, y), init_flags=sample_node_flags, init_adjs=sample_adjs)                
                if warm_up_count >= len(sigma_list)-1:
                    gen_graph_list.extend(adjs_to_graphs(sample_adjs_list[0].detach().cpu().numpy()))
                    warm_up_count += 1
                else:
                    warm_up_count += 1
                
                ##logs the first batch that fully runs through all sigmalevels and saves it at every level to mult_stages list (comment out if you wish to use this)
                """
                if True:
                        if warm_up_count==1:
                            mult_stages = [adjs_to_graphs(sample_adjs_list[-warm_up_count].detach().cpu().numpy())]
                            #mult_stages.append(adjs_to_graphs(sample_adjs_list[-warm_up_count].detach().cpu().numpy()))
                            mult_stages_flags = sample_node_flags[-test_batch_size*(warm_up_count): len(sample_node_flags)-(test_batch_size*(warm_up_count-1))]"""      
                
                
                ##now we remove the sigmoidbatch that we just added or wasnt ready and has to be discarded and add a random adjs at end so that we can further iterate and generate samples
                new_sample_adjs, new_sample_x, new_node_flags = gen_init_data(batch_size=test_batch_size)
                sample_adjs = torch.cat(list(sample_adjs_list[1:]) + [new_sample_adjs], dim=0)
                ####not sure how this converts tensor to tensor and how, probably just adds the new embeddings at end
                sample_x = torch.cat([sample_x[sample_adjs_list[0].size(0):], new_sample_x], dim=0)
                sample_node_flags = torch.cat([sample_node_flags[sample_adjs_list[0].size(0):], new_node_flags], dim=0)

        ##now left with a list of genereated graphs, namely gen_graph_list only left is to plot and evaluate
        result_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=methods)
        logging.info(f'MMD_full {file} {eval_len}: {result_dict}')
        return result_dict, gen_graph_list

    results, gen_graph_list = run_sample(eval_len=config.samplesize)
    return results

## here we do the same as above however using the train set instead of the testdata
def sample_testing_edp(config, modellink,epoch,noise_num,train_dl):
    config.train.sigmas=np.linspace(0,0.5,config.num_levels[0]+1).tolist()
    train_graph_list_adj=[]
    train_graph_list_x=[]
    train_graph_list=[]
    for train_adj_b, train_x_b in train_dl:
        for adj, x  in zip(train_adj_b, train_x_b):
            train_graph_list_adj.append(adj.clone().detach())
            train_graph_list_x.append(x.clone().detach())
        train_graph_list.extend(adjs_to_graphs(train_adj_b.detach().cpu().numpy()))
    
    train_graph_list_adj=torch.stack(train_graph_list_adj)
    train_graph_list_x=torch.stack(train_graph_list_x)
    #mcmc_sampler = get_mc_sampler(config)
    models = prepare_test_model_train(config,modellink)
    max_node_number = config.dataset.max_node_num
    test_batch_size = config.test.batch_size
    ###optional lines here only to be used if we want to logg the progress
    #mult_stages = []

    ##initial batch will be chosen from complete noise as it the forward noise process should in theory always arrive at a completely random distribution if sampled enough times
    def gen_init_data(batch_size):
        ##chose randomly among traindata on how many nodes should be used and on how they are ordered
        rand_idx = np.random.randint(0, len(train_graph_list), batch_size)
        #graph_list = [train_graph_list[i] for i in rand_idx]
        base_adjs, base_x = [train_graph_list_adj[i] for i in rand_idx], [train_graph_list_x[i] for i in rand_idx]
        base_adjs, base_x = torch.stack(base_adjs), torch.stack(base_x)
        #base_adjs, base_x = train_graph_list_adj, train_graph_list_x
        base_adjs, base_x = base_adjs.to(config.dev), base_x.to(config.dev)
        node_flags = base_adjs.sum(-1).gt(1e-5).to(dtype=torch.float32)

        #node_flags=torch.ones_like(node_flags)
        
        ##create a matrix with p=1/2 elements at all positions Aij where i and j not masked by node_flagij=0:
        bernoulli_adj = torch.zeros(batch_size, max_node_number, max_node_number).to(config.dev)
        for k, matrix in enumerate(base_adjs):
            for i,row in enumerate(matrix):
                    for j,col in enumerate(row):
                        if 1/2 < node_flags[k][i] and 1/2 < node_flags[k][j]:
                            bernoulli_adj[k,i,j] = 1/2
        
        noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
        noise_lower = noise_upper.transpose(-1, -2)
        initialmatrix = noise_lower + noise_upper

        ##intialize all attributes as zeros
        base_x=torch.zeros_like(base_x)
        return initialmatrix, base_x, node_flags

    ##load model and sigma configuration
    file, sigma_list, model_params = models[0]
    model = load_model(*model_params)
    sigma_tens = torch.linspace(0,1/2,len(config.train.sigmas))
    sigma_list = sigma_tens.tolist()
    sigma_list.sort()

    def add_stepwise_bernoulli(init_flags, init_adjs):
        init_adjs_list = list(init_adjs.chunk(len(sigma_list), dim=0))
        init_adjs = init_adjs.chunk(len(sigma_list), dim=0)
        init_flags_list = init_flags.chunk(len(sigma_list), dim=0)
        
        for i in range(len(sigma_list)):
            init_adjs_list[i], noise_added = discretenoise(list(init_adjs)[i], list(init_flags_list)[i], sigma_list[i], config)
        return init_adjs_list

    def take_step(noise_func, init_flags, init_adjs):
        init_adjs_list = add_stepwise_bernoulli(init_flags, init_adjs)
        init_adjs = torch.cat(init_adjs_list,dim=0)

        noise_unnormal = noise_func(init_adjs, init_flags)
        ##apply sigmoid since we must normalize this first and then go to 0 or 1 abolute values
        noise_rel = torch.sigmoid(noise_unnormal)
        ##here we take the absolute value of the noise predictions
        #noise = torch.where(noise_rel<0.5, torch.zeros_like(noise_rel), torch.ones_like(noise_rel))
        noise = torch.bernoulli(noise_rel).to(config.dev)
        noise=torch.triu(noise,diagonal=1) + torch.triu(noise,diagonal=1).transpose(-2,-1) 
        noise_list=noise.chunk(len(sigma_list))
        
        ##now we remove the noise from init_adjs
        inter_adjs = torch.where(noise>1/2,init_adjs-1,init_adjs)
        new_adjs = torch.where(inter_adjs < -1/2 , inter_adjs+2 , inter_adjs)
        return new_adjs.chunk(len(sigma_list), dim=0)


    ##loop through each noise level from high noise to low level noise and at each step add 
    def run_sample(eval_len=1024, methods=None):
        warm_up_count = 0
        gen_graph_list = []
        init_adjs, init_x, node_flags = gen_init_data(batch_size = test_batch_size * len(sigma_list))
        
        sample_node_flags=node_flags
        sample_adjs=init_adjs
        sample_x=init_x

        mult_stages = [adjs_to_graphs(sample_adjs.chunk(len(sigma_list),dim=0)[-1].detach().cpu().numpy())]

        while len(gen_graph_list) < eval_len:

            ##disable gradient since we never call .backward() here
            with torch.no_grad():
                sample_adjs_list = take_step(lambda x, y: model(sample_x, x, y), init_flags=sample_node_flags, init_adjs=sample_adjs)
                ##now all adjs have been updated with their respective sigmoid level and returned as a LIST of tensors each list element corresp to 1 sigmoid level
                
                ## run for noiselevel times until the first matrix has gone through every noiselevel
                if warm_up_count >= len(sigma_list)-1:
                    gen_graph_list.extend(adjs_to_graphs(sample_adjs_list[0].detach().cpu().numpy()))
                    warm_up_count += 1
                else:
                    warm_up_count += 1
                
                ###########logs the first batch that fully runs through all sigmalevels and saves it at every level to mult_stages list
                """
                if True:
                        if warm_up_count==1:
                            mult_stages = [adjs_to_graphs(sample_adjs_list[-warm_up_count].detach().cpu().numpy())]
                            #mult_stages.append(adjs_to_graphs(sample_adjs_list[-warm_up_count].detach().cpu().numpy()))
                            mult_stages_flags = sample_node_flags[-test_batch_size*(warm_up_count): len(sample_node_flags)-(test_batch_size*(warm_up_count-1))]"""
                ############
                
                ##now we remove the sigmoidbatch that we just added or wasnt ready and has to be discarded and add a random adjs at end so that we can further iterate and generate samples
                new_sample_adjs, new_sample_x, new_node_flags = gen_init_data(batch_size=test_batch_size)
                sample_adjs = torch.cat(list(sample_adjs_list[1:]) + [new_sample_adjs], dim=0)

                sample_x = torch.cat([sample_x[sample_adjs_list[0].size(0):], new_sample_x], dim=0)
                sample_node_flags = torch.cat([sample_node_flags[sample_adjs_list[0].size(0):], new_node_flags], dim=0)

        ##now left with a list of genereated graphs, namely gen_graph_list only left is to plot and evaluate
        
        result_dict = eval_graph_list(train_graph_list, gen_graph_list, methods=methods)
        logging.info(f'MMD_full {file} {eval_len}: {result_dict}')
        return result_dict, gen_graph_list


    results, gen_graph_list = run_sample(eval_len=128)
    return results



if __name__ == "__main__":
    args = parse_arguments('sample_com_small_ddpm_16.yaml')
    config_dict = get_config(args)
    sample_main(config_dict, args)
        
