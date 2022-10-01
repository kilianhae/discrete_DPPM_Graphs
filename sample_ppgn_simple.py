import logging
import pickle
import os

from easydict import EasyDict as edict
import numpy as np
import torch

from evaluation.stats import eval_torch_batch, adjs_to_graphs, eval_graph_list
from utils.arg_helper import mkdir, set_seed, load_data, graphs_to_tensor, load_model, parse_arguments, \
    get_config
from utils.graph_utils import discretenoise, generate_mask, discretenoise_balanced
from utils.loading_utils import get_mc_sampler, eval_sample_batch, prepare_test_model_train
from utils.visual_utils import plot_graphs_list, plot_inter_graphs, plot_inter_graphs_list



def sample_main(config,modellink,epoch,noise_num):
    train_graph_list, test_graph_list = load_data(config, get_graph_list=True)
    models = prepare_test_model_train(config,modellink)
    max_node_number = config.dataset.max_node_num
    test_batch_size = config.test.batch_size

    ## generate initial noise data. Each graph has a fixed nr of nodes sampled from the data and then each edge is on or off with p=0.5
    def gen_init_data(batch_size):
        rand_idx = np.random.randint(0, len(train_graph_list), batch_size)
        graph_list = [train_graph_list[i] for i in rand_idx]
        base_adjs, base_x = graphs_to_tensor(config, graph_list)
        base_adjs, base_x = base_adjs.to(config.dev), base_x.to(config.dev)
        node_flags = base_adjs.sum(-1).gt(1e-5).to(dtype=torch.float32)

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
        return initialmatrix, base_x, node_flags
        ##returns initialmatrix = tensor of size batchsize x N x N


    file, sigma_list, model_params = models[0]
    model = load_model(*model_params)
    sigma_tens = torch.linspace(0,1/2,noise_num)
    sigma_list = sigma_tens.tolist()
    sigma_list.sort()

    ##adds bernoulli noise 
    def add_bernoulli(flags, init_adjs, noiselevel):
        ##deprecated noisetype balanced
        """
        if config.noisetype=="balanced":
            init_adjs, noise_added = discretenoise_balanced(init_adjs, flags, noiselevel, config)
        else:
            init_adjs, noise_added = discretenoise(init_adjs, flags, noiselevel, config)"""

        init_adjs, noise_added = discretenoise(init_adjs, flags, noiselevel, config)
        return init_adjs


    def take_step(noise_func, flags, init_adjs, noiselevel):
        init_adjs = add_bernoulli(flags, init_adjs, noiselevel)
        mask=generate_mask(flags).to(config.dev)
        noise_unnormal = noise_func(A=init_adjs.to(config.dev),feat=None,mask=mask.to(config.dev),noise=noiselevel)
        noise_unnormal = noise_unnormal.squeeze(-1)
        noise_rel = torch.sigmoid(noise_unnormal)
        noise_rel = (noise_rel+torch.transpose(noise_rel,-2,-1))/2
        noise_rel = noise_rel*mask

        ##sample from predicted reverse distribution
        noise = torch.bernoulli(noise_rel)
        noise=torch.triu(noise,diagonal=1) + torch.triu(noise,diagonal=1).transpose(-2,-1)

        inter_adjs = torch.where(noise>1/2,init_adjs-1,init_adjs)
        new_adjs = torch.where(inter_adjs < -1/2 , inter_adjs+2 , inter_adjs)
        return new_adjs


    def run_sample(eval_len=10, methods=None):
        gen_graph_list = []
        with torch.no_grad():
            while len(gen_graph_list)<eval_len:
                count=0
                init_adjs, init_x, flags = gen_init_data(batch_size = test_batch_size)

                ## enable this if you wish to track the graphs in between the noiselevels
                #mult_stages = [adjs_to_graphs(init_adjs.detach().cpu().numpy())]
                #mult_stages_flags = flags[-test_batch_size*(0+1): len(flags)-(test_batch_size*(0))]

                while count<len(sigma_list):
                    noiselevel=sigma_list[len(sigma_list)-count-1]
                    init_adjs=take_step(lambda feat, A, mask, noise: model(feat, A, mask, noise), flags=flags, init_adjs=init_adjs, noiselevel=noiselevel)
                    count=count+1

                    ## enable this if you wish to track the graphs in between the noiselevels
                    #mult_stages.append(adjs_to_graphs(init_adjs.detach().cpu().numpy()))
                    #mult_stages_flags = torch.cat((mult_stages_flags, flags[-test_batch_size*(count): len(flags)-(test_batch_size*(count-1))]),0)

                gen_graph_list.extend(adjs_to_graphs(init_adjs.detach().cpu().numpy()))

        ##plot a set of the predicted graphs
        pic_title = f'{file.split("/")[-1]}_final_sample_{epoch}_{noise_num}.pdf'
        plot_graphs_list(graphs=gen_graph_list, title=pic_title, save_dir=config.save_dir)

        ## enable this if you wish to print the graphs in between the noiselevels
        #plot_inter_graphs_list(graphs=mult_stages, flags=mult_stages_flags, title='intermediate', save_dir=config.save_dir, nr_to_analyze=steps_to_log)

        ## calculate mmd scores
        result_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=methods)
        return result_dict, gen_graph_list

    result_dict, gen_graph_list = run_sample(eval_len=config.samplesize)
    return result_dict


## here the same as above but now we use traindata and we compare to traindata for the mmd scores (only used for model selection) 
def sample_testing(config,modellink,epoch,noise_num,train_dl):

    ##prepare our traindata received from the training script to be compatible with our sampling script
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

    ## load the model that we are currently at in our training iteration
    models = prepare_test_model_train(config,modellink)
    max_node_number = config.dataset.max_node_num
    test_batch_size = config.test.batch_size

    def gen_init_data(batch_size):
        rand_idx = np.random.randint(0, len(train_graph_list), batch_size)
        base_adjs, base_x = [train_graph_list_adj[i] for i in rand_idx], [train_graph_list_x[i] for i in rand_idx]
        base_adjs, base_x = torch.stack(base_adjs), torch.stack(base_x)
        base_adjs, base_x = base_adjs.to(config.dev), base_x.to(config.dev)
        node_flags = base_adjs.sum(-1).gt(1e-5).to(dtype=torch.float32)
        
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
        return initialmatrix, base_x, node_flags
        ##returns initialmatrix = tensor of size batchsize x N x N

    file, sigma_list, model_params = models[0]
    model = load_model(*model_params)
    sigma_tens = torch.linspace(0,1/2,noise_num)
    sigma_list = sigma_tens.tolist()
    sigma_list.sort()

    def add_bernoulli(flags, init_adjs, noiselevel):
        ##deprecated noisetype balanced
        """
        if config.noisetype=="balanced":
            init_adjs, noise_added = discretenoise_balanced(init_adjs, flags, noiselevel, config)
        else:
            init_adjs, noise_added = discretenoise(init_adjs, flags, noiselevel, config)"""

        init_adjs, noise_added = discretenoise(init_adjs, flags, noiselevel, config)
        return init_adjs

## receives the last levels prediction for x_0, adds enough noise to get to noiselevel x_t and then returns prediction for x_0
    def take_step(noise_func, flags, init_adjs, noiselevel):
        init_adjs = add_bernoulli(flags, init_adjs, noiselevel)
        mask=generate_mask(flags).to(config.dev)
        noise_unnormal = noise_func(A=init_adjs.to(config.dev),feat=None,mask=mask.to(config.dev),noise=noiselevel)
        noise_unnormal = noise_unnormal.squeeze(-1)
        noise_rel = torch.sigmoid(noise_unnormal)
        noise_rel = (noise_rel+torch.transpose(noise_rel,-2,-1))/2
        noise_rel = noise_rel*mask

        ##sample from estimated reverse distribution p(x_0|x_t)
        noise = torch.bernoulli(noise_rel)
        noise=torch.triu(noise,diagonal=1) + torch.triu(noise,diagonal=1).transpose(-2,-1)

        inter_adjs = torch.where(noise>1/2,init_adjs-1,init_adjs)
        new_adjs = torch.where(inter_adjs < -1/2 , inter_adjs+2 , inter_adjs)
        return new_adjs


    def run_sample(eval_len=10, methods=None):
        gen_graph_list = []
        with torch.no_grad():
            while len(gen_graph_list)<eval_len:
                count=0
                init_adjs, init_x, flags = gen_init_data(batch_size = test_batch_size)

                ## use this if you wish to observe the intermediate graphs
                #mult_stages = [adjs_to_graphs(init_adjs.detach().cpu().numpy())]
                #mult_stages_flags = flags[-test_batch_size*(0+1): len(flags)-(test_batch_size*(0))]

                while count<len(sigma_list):
                    noiselevel=sigma_list[len(sigma_list)-count-1]
                    init_adjs=take_step(lambda feat, A, mask, noise: model(feat, A, mask, noise), flags=flags, init_adjs=init_adjs, noiselevel=noiselevel)
                    count=count+1

                    ## use this if you wish to observe the intermediate graphs
                    #mult_stages.append(adjs_to_graphs(init_adjs.detach().cpu().numpy()))
                    #mult_stages_flags = torch.cat((mult_stages_flags, flags[-test_batch_size*(count): len(flags)-(test_batch_size*(count-1))]),0)

                gen_graph_list.extend(adjs_to_graphs(init_adjs.detach().cpu().numpy()))

        ##plot selection of generated graphs
        pic_title = f'{file.split("/")[-1]}_final_sample_{epoch}_{noise_num}.pdf'
        plot_graphs_list(graphs=gen_graph_list, title=pic_title, save_dir=config.save_dir)

        ##use this if you wish to plot the intermediate graphs
        #plot_inter_graphs_list(graphs=mult_stages, flags=mult_stages_flags, title='intermediate', save_dir=config.save_dir, nr_to_analyze=steps_to_log)

        ##evaluet mmd compared to train set
        result_dict = eval_graph_list(train_graph_list, gen_graph_list, methods=methods)

        return result_dict, gen_graph_list

    result_dict, gen_graph_list = run_sample(eval_len=256)
    return result_dict


if __name__ == "__main__":
    args = parse_arguments('sample_com_small_ddpm_16.yaml')
    config_dict = get_config(args)
    sample_main(config_dict, args)











        








        






    



