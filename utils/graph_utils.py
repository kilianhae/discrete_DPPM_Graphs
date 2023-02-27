import numpy as np
import torch

do_check_adjs_symmetry = False


def mask_adjs(adjs, node_flags):
    """

    :param adjs:  B x N x N or B x C x N x N
    :param node_flags: B x N
    :return:
    """
    # assert node_flags.sum(-1).gt(2-1e-5).all(), f"{node_flags.sum(-1).cpu().numpy()}, {adjs.cpu().numpy()}"
    if len(adjs.shape) == 4:
        node_flags = node_flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * node_flags.unsqueeze(-1)
    adjs = adjs * node_flags.unsqueeze(-2)
    return adjs


def get_corrupt_k(min_k=0, max_k=None, p=0.5):
    ret = np.random.geometric(p) + min_k - 1
    if max_k is not None:
        ret = min(ret, max_k)
    # print(ret, end=' ')
    return ret


def remove_self_loop_if_exists(adjs):
    return (adjs - torch.eye(adjs.size()[-1]).unsqueeze(0).to(adjs.device)).clamp(min=0.0)


def add_self_loop_if_not_exists(adjs):
    if len(adjs.shape) == 4:
        return adjs + torch.eye(adjs.size()[-1]).unsqueeze(0).unsqueeze(0).to(adjs.device)
    return adjs + torch.eye(adjs.size()[-1]).unsqueeze(0).to(adjs.device)


def toggle_edge_np(adj, count=1):
    """
    uniformly toggle `count` edges of the graph, suppose that the vertex number is fixed

    :param adj: N x N
    :param count: int
    :return: new adjs and node_flags
    """
    count = min(count, adj.shape[-1])
    x = np.random.choice(adj.shape[0], count)
    y = np.random.choice(adj.shape[1], count)
    change = 1. - adj[x, y]
    adj[x, y] = change
    adj[y, x] = change
    return adj


def check_adjs_symmetry(adjs):
    if not do_check_adjs_symmetry:
        return
    tr_adjs = adjs.transpose(-1, -2)
    assert (adjs - tr_adjs).abs().sum([0, 1, 2]) < 1e-2


def gen_list_of_data(train_x_b, train_adj_b, train_node_flag_b, sigma_list, config):
    """
    :param train_x_b: [batch_size, N, F_in], batch of feature vectors of nodes
    :param train_adj_b: [batch_size, N, N], batch of original adjacency matrices
    :param train_node_flag_b: [batch_size, N], the flags for the existence of nodes
    :param sigma_list: list of noise levels
    :return:
        train_x_b: [len(sigma_list) * batch_size, N, F_in], batch of feature vectors of nodes (w.r.t. `train_noise_adj_b`)
        train_noise_adj_b: [len(sigma_list) * batch_size, N, N], batch of perturbed adjacency matrices
        train_node_flag_b: [len(sigma_list) * batch_size, N], the flags for the existence of nodes (w.r.t. `train_noise_adj_b`)
        grad_log_q_noise_list: [len(sigma_list) * batch_size, N, N], the ground truth gradient (w.r.t. `train_noise_adj_b`)
    """
    assert isinstance(sigma_list, list)
    train_noise_adj_b_list = []
    grad_log_q_noise_list = []
    for sigma_i in sigma_list:
        if config.noisetype=="balanced":
            train_noise_adj_b, grad_log_q_noise = discretenoise_balanced(train_adj_b,
                                                                    node_flags=train_node_flag_b,
                                                                    sigma=sigma_i, config=config)
        else:
            train_noise_adj_b, grad_log_q_noise = discretenoise(train_adj_b,
                                                                    node_flags=train_node_flag_b,
                                                                    sigma=sigma_i, config=config)
        train_noise_adj_b_list.append(train_noise_adj_b)
        grad_log_q_noise_list.append(grad_log_q_noise)
    train_noise_adj_b = torch.cat(train_noise_adj_b_list, dim=0)
    train_x_b = train_x_b.repeat(len(sigma_list), 1, 1)
    train_node_flag_b = train_node_flag_b.repeat(len(sigma_list), 1)
    return train_x_b, train_noise_adj_b, train_node_flag_b, grad_log_q_noise_list


def gen_list_of_data_single(train_x_b, train_adj_b, train_node_flag_b, sigma_list, config):
    """
    :param train_x_b: [batch_size, N, F_in], batch of feature vectors of nodes
    :param train_adj_b: [batch_size, N, N], batch of original adjacency matrices
    :param train_node_flag_b: [batch_size, N], the flags for the existence of nodes
    :param sigma_list: list of noise levels
    :return:
        train_x_b: [len(sigma_list) * batch_size, N, F_in], batch of feature vectors of nodes (w.r.t. `train_noise_adj_b`)
        train_noise_adj_b: [len(sigma_list) * batch_size, N, N], batch of perturbed adjacency matrices
        train_node_flag_b: [len(sigma_list) * batch_size, N], the flags for the existence of nodes (w.r.t. `train_noise_adj_b`)
        grad_log_q_noise_list: [len(sigma_list) * batch_size, N, N], the ground truth gradient (w.r.t. `train_noise_adj_b`)
    """
    assert isinstance(sigma_list, list)
    train_noise_adj_b_list = []
    grad_log_q_noise_list = []
    count=0
    for sigma_i in sigma_list:
        if config.noisetype=="balanced":
            train_noise_adj_b, grad_log_q_noise = discretenoise_balanced_single(train_adj_b[count],
                                                                    node_flags=train_node_flag_b[count],
                                                                    sigma=sigma_i, config=config)
        elif config.noisetype=="density":
            density = train_adj_b[count].sum() / train_node_flag_b[count].sum()
            train_noise_adj_b, grad_log_q_noise = discretenoise_single_density(train_adj_b[count],
                                                                    node_flags=train_node_flag_b[count],
                                                                    sigma=sigma_i, weights=[0,1,0], density=density , config=config)
        elif config.noisetype=="density_balanced":
            density = train_adj_b[count].sum() / train_node_flag_b[count].sum()
            train_noise_adj_b, grad_log_q_noise = discretenoise_balanced_single_density(train_adj_b[count],
                                                                    node_flags=train_node_flag_b[count],
                                                                    sigma=sigma_i, weights=[0,1,0], density=density , config=config)
        else:
            train_noise_adj_b, grad_log_q_noise = discretenoise_single(train_adj_b[count],
                                                                    node_flags=train_node_flag_b[count],
                                                                    sigma=sigma_i, config=config)
        train_noise_adj_b_list.append(train_noise_adj_b)
        grad_log_q_noise_list.append(grad_log_q_noise)
        count=count+1

    train_noise_adj_b = torch.cat(train_noise_adj_b_list, dim=0)
    train_x_b = train_x_b.repeat(len(sigma_list), 1, 1)
    train_node_flag_b = train_node_flag_b.repeat(1, 1)
    return train_x_b, train_noise_adj_b, train_node_flag_b, grad_log_q_noise_list


def add_gaussian_noise(adjs, node_flags, sigma, is_half=False):
    assert isinstance(adjs, torch.Tensor)
    noise = torch.randn_like(adjs).triu(diagonal=1) * sigma
    if is_half:
        noise = noise.abs()
    noise_s = noise + noise.transpose(-1, -2)
    check_adjs_symmetry(noise_s)
    grad_log_noise = - noise_s / (sigma ** 2)
    ret_adjs = adjs + noise_s
    ret_adjs = mask_adjs(ret_adjs, node_flags)
    grad_log_noise = mask_adjs(grad_log_noise, node_flags)
    return ret_adjs, grad_log_noise


def discretenoise(train_adj_b,node_flags,sigma,config):
    train_adj_b = train_adj_b.to(config.dev)
    # If Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    bernoulli_adj = torch.where(train_adj_b>1/2,torch.full_like(train_adj_b,1-sigma).to(config.dev),torch.full_like(train_adj_b,sigma).to(config.dev))
    noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
    noise_lower = noise_upper.transpose(-1, -2)
    grad_log_noise = torch.abs(-train_adj_b + noise_upper + noise_lower)
    train_adj_b = noise_upper + noise_lower
    train_adj_b = mask_adjs(train_adj_b, node_flags)
    grad_log_noise = mask_adjs(grad_log_noise, node_flags)
    return train_adj_b, grad_log_noise

def discretenoise_balanced_single_density(train_adj_b,node_flags,sigma,weights,density,config):
    sigma=sigma*2
    train_adj_b = train_adj_b.to(config.dev)
    edges_tens = train_adj_b.sum()
    nodes_tens = node_flags.sum()
    n=float(nodes_tens)
    m=float(edges_tens)
    # WATCH OUT density here has to be twice the edges / n so A.sum/n
    sigma_on = float((density * n) / (n * (n-1)))
    sigma_off = 1 - sigma_on
    # If Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    noise_upper=torch.bernoulli(torch.full_like(train_adj_b,sigma)).triu(diagonal=1).to(config.dev)
    noise_lower = noise_upper.transpose(-1, -2).to(config.dev)
    noise = noise_upper + noise_lower
    # noise now 1 where it is to be drawn a new edge and 0 if keep the old
    sampled_lower = torch.bernoulli(torch.where(noise>1/2,sigma_on,0.0)).triu(diagonal=1).to(config.dev)
    sampled=sampled_lower+sampled_lower.transpose(-1, -2).to(config.dev)
    # sampled is 1 where we get a new one and 0 where we get a new zero or where we leave the old one and is symmetric
    # noisediff is 0 everywhere
    noisediff = torch.zeros_like(train_adj_b).to(config.dev)
    # whereever noise is 1 (tobedrawn) then put the new value into there from sampled and put a 1.0 into the noisediff matrix so that we know there was a switch
    
    for j,vec in enumerate(noise):
        for k,node  in enumerate(vec):
            if node>0.9:
                if train_adj_b[j][k]>0.9 and 0.9>sampled[j][k]:
                    noisediff[j][k] =  1.0
                elif train_adj_b[j][k]<0.9 and sampled[j][k]>0.9:
                    noisediff[j][k] = 1.0
                train_adj_b[j][k]=sampled[j][k]
    train_adj_b = mask_adjs(train_adj_b, node_flags).to(config.dev)
    noisediff = mask_adjs(noisediff, node_flags).to(config.dev)
    return train_adj_b, noisediff


def discretenoise_single_density(train_adj_b,node_flags,sigma,weights,density,config):
    edges_tens = 0.0
    nodes_tens = 0.0
    a = weights[0]
    b = weights[1]
    c = weights[2]
    edges_tens = train_adj_b.sum()
    nodes_tens = node_flags.sum()
    sigma_off = sigma
    n=float(nodes_tens)
    m=float(edges_tens)
    # Since density is equal to nrofedges_in_original/nr_of_nodes thus we can replace the term desnity * n by "norofedges_in_original" = thus this would here be equal to: 
    if m == n*(n-1):
        sigma_on = 1/2
    else:
        sigma_on_uncontrolled = ( density*(a+b*n+c*n*n) + (sigma_off-1)*m ) / ((n*(n-1))-m)
        if sigma_on_uncontrolled > 1/2:
            sigma_on = 0.5
        elif sigma_on_uncontrolled < 0:
            sigma_on = 0.0
        else:
            sigma_on = sigma_on_uncontrolled
    train_adj_b = train_adj_b.to(config.dev)
    ##if Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    bernoulli_adj = torch.where(train_adj_b>1/2,torch.full_like(train_adj_b,1-sigma_off).to(config.dev),torch.full_like(train_adj_b,sigma_on).to(config.dev))
    noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
    noise_lower = noise_upper.transpose(-1, -2)
    grad_log_noise = torch.abs(-train_adj_b + noise_upper + noise_lower)
    train_adj_b = noise_upper + noise_lower
    train_adj_b = mask_adjs(train_adj_b, node_flags)
    grad_log_noise = mask_adjs(grad_log_noise, node_flags)
    return train_adj_b, grad_log_noise


def discretenoise_single(train_adj_b,node_flags,sigma,config):
    train_adj_b = train_adj_b.to(config.dev)
    ##if Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    bernoulli_adj = torch.where(train_adj_b>1/2,torch.full_like(train_adj_b,1-sigma).to(config.dev),torch.full_like(train_adj_b,sigma).to(config.dev))
    noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
    noise_lower = noise_upper.transpose(-1, -2)
    grad_log_noise = torch.abs(-train_adj_b + noise_upper + noise_lower)
    train_adj_b = noise_upper + noise_lower
    train_adj_b = mask_adjs(train_adj_b, node_flags)
    grad_log_noise = mask_adjs(grad_log_noise, node_flags)
    return train_adj_b, grad_log_noise


def discretenoise_balanced(train_adj_b,node_flags,sigma,config):
    train_adj_b = train_adj_b.to(config.dev)
    ##if Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    sigma=sigma*2
    noise_upper=torch.bernoulli(torch.full_like(train_adj_b,sigma)).triu(diagonal=1).to(config.dev)
    noise_lower = noise_upper.transpose(-1, -2).to(config.dev)
    noise = noise_upper + noise_lower
    sampled_lower = torch.bernoulli(torch.where(noise>1/2,1/2,0.0)).triu(diagonal=1).to(config.dev)
    sampled=sampled_lower+sampled_lower.transpose(-1, -2).to(config.dev)
    noisediff = torch.zeros_like(train_adj_b).to(config.dev)
    for i,adj in enumerate(noise):
        for j,vec in enumerate(adj):
            for k,node  in enumerate(vec):
                if node>0.9:
                    if train_adj_b[i][j][k]>sampled[i][j][k]:
                        noisediff[i][j][k] =  1.0
                    elif train_adj_b[i][j][k] < sampled[i][j][k]:
                        noisediff[i][j][k] = 1.0

                    train_adj_b[i][j][k]=sampled[i][j][k] 
    train_adj_b=mask_adjs(train_adj_b, node_flags).to(config.dev)
    noisediff = mask_adjs(noisediff, node_flags).to(config.dev)
    return train_adj_b, noisediff


def discretenoise_balanced_single(train_adj_b,node_flags,sigma,config):
    train_adj_b = train_adj_b.to(config.dev)
    ##if Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    sigma = sigma * 2
    noise_upper=torch.bernoulli(torch.full_like(train_adj_b,sigma)).triu(diagonal=1).to(config.dev)
    noise_lower = noise_upper.transpose(-1, -2).to(config.dev)
    noise = noise_upper + noise_lower
    sampled_lower = torch.bernoulli(torch.where(noise>1/2,1/2,0.0)).triu(diagonal=1).to(config.dev)
    sampled=sampled_lower+sampled_lower.transpose(-1, -2).to(config.dev)
    noisediff = torch.zeros_like(train_adj_b).to(config.dev)
   
    for j,vec in enumerate(noise):
        for k,node  in enumerate(vec):
            if node > 0.9:
                if train_adj_b[j][k] > sampled[j][k]:
                    noisediff[j][k] =  1.0
                elif train_adj_b[j][k] < sampled[j][k]:
                    noisediff[j][k] = 1.0
                train_adj_b[j][k] = sampled[j][k]
    
    train_adj_b=mask_adjs(train_adj_b, node_flags).to(config.dev)
    noisediff = mask_adjs(noisediff, node_flags).to(config.dev)
    return train_adj_b, noisediff     
    
def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    # a = np.logical_or(a, np.identity(node_number))
    return a

# input of size batchsize x N 
# output of batchsize x N x N
def generate_mask(node_flags):
    groundtruth=torch.zeros(node_flags.size(0),node_flags.size(1),node_flags.size(1))
    for i,graph in enumerate(node_flags):
        for j,flag in enumerate(graph):
            for k,flag2 in enumerate(graph):
                if flag2>=0.9 and flag >=0.9:
                    groundtruth[i][j][k]=1
    return groundtruth