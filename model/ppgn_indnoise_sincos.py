import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from util.model_helper import masked_instance_norm2D, masked_layer_norm2D, zero_diag
import math

SLOPE=0.01


class PowerfulLayer(nn.Module):
    ##in_feature will be given as hidden-dim and out as well hidden-dim
    def __init__(self, in_feat: int, out_feat: int, num_layers: int, activation=nn.LeakyReLU(negative_slope=SLOPE), spectral_norm=(lambda x: x)):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat= out_feat
        ##generate num_layers linear layers with the first one being of dim in_feat and all others out_feat, in between those layers we add an activation layer 
        self.m1 =  nn.Sequential(*[spectral_norm(nn.Linear(in_feat if i ==0 else out_feat, out_feat)) if i % 2 == 0 else activation for i in range(num_layers*2-1)])
        self.m2 =  nn.Sequential(*[spectral_norm(nn.Linear(in_feat if i ==0 else out_feat, out_feat)) if i % 2 == 0 else activation for i in range(num_layers*2-1)])
        ##a linear layer that has input dim in_feat + out_feat and outputdim out_feat and a possible bias
        ##this is mentioned in paper as the layer after each concatenbation back to outputdim
        self.m4 = nn.Sequential(spectral_norm(nn.Linear(in_feat + out_feat, out_feat, bias=True)))

    ##expects x as batch x N x N x in_feat and mask as batch x N x N x 1
    def forward(self, x, mask):
        """ x: batch x N x N x in_feat """

        ##norm by taking uppermost col of mask gives nr of nodes active and then sqrt of that and put it in matching array dim
        norm = mask[:,0].squeeze(-1).float().sum(-1).sqrt().view(mask.size(0), 1, 1, 1)

        ##here i will start to treat mask as in the edp paper, namely batch x N with then a boolean value
        #batch * N * N * 1 gets to batch * 1 * N * N
        mask = mask.unsqueeze(1).squeeze(-1)

        ##run the two mlp on input and permute dimensions so that now matches dim of mask: batch * N * N * out_features
        out1 = self.m1(x).permute(0, 3, 1, 2) * mask           # batch, out_feat, N, N
        out2 = self.m2(x).permute(0, 3, 1, 2) * mask           # batch, out_feat, N, N

        ##matrix multiply each matching layer of features as well as adjacencies
        out = out1 @ out2 
        del out1, out2
        out =  out / norm 
        ##permute back to correct dim and concat with the skip-mlp in last dim
        out = torch.cat((out.permute(0, 2, 3, 1), x), dim=3) # batch, N, N, out_feat
        del x
        ##run through last layer to go back to out_features dim
        out = self.m4(out)

        return out



##this is the class for the invariant layer that makes the whole thing invariant
class FeatureExtractor(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation=nn.LeakyReLU(negative_slope=SLOPE), spectral_norm=(lambda x: x)):
        super().__init__()
        self.lin1 = nn.Sequential(spectral_norm(nn.Linear(in_features, out_features, bias=True)))
        self.lin2 = nn.Sequential(spectral_norm(nn.Linear(in_features, out_features, bias=False)))
        self.lin3 = nn.Sequential(spectral_norm(nn.Linear(out_features, out_features, bias=False)))
        self.activation = activation

    def forward(self, u, mask):
        """ u: (batch_size, num_nodes, num_nodes, in_features)
            output: (batch_size, out_features). """
        u = u * mask

        ##tensor of batch * 1 that represernts nr of active nodes
        n = mask[:,0].sum(1)
        ##tensor of batches * features * their diagonal elements (this retrieves the node elements that are stored on the diagonal)
        diag = u.diagonal(dim1=1, dim2=2)       # batch_size, channels, num_nodes

        #tensor of batch * features with storing the sum of diagonals
        trace = torch.sum(diag, dim=2)
        del diag

        ##run the first layer with input of batchsize * inputfeatures storing the summ of all diagonal features(a.k.a the node features) divided by norm
        out1 = self.lin1.forward(trace / n)

        ####didnt look whats happening here since its somehow different from paper as they do MPM
        ####i think essence of it is that since they sum all nodes anyways so it kind of doesnt make a difference where which node is
        s = (torch.sum(u, dim=[1, 2]) - trace) / (n * (n-1))
        del trace
        out2 = self.lin2.forward(s)  # bs, out_feat
        del s
        out = out1 + out2
        out = out + self.lin3.forward(self.activation(out))

        ##returns output tensor of size batches * outfeatures
        return out
        
class SinusoidalPosEmb(nn.Module):
   def __init__(self, dim):
       super().__init__()
       self.dim = dim
   def forward(self, x):
       device = x.device
       half_dim = self.dim // 2
       emb = math.log(10000) / (half_dim - 1)
       emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
       emb = x[:, None] * emb[None, :]
       emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
       return emb

class Powerful_sincos(nn.Module):
    def __init__(self,use_norm_layers:bool ,name: str ,channel_num_list, feature_nums: list, gnn_hidden_num_list: list, num_layers: int, input_features: int, hidden: int, hidden_final: int, dropout_p: 0.000001,
                 simplified: bool, n_nodes: int, config, normalization: str = 'none', cat_output: bool = True, adj_out: bool = False,
                 output_features: int = 1, residual: bool = False, activation=nn.LeakyReLU(negative_slope=SLOPE), spectral_norm=(lambda x: x),
                 project_first: bool = False, node_out: bool = False, noise_mlp: bool = False):
        super().__init__()
        self.cat_output = cat_output
        self.normalization = normalization
        layers_per_conv = 2 # was 1 originally, try 2?
        self.layer_after_conv = not simplified
        self.dropout_p = dropout_p
        self.adj_out = adj_out
        self.residual = residual
        self.activation = activation
        self.project_first = project_first
        self.node_out = node_out
        self.output_features= output_features
        self.node_output_features = output_features
        self.noise_mlp = noise_mlp
        self.config=config


        ###this is added by me for noiselevel conditioning
        self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(hidden),
                nn.Linear(hidden, 4*hidden),
                self.activation,
                nn.Linear(4*hidden, hidden),
                self.activation
            )


        ##one input layer to go from inputdim to hidden
        #self.in_lin = nn.Sequential(spectral_norm(nn.Linear(input_features, hidden)))
        self.in_lin = nn.Sequential(spectral_norm(nn.Linear(1, hidden)))

        ##cat_output always true
        if self.cat_output:
            if self.project_first:
                self.layer_cat_lin = nn.Sequential(spectral_norm(nn.Linear((hidden)*(num_layers+1), hidden)))
            else:
                self.layer_cat_lin = nn.Sequential(spectral_norm(nn.Linear((hidden)*num_layers+input_features, hidden)))
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        ##needs one conditional extra for befor the in_lin layer
        self.times = nn.ModuleList([nn.Sequential(spectral_norm(nn.Linear(hidden, 1)))])
        for i in range(num_layers):
            self.convs.append(PowerfulLayer(hidden, hidden, layers_per_conv, activation=self.activation, spectral_norm=spectral_norm))
            self.times.append(nn.Sequential(spectral_norm(nn.Linear(hidden, hidden))))

        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            # self.bns.append(nn.BatchNorm2d(hidden))
            if self.normalization == 'layer':
                # self.bns.append(None)
                self.bns.append(nn.LayerNorm([n_nodes,n_nodes,hidden], elementwise_affine=False))
            elif self.normalization == 'batch':
                self.bns.append(nn.BatchNorm2d(hidden))
            else:
                self.bns.append(None)
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final, activation=self.activation, spectral_norm=spectral_norm))
        if self.layer_after_conv:
            self.after_conv = nn.Sequential(spectral_norm(nn.Linear(hidden_final, hidden_final)))
        self.final_lin = nn.Sequential(spectral_norm(nn.Linear(hidden_final, output_features)))

        if self.node_out:
            if self.cat_output:
                if self.project_first:
                    self.layer_cat_lin_node = nn.Sequential(spectral_norm(nn.Linear(hidden*(num_layers+1), hidden)))
                else:
                    self.layer_cat_lin_node = nn.Sequential(spectral_norm(nn.Linear(hidden*num_layers+input_features, hidden)))
            if self.layer_after_conv:
                self.after_conv_node = nn.Sequential(spectral_norm(nn.Linear(hidden_final, hidden_final)))
            self.final_lin_node = nn.Sequential(spectral_norm(nn.Linear(hidden_final, node_output_features)))
 

    def get_out_dim(self):
        return self.output_features


    ##expects the input as the adjacency tensor: batchsize x N x N 
    ##expects the node_features as tensor: batchsize x N x node_features
    ##expects the mask as tensor: batchsize x N x N
    ##expects noiselevel as the noislevel that was used as single float
    def forward(self, node_features, A, mask, noiselevel):
        out = self.forward_cat(A, node_features, mask, noiselevel)
        # if self.cat_output:
            # out = self.forward_cat(A, node_features, mask)
        # else:
            # out = self.forward_old(A, node_features, mask)
        return out

    def forward_cat(self, A, node_features, mask, noiselevel):
        mask = mask[..., None]
        print(mask)
        if len(A.shape) < 4:
            print(A)
            u = A[..., None]           # batch, N, N, 1
            print("AAAAAAAAA")
            print(u)
            print(u.size())
            #u=A
        else:
            u = A

        ###condition the noise level, this implementation just runs it through a 1 layer mlp with gelu activation
        ###attention here we always must ggo with noise_mlp=True
        if self.noise_mlp:
            noiselevel_tensor = torch.full([u.size(0)],noiselevel).to(self.config.dev)
            print(noiselevel_tensor.size())
            noiselevel_tensor_prepro = self.time_mlp(noiselevel_tensor)
            print(noiselevel_tensor_prepro.size())

            ##now size batchsize x hidden

            """
            noiselevel=self.time_mlp(torch.full([1],noiselevel).to(self.config.dev))
            noiselevel_in = self.times[0](noiselevel)
            noise_level_matrix=noiselevel_in.expand(u.size(0),u.size(1),u.size(3)).to(self.config.dev)
            noise_level_matrix=torch.diag_embed(noise_level_matrix.transpose(-2,-1), dim1=1, dim2=2)
            #noise_level_matrix=noiselevel.expand(u.size(0),u.size(1),u.size(2),u.size(3)).to(self.config.dev)"""
        """
        else:
            noiselevel=torch.full([1],noiselevel).to(self.config.dev)
            noise_level_matrix=noiselevel.expand(u.size(0),u.size(1),u.size(3)).to(self.config.dev)
            noise_level_matrix=torch.diag_embed(noise_level_matrix.transpose(-2,-1), dim1=1, dim2=2)"""


        
        ####this would be the line if we did use the initial node features but in a random noise case maybe not necessary, otherwise the dim of features is 18
        #u = torch.cat([u, torch.diag_embed(node_features.transpose(-2,-1), dim1=1, dim2=2),noise_level_matrix], dim=-1).to(self.config.dev)
        #u = torch.cat([u,noise_level_matrix], dim=-1).to(self.config.dev)

        ##add noise to input adjacency for in_lin layer of dim =1
        noiselevel_in = self.times[0](noiselevel_tensor_prepro)
        print(noiselevel_in.size())
        noiselevel_in = self.activation(noiselevel_in)
        noiselevel_in=torch.unsqueeze(noiselevel_in,1).to(self.config.dev)
        noiselevel_in=torch.unsqueeze(noiselevel_in,1).to(self.config.dev)
        noise_level_matrix=noiselevel_in.expand(u.size(0),u.size(1),u.size(2),u.size(3)).to(self.config.dev)
        u = u + noise_level_matrix

        del node_features
        print(u)
        # if edge_features is not None:
        #     u[:,:,:,-edge_features.size(-1):][:, ~torch.eye(mask.size(1), dtype=torch.bool)] = edge_features
        # del edge_features
        print(mask)
        #u = u * mask
        if self.project_first:
            print(u.size())
            u = self.in_lin(u)
            out = [u]
        else:
            print(u.size())
            out = [u]
            u = self.in_lin(u)

        noisecounter=1
        for conv, extractor, bn in zip(self.convs, self.feature_extractors, self.bns):
            noiselevel_in = self.times[noisecounter](noiselevel_tensor_prepro)
            noiselevel_in = self.activation(noiselevel_in)
            ##now noiselevel_i is a tensor of form batchsize x hidden or batchsize x 1 before first layer
            #now expand to batchsize x N x N x hidden or ... x 1 so that we can then bring it to
            noiselevel_in=torch.unsqueeze(noiselevel_in,1).to(self.config.dev)
            noiselevel_in=torch.unsqueeze(noiselevel_in,1).to(self.config.dev)
            noise_level_matrix=noiselevel_in.expand(u.size(0),u.size(1),u.size(2),u.size(3)).to(self.config.dev)
            u = u + noise_level_matrix

            u = conv(u, mask) + (u if self.residual else 0)
            if self.normalization == 'none':
                u = u
            elif self.normalization == 'instance':
                u = masked_instance_norm2D(u, mask)
            elif self.normalization == 'layer':
                # u = bn(u)
                u = masked_layer_norm2D(u, mask)
            elif self.normalization == 'batch':
                u = bn(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            else:
                raise ValueError
            u = u * mask # Unnecessary with fixed instance norm
            out.append(u)
            noisecounter=noisecounter+1
        del u
        out = torch.cat(out, dim=-1)
        if self.node_out and self.adj_out:
            node_out = self.layer_cat_lin_node(out.diagonal(dim1=1, dim2=2).transpose(-2,-1))
            if self.layer_after_conv:
                node_out = node_out + self.activation(self.after_conv_node(node_out))
            node_out = F.dropout(node_out, p=self.dropout_p, training=self.training)
            node_out = self.final_lin_node(node_out)
        out = self.layer_cat_lin(out)
        if not self.adj_out:
            out = self.feature_extractors[-1](out, mask)
        if self.layer_after_conv:
            out = out + self.activation(self.after_conv(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.final_lin(out)
        if self.node_out and self.adj_out:
            return out, node_out
        else:
            return out



