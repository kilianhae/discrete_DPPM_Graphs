import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from util.model_helper import masked_instance_norm2D, masked_layer_norm2D, zero_diag

SLOPE = 0.01


class PowerfulLayer(nn.Module):
    # in_feature will be given as hidden-dim and out as well hidden-dim
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        num_layers: int,
        activation=nn.LeakyReLU(negative_slope=SLOPE),
        spectral_norm=(lambda x: x),
    ):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        # Generate num_layers linear layers with the first one being of dim in_feat and all others out_feat, in between those layers we add an activation layer
        self.m1 = nn.Sequential(
            *[
                spectral_norm(nn.Linear(in_feat if i == 0 else out_feat, out_feat))
                if i % 2 == 0
                else activation
                for i in range(num_layers * 2 - 1)
            ]
        )
        self.m2 = nn.Sequential(
            *[
                spectral_norm(nn.Linear(in_feat if i == 0 else out_feat, out_feat))
                if i % 2 == 0
                else activation
                for i in range(num_layers * 2 - 1)
            ]
        )
        # This is mentioned in paper as the layer after each concatenbation back to outputdim
        self.m4 = nn.Sequential(
            spectral_norm(nn.Linear(in_feat + out_feat, out_feat, bias=True))
        )

    # expects x as batch x N x N x in_feat and mask as batch x N x N x 1
    def forward(self, x, mask):
        """x: batch x N x N x in_feat"""

        # Norm by taking uppermost col of mask gives nr of nodes active and then sqrt of that and put it in matching array dim
        norm = mask[:, 0].squeeze(-1).float().sum(-1).sqrt().view(mask.size(0), 1, 1, 1)

        # Here i will start to treat mask as in the edp paper, namely batch x N with then a boolean value
        # batch * N * N * 1 gets to batch * 1 * N * N
        mask = mask.unsqueeze(1).squeeze(-1)

        # Run the two mlp on input and permute dimensions so that now matches dim of mask: batch * N * N * out_features
        out1 = self.m1(x).permute(0, 3, 1, 2) * mask  # batch, out_feat, N, N
        out2 = self.m2(x).permute(0, 3, 1, 2) * mask  # batch, out_feat, N, N

        # Matrix multiply each matching layer of features as well as adjacencies
        out = out1 @ out2
        del out1, out2
        out = out / norm
        # Permute back to correct dim and concat with the skip-mlp in last dim
        out = torch.cat((out.permute(0, 2, 3, 1), x), dim=3)  # batch, N, N, out_feat
        del x
        # Run through last layer to go back to out_features dim
        out = self.m4(out)

        return out


# This is the class for the invariant layer that makes the whole thing invariant
class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=nn.LeakyReLU(negative_slope=SLOPE),
        spectral_norm=(lambda x: x),
    ):
        super().__init__()
        self.lin1 = nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features, bias=True))
        )
        self.lin2 = nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features, bias=False))
        )
        self.lin3 = nn.Sequential(
            spectral_norm(nn.Linear(out_features, out_features, bias=False))
        )
        self.activation = activation

    def forward(self, u, mask):
        """u: (batch_size, num_nodes, num_nodes, in_features)
        output: (batch_size, out_features)."""
        u = u * mask

        # Tensor of batch * 1 that represernts nr of active nodes
        n = mask[:, 0].sum(1)
        # tensor of batches * features * their diagonal elements (this retrieves the node elements that are stored on the diagonal)
        # batch_size, channels, num_nodes
        diag = u.diagonal(dim1=1, dim2=2)

        # Tensor of batch * features with storing the sum of diagonals
        trace = torch.sum(diag, dim=2)
        del diag

        # Run the first layer with input of batchsize * inputfeatures storing the summ of all diagonal features(a.k.a the node features) divided by norm
        out1 = self.lin1.forward(trace / n)

        # I think essence of it is that since they sum all nodes anyways so it kind of doesnt make a difference where which node is
        s = (torch.sum(u, dim=[1, 2]) - trace) / (n * (n - 1))
        del trace
        out2 = self.lin2.forward(s)  # bs, out_feat
        del s
        out = out1 + out2
        out = out + self.lin3.forward(self.activation(out))

        # Returns output tensor of size batches * outfeatures
        return out


class Powerful_Ind(nn.Module):
    def __init__(
        self,
        use_norm_layers: bool,
        name: str,
        channel_num_list,
        feature_nums: list,
        gnn_hidden_num_list: list,
        num_layers: int,
        input_features: int,
        hidden: int,
        hidden_final: int,
        dropout_p: 0.000001,
        simplified: bool,
        n_nodes: int,
        config,
        normalization: str = "none",
        cat_output: bool = True,
        adj_out: bool = False,
        output_features: int = 1,
        residual: bool = False,
        activation=nn.LeakyReLU(negative_slope=SLOPE),
        spectral_norm=(lambda x: x),
        project_first: bool = False,
        node_out: bool = False,
        noise_mlp: bool = False,
    ):
        super().__init__()
        self.cat_output = cat_output
        self.normalization = normalization
        layers_per_conv = 2  # was 1 originally, try 2?
        self.layer_after_conv = not simplified
        self.dropout_p = dropout_p
        self.adj_out = adj_out
        self.residual = residual
        self.activation = activation
        self.project_first = project_first
        self.node_out = node_out
        self.output_features = output_features
        self.node_output_features = output_features
        self.noise_mlp = noise_mlp
        self.config = config

        # This is added by me for noiselevel conditioning
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(dim),
            nn.Linear(1, 4),
            nn.GELU(),
            nn.Linear(4, 1),
        )

        # One input layer to go from inputdim to hidden
        self.in_lin = nn.Sequential(
            spectral_norm(nn.Linear(input_features, hidden - 1))
        )

        # cat_output always true
        if self.cat_output:
            if self.project_first:
                self.layer_cat_lin = nn.Sequential(
                    spectral_norm(
                        nn.Linear((hidden - 1) * (num_layers + 1), hidden - 1)
                    )
                )
            else:
                self.layer_cat_lin = nn.Sequential(
                    spectral_norm(
                        nn.Linear(
                            (hidden - 1) * num_layers + input_features, hidden - 1
                        )
                    )
                )
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        # Needs one conditional extra for befor the in_lin layer
        self.times = nn.ModuleList([nn.Sequential(spectral_norm(nn.Linear(1, 1)))])
        for i in range(num_layers):
            self.convs.append(
                PowerfulLayer(
                    hidden,
                    hidden - 1,
                    layers_per_conv,
                    activation=self.activation,
                    spectral_norm=spectral_norm,
                )
            )
            self.times.append(nn.Sequential(spectral_norm(nn.Linear(1, 1))))
        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            # self.bns.append(nn.BatchNorm2d(hidden))
            if self.normalization == "layer":
                # self.bns.append(None)
                self.bns.append(
                    nn.LayerNorm([n_nodes, n_nodes, hidden], elementwise_affine=False)
                )
            elif self.normalization == "batch":
                self.bns.append(nn.BatchNorm2d(hidden))
            else:
                self.bns.append(None)
            self.feature_extractors.append(
                FeatureExtractor(
                    hidden - 1,
                    hidden_final,
                    activation=self.activation,
                    spectral_norm=spectral_norm,
                )
            )
        if self.layer_after_conv:
            self.after_conv = nn.Sequential(
                spectral_norm(nn.Linear(hidden_final - 1, hidden_final - 1))
            )
        self.final_lin = nn.Sequential(
            spectral_norm(nn.Linear(hidden_final - 1, output_features))
        )

        if self.node_out:
            if self.cat_output:
                if self.project_first:
                    self.layer_cat_lin_node = nn.Sequential(
                        spectral_norm(nn.Linear(hidden * (num_layers + 1), hidden))
                    )
                else:
                    self.layer_cat_lin_node = nn.Sequential(
                        spectral_norm(
                            nn.Linear(hidden * num_layers + input_features, hidden)
                        )
                    )
            if self.layer_after_conv:
                self.after_conv_node = nn.Sequential(
                    spectral_norm(nn.Linear(hidden_final, hidden_final))
                )
            self.final_lin_node = nn.Sequential(
                spectral_norm(nn.Linear(hidden_final, self.node_output_features))
            )

    def get_out_dim(self):
        return self.output_features

    # Expects the input as the adjacency tensor: batchsize x N x N
    # the node_features as tensor: batchsize x N x node_features
    # the mask as tensor: batchsize x N x N
    # noiselevel as the noislevel that was used as single float

    def forward(self, node_features, A, mask, noiselevel):
        out = self.forward_cat(A, node_features, mask, noiselevel)
        return out

    def forward_cat(self, A, node_features, mask, noiselevel):
        mask = mask[..., None]
        if len(A.shape) < 4:
            u = A[..., None]  # batch, N, N, 1
        else:
            u = A

        # Condition the noise level, this implementation just runs it through a 1 layer mlp with gelu activation
        if self.noise_mlp:
            noiselevel = self.time_mlp(torch.full([1], noiselevel).to(self.config.dev))
            noiselevel_in = self.times[0](noiselevel)
            noise_level_matrix = noiselevel_in.expand(
                u.size(0), u.size(1), u.size(3)
            ).to(self.config.dev)
            noise_level_matrix = torch.diag_embed(
                noise_level_matrix.transpose(-2, -1), dim1=1, dim2=2
            )
        else:
            noiselevel = torch.full([1], noiselevel).to(self.config.dev)
            noise_level_matrix = noiselevel.expand(u.size(0), u.size(1), u.size(3)).to(
                self.config.dev
            )
            noise_level_matrix = torch.diag_embed(
                noise_level_matrix.transpose(-2, -1), dim1=1, dim2=2
            )

        u = torch.cat([u, noise_level_matrix], dim=-1).to(self.config.dev)
        del node_features
        if self.project_first:
            u = self.in_lin(u)
            out = [u]
        else:
            out = [u]
            u = self.in_lin(u)

        noisecounter = 1
        for conv, extractor, bn in zip(self.convs, self.feature_extractors, self.bns):
            noiselevel_i = self.times[noisecounter](noiselevel)
            noise_level_matrix = noiselevel_i.expand(u.size(0), u.size(1), 1).to(
                self.config.dev
            )
            noise_level_matrix = torch.diag_embed(
                noise_level_matrix.transpose(-2, -1), dim1=1, dim2=2
            )
            u = torch.cat([u, noise_level_matrix], dim=-1).to(self.config.dev)
            u = conv(u, mask) + (u if self.residual else 0)
            if self.normalization == "none":
                u = u
            elif self.normalization == "instance":
                u = masked_instance_norm2D(u, mask)
            elif self.normalization == "layer":
                # u = bn(u)
                u = masked_layer_norm2D(u, mask)
            elif self.normalization == "batch":
                u = bn(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            else:
                raise ValueError
            u = u * mask  # Unnecessary with fixed instance norm
            out.append(u)
            noisecounter = noisecounter + 1
        del u
        out = torch.cat(out, dim=-1)
        if self.node_out and self.adj_out:
            node_out = self.layer_cat_lin_node(
                out.diagonal(dim1=1, dim2=2).transpose(-2, -1)
            )
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
