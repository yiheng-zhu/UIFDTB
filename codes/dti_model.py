import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import dgl
import dgl.nn as dglnn


class QSI(torch.nn.Module):
    def __init__(self, num_features_xd=78, output_dim=1024, dropout=0.2, graph_size=1024, graph_hid=1024):
        super(QSI, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES graph branch
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)

        self.bn1 = nn.BatchNorm1d(num_features_xd)
        self.bn2 = nn.BatchNorm1d(num_features_xd * 2)
        self.bn3 = nn.BatchNorm1d(num_features_xd * 4)

        # ECFP
        fp_feature = 1024
        fc_feature = 768
        interpro_feature = 45899

        self.fc_fp = nn.Linear(fp_feature, 1024)
        self.bn_fp = nn.BatchNorm1d(1024)

        self.fc_ch = nn.Linear(fc_feature, 1024)
        self.bn_ch = nn.BatchNorm1d(1024)

        self.fc_fu = nn.Linear(1024, output_dim * 2)
        self.fc_in1 = nn.Linear(interpro_feature, 2048)
        self.fc_in2 = nn.Linear(2048, 2048)
        self.bn_in = nn.BatchNorm1d(2048)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(2 * 1024, 1024 // 4),
            nn.ReLU(),
            nn.Linear(1024 // 4, 1)
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(2 * 2048, 2048 // 4),
            nn.ReLU(),
            nn.Linear(2048 // 4, 1)
        )

        self.GCN = GCN(graph_size, graph_hid)

        self.fc_esmfold = nn.Sequential(
            nn.Linear(graph_size, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),
            # nn.BatchNorm1d(2048)
        )

        self.fc_esm = nn.Linear(2560, output_dim * 2)

        # 1024*4 concat后的维度
        self.dci = DCiPatho2(freqs_nums=1024 * 5, deep_layers=[1024 * 5, 4096, 2048, 1024, 512])

        self.cross = CrossAttentionDTI(
            drug_dim=num_features_xd * 4,  # drug conv3 输出维度
            prot_dim=1024,  # protein layer3 输出维度
            d_model=512,
            n_heads=8,
            dropout=0.2
         )

        self.fc_attention = torch.nn.Linear(1024, 1024)
        self.layer_norm1 = nn.LayerNorm(1024)

        self.bn_xf = nn.BatchNorm1d(1024)
        self.bn_pro_f = nn.BatchNorm1d(2048)

    def forward(self, data, g, h):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # esm = data.esm   #xk

        fp = data.fp  # xf
        ch = data.ch
        interpro = data.interpro

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x_atom = x
        x = gmp(x, batch)

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # drug fp branch (fully connected layer)
        fp = self.relu(self.fc_fp(fp.float()))
        ch = self.relu(self.fc_ch(ch.float()))
        interpro = self.dropout(self.relu(self.fc_in1(interpro.float())))
        interpro = self.dropout(self.relu(self.fc_in2(interpro)))

        alpha = torch.sigmoid(self.alpha_mlp(torch.cat([fp, ch], dim=1)))

        xf = alpha * fp + (1 - alpha) * ch

        xf = self.dropout(self.relu(self.fc_fu(xf)))

        esmgcn, esm_h = self.GCN(g, h)
        esmgcn = self.fc_esmfold(esmgcn)

        beta = torch.sigmoid(self.beta_mlp(torch.cat([esmgcn, interpro], dim=1)))
        pro_f = beta * esmgcn + (1-beta) * interpro


        num_res = g.batch_num_nodes().tolist()
        z_pair = self.cross(x_atom, batch, esm_h, num_res)

        # concat
        xc = torch.cat((xf, pro_f, z_pair), 1)

        out = torch.sigmoid(self.dci(xc))

        z_t = F.normalize(pro_f, dim=-1)
        # drug global 可以不 detach，或也 detach，看你想让谁学
        z_d = F.normalize(xf, dim=-1)

        return out, z_t, z_d


class GCN(nn.Module):
    def __init__(self, in_dim=2560, hidden_dim=2560):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.project_esm_pssm = nn.Linear(in_dim + 20, hidden_dim)

        self.alpha_mlp1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.alpha_mlp2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, g, h):
        pre = h
        h = self.conv1(g, h)
        h = self.bn1(h)
        h = self.dropout(F.relu(h))

        alpha1 = torch.sigmoid(self.alpha_mlp1(torch.cat([pre, h], dim=1)))

        h = alpha1 * pre + (1 - alpha1) * h

        pre = h
        # layer 2
        h = self.conv2(g, h)
        h = self.bn2(h)
        h = self.dropout(F.relu(h))

        alpha2 = torch.sigmoid(self.alpha_mlp2(torch.cat([pre, h], dim=1)))

        h = alpha2 * pre + (1 - alpha2) * h

        with g.local_scope():
            g.ndata['h'] = h
            readout = dgl.mean_nodes(g, 'h')

        return readout, h


class SENetAttention(nn.Module):
    def __init__(self, num_fields, reduction_ratio=3):
        super().__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(reduced_size, num_fields, bias=False),
                                        nn.ReLU())

    def forward(self, x):
        A = self.excitation(x)
        V = x * A
        return V


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=True)

    def forward(self, x):
        out = self.linear2(torch.relu(self.linear1(x)))
        out += x
        out = torch.relu(out)
        return out


class Deep(nn.Module):
    def __init__(self, input_dim, deep_layers):
        super(Deep, self).__init__()

        deep_ayer_list = []
        for layer in list(zip(deep_layers[:-1], deep_layers[1:])):
            deep_ayer_list.append(nn.Linear(layer[0], layer[1]))
            deep_ayer_list.append(nn.BatchNorm1d(layer[1], affine=False))
            deep_ayer_list.append(nn.ReLU(inplace=True))
        self._deep = nn.Sequential(*deep_ayer_list)

    def forward(self, x):
        out = self._deep(x)
        return out


class DCiPatho2(nn.Module):
    def __init__(self, freqs_nums=1024 * 4, end_dims=[1024, 512, 256], out_layer_dims=1024,
                 deep_layers=[1024 * 4, 4096, 2048, 1024, 512], hidden_layers=[512, 256, 128], num_cross_layers=7,
                 Dropout=0.3):
        super(DCiPatho2, self).__init__()
        self._input_dim = freqs_nums

        self.out_layer1 = nn.Linear(self._input_dim, out_layer_dims)
        self.out_layer2 = nn.Linear(deep_layers[-1], out_layer_dims)
        self.drop_layer = nn.Dropout(p=Dropout)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self._input_dim, layer) for layer in hidden_layers
        ])
        self._SENet = SENetAttention(1024 * 5)

        self._deepNet = Deep(self._input_dim, deep_layers)
        self._final_dim = 2 * out_layer_dims
        self.end_layer1 = nn.Linear(self._final_dim, 2048)
        self.end_layer2 = nn.Linear(2048, 1024 * 5)
        self._final_linear = nn.Linear(end_dims[-1], 1)

        self.out_layer3 = nn.Linear(self._input_dim, out_layer_dims)
        self.out_layer4 = nn.Linear(deep_layers[-1], out_layer_dims)
        self.drop_layer2 = nn.Dropout(p=Dropout)
        self.residual_layers2 = nn.ModuleList([
            ResidualBlock(self._input_dim, layer) for layer in hidden_layers
        ])
        self._SENet2 = SENetAttention(1024 * 5)

        self._deepNet2 = Deep(self._input_dim, deep_layers)
        self.end_layer3 = nn.Linear(self._final_dim, end_dims[-3])
        self.end_layer4 = nn.Linear(end_dims[-3], end_dims[-1])

    def forward(self, x):
        input = x
        input = self._SENet(input)
        for residual in self.residual_layers:
            input = residual(input)
        res_out = torch.relu(self.out_layer1(input))
        deep_out = self._deepNet(x)
        deep_out = torch.relu(self.out_layer2(deep_out))

        final_input = torch.cat([res_out, deep_out], dim=1)
        # final_input=deep_out

        final_input = self.drop_layer(final_input.view(x.size(0), -1))

        end1 = self.end_layer1(final_input)
        end2 = self.end_layer2(end1)

        input = self._SENet2(end2)
        for residual in self.residual_layers2:
            input = residual(input)
        res_out = torch.relu(self.out_layer3(input))
        deep_out = self._deepNet2(x)
        deep_out = torch.relu(self.out_layer4(deep_out))

        final_input = torch.cat([res_out, deep_out], dim=1)
        # final_input=deep_out
        final_input = self.drop_layer2(final_input.view(x.size(0), -1))

        end1 = self.end_layer3(final_input)
        end2 = self.end_layer4(end1)

        output = self._final_linear(end2)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch


class CrossAttentionDTI(nn.Module):
    """
    Cross-attention between drug atom nodes (PyG) and protein residue nodes (DGL).
    Inputs:
      x_atom: [sum_atoms, Dd]
      atom_batch: [sum_atoms] with values in [0..B-1]
      h_res: [sum_res, Dp]
      num_res_per_graph: list[int] length B, e.g. g.batch_num_nodes().tolist()

    Returns:
      z_pair: [B, 2*d_model]
      (optional) attention weights if you need them (currently not returned)
    """

    def __init__(
            self,
            drug_dim: int,
            prot_dim: int,
            d_model: int = 256,
            n_heads: int = 8,
            dropout: float = 0.1,
            ffn_mult: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        # project to same dimension
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # cross-attn (batch_first=True => [B, L, D])

        self.attn_d2p = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.attn_p2d = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # norms + FFN (Transformer-style)
        self.norm_d1 = nn.LayerNorm(d_model)
        self.norm_p1 = nn.LayerNorm(d_model)
        self.norm_d3 = nn.LayerNorm(d_model)
        self.norm_p3 = nn.LayerNorm(d_model)
        self.norm_all = nn.LayerNorm(d_model * 2)

        self.ffn_d = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_p = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        x: [B, L, D]
        mask: [B, L] (True=valid)
        """
        mask_f = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        x_sum = (x * mask_f).sum(dim=dim)
        denom = mask_f.sum(dim=dim).clamp(min=1.0)
        return x_sum / denom

    def forward(self, x_atom, atom_batch, h_res, num_res_per_graph):
        device = x_atom.device

        # ---- drug: [sum_atoms, Dd] -> [B, NA, Dd] + mask
        x_atom = self.drug_proj(x_atom)  # [sum_atoms, d_model]
        Xd, atom_mask = to_dense_batch(x_atom, atom_batch)  # Xd:[B, NA, d_model], mask:[B, NA]

        # ---- protein: [sum_res, Dp] -> list -> pad -> [B, NL, Dp] + mask
        h_res = self.prot_proj(h_res)  # [sum_res, d_model]
        if not isinstance(num_res_per_graph, (list, tuple)):
            num_res_per_graph = num_res_per_graph.tolist()

        h_list = torch.split(h_res, num_res_per_graph)  # tuple of [Ni, d_model]
        Xp = pad_sequence(h_list, batch_first=True)  # [B, NL, d_model]

        B, NL, _ = Xp.shape
        num_res_t = torch.tensor(num_res_per_graph, device=device)
        res_mask = torch.arange(NL, device=device)[None, :] < num_res_t[:, None]  # [B, NL], True=valid

        # key_padding_mask expects True for padding positions
        res_kpm = ~res_mask
        atom_kpm = ~atom_mask

        # ---- drug attends protein (d2p)
        # residual + pre-norm

        d2p_out, _ = self.attn_d2p(Xd, Xp, Xp, key_padding_mask=res_kpm)
        p2d_out, _ = self.attn_p2d(Xp, Xd, Xd, key_padding_mask=atom_kpm)


        # ---- pool to pair embedding
        z_d = self.masked_mean(d2p_out, atom_mask, dim=1)  # [B, d_model]
        z_p = self.masked_mean(p2d_out, res_mask, dim=1)  # [B, d_model]
        z_pair = torch.cat([z_d, z_p], dim=-1)  # [B, 2*d_model]

        return z_pair
