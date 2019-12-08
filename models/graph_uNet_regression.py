import torch.nn as nn
import torch
import utils.util as utils
import numpy as np
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_size, out_size, kernel_size):
        super(GCN, self).__init__()
        self.input_dim = in_size
        self.output_dim = out_size
        self.kernel_size = kernel_size
        self.kernel_weight = nn.Linear(in_size*kernel_size, out_size)

    def forward(self, inputs, support):
        """
        Do GCN Operation

        :param inputs: tensor, (batch_size, in_size, num_nodes)
        :param supports: tensors, [(num_nodes, num_nodes), (num_nodes, num_nodes), ...]

        :return: outputs: tensor, (batch_size, out_size, num_nodes)
        """
        batch_size = inputs.shape[0]
        num_nodes = inputs.shape[2]

        x = inputs
        x0 = x.permute(2, 1, 0) # (num_nodes, in_size, batch_size)
        x0 = x0.reshape(num_nodes, -1)
        x = x0.unsqueeze(0)
        x1 = torch.matmul(support, x0)
        if self.kernel_size > 1:
            x = GCN.concat(x, x1)
        for k in range(2, self.kernel_size):
            x2 = 2 * torch.matmul(support, x1) - x0
            x = GCN.concat(x, x2)
            x1, x0 = x2, x1

        x = x.reshape(self.kernel_size, num_nodes, self.input_dim, batch_size)
        x = x.permute(3, 1, 2, 0)
        x = x.reshape(batch_size * num_nodes, self.input_dim * self.kernel_size)
        x = self.kernel_weight(x)
        x = x.reshape(batch_size, num_nodes, self.output_dim)
        x = x.permute(0, 2, 1) # (batch_size, out_size, num_nodes)

        return x

    @staticmethod
    def concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm):
        super(UNetConvBlock, self).__init__()
        self.block = nn.ModuleList()
        self.batch_norm = batch_norm
        self._repeat = 3 if batch_norm else 2
        self.block.append(GCN(in_size, out_size, kernel_size=3))
        self.block.append(nn.ReLU())
        if batch_norm:
            # forward (N, C, L)
            self.block.append(nn.BatchNorm1d(out_size))

        self.block.append(GCN(out_size, out_size, kernel_size=3))
        self.block.append(nn.ReLU())
        if batch_norm:
            # forward (N, C, L)
            self.block.append(nn.BatchNorm1d(out_size))

    def forward(self, x, support):
        """
        Do GCN + Relu + BN + GCN + Relu + BN

        :param x: tensor, (batch_size, in_size, num_nodes)
        :param support: tensor, (num_nodes, num_nodes)
        :return: tensor, (batch_size, out_size, num_nodes)
        """
        for i, block in enumerate(self.block):
            if i % self._repeat == 0:
                x = block(x, support)
            else:
                x = block(x)

        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode='upsample', batch_norm=True):
        """

        :param in_size:
        :param out_size: out_size = in_size/2
        :param up_mode:
        :param batch_norm:
        """
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ModuleList([nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, dilation=2),
                       # GCN(in_size, out_size, kernel_size=1)
                       ])
        elif up_mode == 'upsample':
            self.up = nn.ModuleList([
                nn.Upsample(mode='bilinear', scale_factor=2),
                GCN(in_size, out_size, kernel_size=1),
            ])

        self.conv_block = UNetConvBlock(out_size*2, out_size, batch_norm)

    def center_crop2d(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def center_crop1d(self, layer, target_size):
        _, _, layer_height = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0])
               ]

    def everysecond_crop1d(self, layer, sep=2):
        _, _, layer_height = layer.size()
        # diff_y = (layer_height - target_size[0]) // 2
        return layer[:, :, ::sep] # return every second elements

    def forward(self, x, bridge, support, node_pos, height_width):

        up = self.up[0](x) # (2*h+1, 2*w+1)
        up = self.center_crop2d(up, height_width)
        up = up[:, :, node_pos[:, 0], node_pos[:, 1]]
        # up = self.up[1](up, support)
        # crop1 = self.center_crop1d(bridge, up.shape[2:])
        # # print("EverySecond Crop shape is ", crop1.shape)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out, support)

        return out

class DCRNN_Single(nn.Module):
    def __init__(self, graphs, node_pos_list, horizon, seq_len, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, filter_type="laplacian", coarsen_levels=4,
                 input_dim=3, output_dim=3, wf=5, up_mode='upconv', batch_norm=False, dropout=0.5):
        super(DCRNN_Single, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        # kernel_size = kernel_size[0]

        self.horizon = horizon
        self.seq_len = seq_len

        self.node_pos_list = node_pos_list

        # main rnn
        # self.hidden_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.batch_first = batch_first
        self.bias = bias
        # direction variable
        self.input_dim = input_dim
        # output dim should be two dim for volume and speed, 5 dim for direction
        self.output_dim = output_dim

        self.dropout_layer = nn.Dropout(dropout)
        self.sp_supports = []
        supports = [utils.calculate_scaled_laplacian(A, lambda_max=2.0, undirected=True) for A in graphs]
        for i, L in enumerate(supports):
            # using dense adjacency matrix
            L = L.tocoo()
            indices = np.vstack((L.row, L.col))
            self.sp_supports.append((indices, L.data, L.shape))
        self.coarsen_levels = coarsen_levels

        assert len(supports) == self.coarsen_levels

        # feat_dims = [32, hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim]
        #
        # assert len(feat_dims) >= self.coarsen_levels + 1
        # Do Pooling
        prev_channels = input_dim * self.seq_len
        self.down_path = nn.ModuleList()
        for i in range(self.coarsen_levels):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        # parameters need to pass into GPredGRU
        self._num_nodes = supports[-1].shape[0]

        # Do UnPooling
        self.up_path = nn.ModuleList()
        for i in reversed(range(self.coarsen_levels-1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = GCN(prev_channels, output_dim*self.horizon, kernel_size=1)

    def forward(self, input_t):
        """

        Parameters
        ----------
        input_tensor: (b, t*c, n)

        Returns
        -------
        layer_output: (b, t, c, n)
        """

        batch_size = input_t.size(0)
        num_nodes = input_t.size(2)

        supports = []
        # Get the current running device
        get_device = input_t.get_device()
        device_code = 'cuda:' + str(get_device)
        for i, data, shape in self.sp_supports:
            i = torch.from_numpy(i).long()
            v = torch.from_numpy(data).float()
            supports.append(torch.sparse.FloatTensor(i, v, shape).to(device_code))

        # input_t = input_tensor[:, :self.seq_len, :, :]
        # input_t = input_t.reshape(batch_size, self.seq_len*num_feat, num_nodes)

        blocks = []
        heights_widths = [(495, 436)]
        for i, down in enumerate(self.down_path):
            input_t = down(input_t, supports[i])
            input_t = self.dropout_layer(input_t)

            if i != len(self.down_path) - 1:
                blocks.append(input_t)
                tmp_num_feat = input_t.size(-2)
                # make sure the gird is minus infinity such that max_pooling is taking effect
                input_t_grid = torch.ones((batch_size, tmp_num_feat, heights_widths[-1][0],
                                           heights_widths[-1][1])).to(get_device) * -np.inf
                input_t_grid[:, :, self.node_pos_list[i][:, 0], self.node_pos_list[i][:, 1]] = input_t
                input_t_grid = F.max_pool2d(input_t_grid, 2)
                # print(f"down {i}, ")
                # print(f"len of node_pos_list is ", len(self.node_pos_list))
                input_t = input_t_grid[:, :, self.node_pos_list[i+1][:, 0], self.node_pos_list[i+1][:, 1]]
                heights_widths.append((input_t_grid.size(-2), input_t_grid.size(-1)))

        for i, up in enumerate(self.up_path):
            # print("block shape is ", blocks[-i-1].shape)

            tmp_num_feat = input_t.size(1)
            input_t_grid = torch.zeros(
                (batch_size, tmp_num_feat, heights_widths[-i-1][0], heights_widths[-i-1][1])).to(get_device)
            # print("input_t shape is ", input_t.shape)
            # print("input t grid shape is ", input_t_grid.shape)
            input_t_grid[:, :, self.node_pos_list[-i-1][:, 0], self.node_pos_list[-i-1][:, 1]] = input_t
            input_t = up(input_t_grid, blocks[-i - 1], supports[-i - 2], self.node_pos_list[-i-2], heights_widths[-i-2])
            input_t = self.dropout_layer(input_t)

        preds = self.last(input_t, supports[0])
        # preds = torch.relu(preds)
        # preds = torch.sigmoid(preds)
        # preds = torch.clamp(preds, min=0.0, max=255.0)
        preds = preds.reshape(batch_size, self.horizon, self.output_dim, num_nodes)

        return preds

class DCRNN_Single_2in1(nn.Module):

    def __init__(self, graphs,  node_pos_list, horizon, seq_len, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, filter_type="laplacian", coarsen_levels=4,
                 input_dim=2, output_dim=2, wf=5, up_mode='upconv', batch_norm=False, dropout=0.5):
        super(DCRNN_Single_2in1, self).__init__()

        self.dcrnn = DCRNN_Single(graphs, node_pos_list, horizon, seq_len, hidden_dim, kernel_size, num_layers,
                                  batch_first, bias, filter_type, coarsen_levels=coarsen_levels,
                                  input_dim=input_dim, output_dim=output_dim,
                                  wf=wf, up_mode=up_mode, batch_norm=batch_norm, dropout=dropout) # params for UNet

    def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor: todo
            4-D Tensor either of shape (t, b, n, c) or (b, t, n, c)~(default)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        layer_output
        """

        pred = self.dcrnn(input_tensor)

        return pred
