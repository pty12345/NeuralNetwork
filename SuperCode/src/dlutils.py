import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import random as random 

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class AdditiveScore(nn.Module):
    '''
        https://arxiv.org/pdf/1409.0473.pdf
        采用该论文中提出的计算方式
    '''
    '''
        e(q, k) = V^T * tanh(W*q + U*k)
    '''

    def __init__(self, v_d, q_d, k_d):
        '''
        :param v_d: dimension of vector v
        :param q_d: feature dimension of Query
        :param k_d: feature dimension of Key
        '''
        super(AdditiveScore, self).__init__()
        self.V = nn.Parameter(torch.randn(v_d, 1), requires_grad=True)
        self.W = nn.Parameter(torch.randn(v_d, q_d), requires_grad=True)
        self.U = nn.Parameter(torch.randn(v_d, k_d), requires_grad=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        '''
        e(q, k) = V^T * tanh(W*q + U*k)
        :param q: (batch_size, 1, q_d)
        :param k: (batch_size, window, k_d)
        :return: (batch_size, 1, window)
        '''
        # (batch_size, v_d, seq)
        Wq = torch.matmul(self.W, q.repeat(1, k.size()[1], 1).permute(0, 2, 1))
        
        # (batch_size, v_d, seq)
        Uk = torch.matmul(self.U, k.permute(0, 2, 1))
        
        # (batch_size, 1, seq)
        att_score = torch.matmul(self.V.T, torch.tanh(Wq + Uk))
        
        return torch.softmax(att_score, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        
class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
    
class Linear_thetafunc(nn.Module):
    '''
        A linear function that translate a matrix to a vector.
    '''
    def __init__(self, size, window, plength):
        super(Linear_thetafunc, self).__init__()
        # torch.empty : 分配内存空间，但是没有初始化里面的值，会存在nan情况
        # torch.randn : 分配内存空间，并用高斯噪声初始化里面的值
        self.weight_Q = Parameter(torch.randn(1, window), requires_grad=True)
        self.weight_P = Parameter(torch.randn(size, size), requires_grad=True)
        self.plength = plength
        # self.device = device
        self.window = window

    def mul(self, x: torch.Tensor) -> torch.Tensor:
        '''

        :param x: (batch, window, size)
        :return: (batch, 1, size)
        '''
        return torch.matmul(self.weight_Q, torch.matmul(x, self.weight_P))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''

        :param x: (batch, window, size)
        :return: (batch, plength, size)
        '''
        # 先深度拷贝原 tensor ，防止后面的 cat操作 对其原内存空间 有误操作。
        tx = torch.clone(x)
        buffer = []
        for l in range(0, self.plength):
            buffer.append(self.mul(tx))
            # print(buffer[-1].size())
            tx = torch.cat([tx[:, 1:, :], buffer[-1]], dim=1)
        return torch.cat(buffer, dim=1)

class LSTM_thetafunc(nn.Module):
    '''
        Basic seq2seq model based on lstm neural cell.
    '''
    def __init__(self, size, window, plength):
        super(LSTM_thetafunc, self).__init__()
        self.size = size
        self.window = window
        self.lstm = nn.LSTM(input_size=size, hidden_size=size, num_layers=1, batch_first=True)
        self.plength = plength

    def forward(self, x):
        '''

        :param x: (batch, window, size)
        :return:
        '''
        output, (h, c) = self.lstm(x)
        # print(h.size(), c.size())
        buffer = []
        for i in range(self.plength):
            _, (h, c) = self.lstm(h.permute(1, 0, 2), (h, c))
            buffer.append(_)
        return torch.cat(buffer, dim=1)

class Seq2Seq_with_attention(nn.Module):
    def __init__(self, input_size, window, hidden_size, output_size, plength):
        super(Seq2Seq_with_attention, self).__init__()
        self.input_size = input_size
        self.window = window
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.plength = plength
        self.attention = AdditiveScore(v_d=input_size, q_d=2 * hidden_size, k_d=input_size)
        self.lstm = nn.LSTM(input_size=input_size + output_size, hidden_size=hidden_size, num_layers=1,
                            bidirectional=False, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=output_size, bias=True))

    def forward(self, x: torch.Tensor):
        '''

        :param x: (batch_size, window, input_size)
        :return:
        '''

        device = x.device
        # (1, batch_size, hidden_size)
        s0 = torch.autograd.Variable(torch.zeros(1, x.size()[0], self.hidden_size)).to(device)
        
        # (1, batch_size, hidden_size)
        c0 = torch.autograd.Variable(torch.zeros(1, x.size()[0], self.hidden_size)).to(device)
        
        # (batch_size, 1, output_size)
        y0 = torch.autograd.Variable(torch.zeros(x.size()[0], 1, self.output_size)).to(device)
        
        # (batch_size, plength, window)
        weight = torch.autograd.Variable(torch.zeros(x.size()[0], self.plength, x.size()[1])).to(device)
        
        # (batch_size, plength, output_size)
        predict_y = torch.autograd.Variable(torch.zeros(x.size()[0], self.plength, self.output_size)).to(device)
        
        for i in range(self.plength):
            # (batch_size, 1, 2 * hidden_size)
            q = torch.cat([s0, c0], dim=2).permute(1, 0, 2)
            
            # (batch_size, 1, window)
            weight_i = self.attention(q, x)
            weight[:, i, :] = weight_i[:, 0, :]
            
            # (batch_size, 1, input_size)
            context = torch.matmul(weight_i, x)
            
            # (batch_size, 1, output_size + input_size)
            new_x = torch.cat([y0, context], dim=2)
            _, lstm_state = self.lstm(new_x, (s0, c0))
            s0 = lstm_state[0]
            c0 = lstm_state[1]
            predict_y[:, i, :] = self.dense(lstm_state[0].permute(1, 0, 2))[:, 0, :]
            y0 = predict_y[:, i, :].unsqueeze(1)

        return predict_y, weight

class Seq2Seq_with_attention_modified(nn.Module):
    def __init__(self, input_size, window, hidden_size, output_size, plength):
        super(Seq2Seq_with_attention_modified, self).__init__()
        self.input_size = input_size
        self.window = window
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.plength = plength
        self.device = None
        self.attention = AdditiveScore(v_d=input_size, q_d=2 * hidden_size, k_d=input_size)
        self.lstm = nn.LSTM(input_size=input_size + output_size, hidden_size=hidden_size, num_layers=1,
                            bidirectional=False, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=output_size, bias=True))
        self.context_s = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size, bias=True), nn.Tanh())
        self.context_c = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        '''

        :param x: (batch_size, window, input_size)
        :return:
        '''
        self.device = x.device
        # (1, batch_size, hidden_size)
        # s0 = torch.autograd.Variable(torch.zeros(1, x.size()[0], self.hidden_size)).to(self.device)
        s0 = self.context_s(x[:, -1, :]).unsqueeze(0)
        # print(s0.size())
        # (1, batch_size, hidden_size)
        # c0 = torch.autograd.Variable(torch.zeros(1, x.size()[0], self.hidden_size)).to(self.device)
        c0 = self.context_c(x[:, -1, :]).unsqueeze(0)
        # print(c0.size())
        # (batch_size, 1, output_size)
        y0 = torch.autograd.Variable(torch.zeros(x.size()[0], 1, self.output_size)).to(self.device)
        # (batch_size, plength, window)
        weight = torch.autograd.Variable(torch.zeros(x.size()[0], self.plength, x.size()[1])).to(self.device)
        # (batch_size, plength, output_size)
        predict_y = torch.autograd.Variable(torch.zeros(x.size()[0], self.plength, self.output_size)).to(self.device)
        for i in range(self.plength):
            # (batch_size, 1, 2 * hidden_size)
            q = torch.cat([s0, c0], dim=2).permute(1, 0, 2)
            # (batch_size, 1, window)
            weight_i = self.attention(q, x)
            weight[:, i, :] = weight_i[:, 0, :]
            # (batch_size, 1, input_size)
            context = torch.matmul(weight_i, x)
            # (batch_size, 1, output_size + input_size)
            new_x = torch.cat([y0, context], dim=2)
            _, lstm_state = self.lstm(new_x, (s0, c0))
            s0 = lstm_state[0]
            c0 = lstm_state[1]
            # print(self.dense(lstm_state[0].permute(1, 0, 2)).size())
            predict_y[:, i, :] = self.dense(lstm_state[0].permute(1, 0, 2))[:, 0, :]
            y0 = predict_y[:, i, :].unsqueeze(1)

        return predict_y, weight
    
class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # get the latest weight
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
    
def initialParameter(model: torch.nn.Module):
    # Reference:
    # https://www.cnblogs.com/quant-q/p/15056396.html
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU)):
            
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.zeros_(m.bias_ih_l0)
            nn.init.zeros_(m.bias_hh_l0)
            
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.1)
            
    # print('initial parameters...done!')
    
def initRandomSeeds(seed=0):
    # sets the seed for generating random numbers.
    torch.manual_seed(seed) 
    
    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed) 
    
    # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) 
    
    # Numpy module.
    np.random.seed(seed)
    
    # Python random module.
    random.seed(seed)

def MigrateToGPU(model, optimizer):
    model.cuda()
    for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    return model, optimizer