import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils import cal_cos_similarity
from transtest import selfAttention

class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=512, num_layers=3, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d'%i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d'%i, nn.Linear(
                360 if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d'%i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))

    def forward(self, x):
        # feature
        # [N, F]
        return self.mlp(x).squeeze()

class GRUPFG(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, K =3):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )

        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps.weight)
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs.weight)

        self.fc_ps_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_fore.weight)
        self.fc_ps_change = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_change.weight)
        self.fc_hs_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_fore.weight)

        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)
        self.fc_indi_change=nn.Linear(hidden_size, hidden_size)

        self.hsc = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.hsc.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim = 0)
        self.softmax_t2s = torch.nn.Softmax(dim = 1)


        self.fc_out_ps = nn.Linear(hidden_size, 1)
        self.fc_out_hs = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.K = K
        # self.result=nn.Linear(hidden_size, hidden_size)
        # self.self_att_net = selfAttention(8, d_feat, hidden_size)
        self.market_combine_network1= nn.Linear(hidden_size, hidden_size)
        self.market_combine_network01= nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.market_combine_network1.weight)
        self.market_combine_network2 = nn.Linear(hidden_size, hidden_size)
        self.market_combine_network3 = nn.Linear(hidden_size, hidden_size)
        self.market_combine_network02= nn.Linear(hidden_size, hidden_size)
        self.market_combine_network03= nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.market_combine_network2.weight)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear03 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        self.linear06 = nn.Linear(hidden_size, hidden_size)
        self.linear003 = nn.Linear(hidden_size, hidden_size)
        # self.linear18 = nn.Linear(360, hidden_size)
        self.linear12 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.linear7 = nn.Linear(hidden_size, hidden_size)
        self.linear8 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.linear6.weight)
        torch.nn.init.xavier_uniform_(self.linear7.weight)
        self.softmax_3=torch.nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        
    def cal_pearson_correlation_normalized(self, x, y): # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
        cos_similarity = xy/x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    # def cal_pearson_correlation_normalized(self, x, y):
    #     # 求 x 和 y 的均值
    #     x_mean = torch.mean(x, dim=1, keepdim=True)
    #     y_mean = torch.mean(y, dim=1, keepdim=True)
    #
    #     # 中心化 x 和 y
    #     x_centered = x - x_mean
    #     y_centered = y - y_mean
    #
    #     # 计算 x_centered 和 y_centered 的点积
    #     cov_xy = x_centered.mm(torch.t(y_centered))
    #
    #     # 计算 x_centered 和 y_centered 的模（标准差）
    #     x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=1)).reshape(-1, 1)
    #     y_std = torch.sqrt(torch.sum(y_centered ** 2, dim=1)).reshape(-1, 1)
    #
    #     # 计算相关系数
    #     correlation = cov_xy / (x_std.mm(torch.t(y_std)))
    #
    #     # 处理可能的 NaN 值
    #     correlation[correlation != correlation] = 0
    #
    #     # 将相关系数从[-1, 1]归一化到[0, 1]
    #     normalized_correlation = (correlation + 1) / 2
    #
    #     return normalized_correlation

    def forward(self, x, concept_matrix, market_value):
        device = torch.device(torch.get_device(x))

        x_hidden = x.reshape(len(x), self.d_feat, -1) # [N, F, T]      
        x_hidden = x_hidden.permute(0, 2, 1) # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]
        
        # x_hidden=self.linear3(self.softmax_3(self.linear6(x_hidden)))
        x_hidden_taking= self.linear7(x_hidden)

        x_hidden_taking_2= self.linear8(x_hidden)
        # x_hidden = self.market_combine_network3(x_hidden[:, :, -1])

        # # Predefined Concept Module
        # #(99,64)

        #  Test
        market_matrix_xhi1 = self.softmax_3(self.tanh(self.linear3(x_hidden)))
        market_matrix_xhi2 = self.softmax_s2t(self.tanh(self.linear6(x_hidden)))

        market_concept_matrix=self.cal_pearson_correlation_normalized(market_matrix_xhi1,market_matrix_xhi2)
        output_hs_one = self.market_combine_network1(x_hidden_taking) + self.market_combine_network2(
            market_concept_matrix.mm(x_hidden_taking))

        m = x_hidden - self.market_combine_network3(market_matrix_xhi1)-self.linear12(market_matrix_xhi2)
        # m = x_hidden - self.market_combine_network3(x_hidden_taking)
        market_matrix_m1 = self.softmax_3(self.tanh(self.linear03(m)))
        market_matrix_m2 = self.softmax_s2t(self.tanh(self.linear06(m)))
        m_matrix=self.cal_pearson_correlation_normalized(market_matrix_m1,market_matrix_m2)
        output_three_one = self.market_combine_network01(x_hidden_taking_2) + self.market_combine_network02(
            m_matrix.mm(x_hidden_taking_2))

        all_info = output_hs_one+output_three_one
        pred_all = self.fc_out(all_info).squeeze()

        return pred_all
