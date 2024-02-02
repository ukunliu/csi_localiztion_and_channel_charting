import torch as th
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, in_feats, out_dim, hiddens=[64, 32]):
        super().__init__()
        self.name = "ANN"
        th.manual_seed(24)
        th.set_default_dtype(th.float32)
        assert len(hiddens) > 1
        list_FC_layers = [nn.Linear(in_feats, hiddens[0]), nn.ReLU(), nn.BatchNorm1d(hiddens[0])]
        n_hiddens = len(hiddens)
        for l in range(n_hiddens - 1):
            list_FC_layers.append(nn.Linear(hiddens[l], hiddens[l+1]))
            list_FC_layers.append(nn.ReLU())
            list_FC_layers.append(nn.BatchNorm1d(hiddens[l+1]))
        list_FC_layers.append(nn.Linear(hiddens[n_hiddens - 1], out_dim))
        list_FC_layers.append(nn.BatchNorm1d(out_dim))
        self.model = nn.Sequential(*list_FC_layers)

    def forward(self, X):
        # return th.exp(self.model.forward(X))
        return self.model.forward(X.to(dtype=th.float32))
    
    def get_embedding(self, x):
        return self.forward(x)
    

class MetricNet(nn.Module):
    def __init__(self, in_feats, out_dim=1, hiddens=[64, 32]):
        super().__init__()
        self.name = "MetricNet"
        out_dim = 1
        in_feats = in_feats * 2
        th.set_default_dtype(th.float32)
        th.manual_seed(24)
        assert len(hiddens) > 1
        list_FC_layers = [nn.Linear(in_feats, hiddens[0]), nn.ReLU()]
        n_hiddens = len(hiddens)
        for l in range(n_hiddens - 1):
            list_FC_layers.append(nn.Linear(hiddens[l], hiddens[l+1]))
            list_FC_layers.append(nn.ReLU())
            # list_FC_layers.append(nn.BatchNorm1d(hiddens[l+1]))
        list_FC_layers.append(nn.Linear(hiddens[n_hiddens - 1], out_dim))
        list_FC_layers.append(nn.ReLU())
        # list_FC_layers.append(nn.BatchNorm1d(out_dim))
        self.model = nn.Sequential(*list_FC_layers)

        if isinstance(self.model, nn.Linear):
            th.nn.init.xavier_uniform(self.model.weight)


    def sys_forward(self, x, y):
        d_xy = self.model.forward(th.cat((x,y), -1))
        d_yx = self.model.forward(th.cat((y,x), -1))
        return (d_xy + d_yx) / 2


    def forward(self, x, y, z):
        # return th.exp(self.model.forward(X))
        d_xy = self.model.forward(th.cat((x,y), -1))
        # d_yx = self.model.forward(th.cat((y,x), -1))

        d_xz = self.model.forward(th.cat((x,z), -1))
        # d_zx = self.model.forward(th.cat((z,x), -1))

        # d_xy = self.sys_forward(x, y)
        # d_xz = self.sys_forward(x, z)

        # d_sym_xy = (d_xy + d_yx) / 2
        # d_sym_xz = (d_xz + d_zx) / 2
        return d_xy, d_xz
    
    def get_embedding(self, x, y):
        if not th.is_tensor(x):
            x = th.tensor(x, dtype=th.float32)
        if not th.is_tensor(y):
            y = th.tensor(y, dtype=th.float32)
        d = (self.model.forward(th.cat((x, y), -1)) + self.model.forward(th.cat((y, x), -1))) / 2
        return d
    
    
class Conv1d(nn.Module):
    def __init__(self, in_feats, out_dim):
        super(Conv1d, self).__init__()
        self.name = 'CNN'
        self.downConv = nn.Conv1d(in_channels=in_feats,
                                  out_channels=in_feats,
                                  kernel_size=3)
        self.norm = nn.BatchNorm1d(in_feats)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3)
        self.dense = nn.Linear(67, out_dim)

    def forward(self, x):
        x = self.downConv(x[:, None, :])
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # x = x.transpose(1,2)
        x = self.dense(x.squeeze())
        return x
    
    def get_embedding(self, x):
        return self.forward(x)
    
    
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        # x1, x2, x3 = x(0), x(1), x(2)
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    

class TripletMetric(nn.Module):
    def __init__(self, embedding_net):
        super(TripletMetric, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        # x1, x2, x3 = x(0), x(1), x(2)
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    

class PairNet(nn.Module):
    def __init__(self, embedding_net):
        super(PairNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        # x1, x2, x3 = x(0), x(1), x(2)
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)