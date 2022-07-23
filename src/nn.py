import operator
import functools
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims):
        assert len(dims) > 1
        super().__init__()
        self.num_linears = len(dims) - 1
        for i in range(self.num_linears):
            if i + 1 < self.num_linears:
                L = nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1], momentum=0.06),
                    nn.LeakyReLU(0.1),
                )
            else:
                L = nn.Linear(dims[i], dims[i + 1])
            setattr(self, 'L' + str(i), L)

    def forward(self, x):
        for i in range(self.num_linears):
            x = getattr(self, 'L' + str(i))(x)
        return x


class CouplingLayer(nn.Module):
    def __init__(self, dims, reverse):
        assert len(dims) > 1 and dims[0] == dims[-1]
        assert all([dim % 2 == 0 for dim in dims])
        super().__init__()
        dims = [dim // 2 for dim in dims]
        self.s_mlp = MLP(dims)
        self.t_mlp = MLP(dims)
        self.reverse = reverse

    def __partition(self, x):
        if self.reverse:
            return x.flip(1)
        return x

    def __flow(self, x, inverse=False):
        assert len(x.shape) == 2
        x = self.__partition(x)
        x1, x2 = x.chunk(2, dim=1)
        s = torch.tanh(self.s_mlp(x1))
        t = self.t_mlp(x1)
        if inverse:
            x2 = (x2 - t) * (-s).exp()
        else:
            x2 = x2 * s.exp() + t
        x = torch.cat((x1, x2), dim=1)
        x = self.__partition(x)
        jacobian_log = torch.sum(s, dim=1)
        return x, jacobian_log

    def inverse_flow(self, y):
        x, _ = self.__flow(y, inverse=True)
        return x

    def forward(self, x):
        y, jacobian_log = self.__flow(x)
        return y, jacobian_log


class RealNVP(nn.Module):
    def __init__(self, dims, shape, device):
        assert len(dims) > 0
        assert len(shape) == 3
        super().__init__()
        self.num_cl = len(dims)
        self.shape = shape
        self.device = device
        self.dims_after_flatten = functools.reduce(operator.mul, shape)
        for i in range(self.num_cl):
            ith_dims = [self.dims_after_flatten] + dims[i] + [self.dims_after_flatten]
            setattr(self, 'CL' + str(i), CouplingLayer(ith_dims, i % 2 == 0))
        self.to(device)

    def inverse_flow(self, y):
        assert len(y.shape) == 2
        for i in range(self.num_cl - 1, -1, -1):
            y = getattr(self, 'CL' + str(i)).inverse_flow(y)
        y = torch.reshape(y, (y.shape[0], *self.shape))
        return y

    def forward(self, x):
        assert len(x.shape) == 4
        x = torch.flatten(x, start_dim=1)
        sum_of_jacobian_log = torch.zeros(x.shape[0], device=self.device)
        for i in range(self.num_cl):
            x, jacobian_log = getattr(self, 'CL' + str(i))(x)
            sum_of_jacobian_log += jacobian_log
        return x, sum_of_jacobian_log


class NLLLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.to(device)

    def forward(self, x, jacobian_log):
        assert x.shape[0] == jacobian_log.shape[0]
        x_log_prob = self.normal.log_prob(x.cpu()).to(self.device)
        ll_loss = jacobian_log + x_log_prob.sum(dim=1)
        nll_loss = -ll_loss.mean()
        return nll_loss
