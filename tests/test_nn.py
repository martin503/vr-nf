import torch
import random
import operator
import functools
from src.nn import *


class TestMLP:
    def test_channels(self):
        dims = [1, 2]
        mlp = MLP(dims)
        x = torch.rand((2, 1))
        y = mlp(x)
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] == dims[1]

        for _ in range(25):
            dims = [random.randint(1, 33), random.randint(1, 33),
                    random.randint(1, 33)]
            mlp = MLP(dims)
            x = torch.rand((3, dims[0]))
            y = mlp(x)
            assert y.shape[0] == x.shape[0]
            assert y.shape[1] == dims[2]


class TestCouplingLayer:
    def test_channels(self):
        dims = [2, 2]
        cl = CouplingLayer(dims, False)
        x = torch.rand((2, 2))
        y, _ = cl(x)
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] == dims[1]
        assert y.shape[1] == dims[0]

        for _ in range(25):
            in_out_dim = 2 * random.randint(1, 10)
            dims = [in_out_dim, 2 * random.randint(1, 33),
                    2 * random.randint(1, 33), in_out_dim]
            cl = CouplingLayer(dims, False)
            x = torch.rand((3, dims[0]))
            y, _ = cl(x)
            assert y.shape[0] == x.shape[0]
            assert y.shape[1] == dims[-1]
            assert y.shape[1] == dims[0]

        for _ in range(25):
            in_out_dim = 2 * random.randint(1, 10)
            dims = [in_out_dim, 2 * random.randint(1, 33),
                    2 * random.randint(1, 33), in_out_dim]
            cl = CouplingLayer(dims, True)
            x = torch.rand((3, dims[0]))
            y, _ = cl(x)
            assert y.shape[0] == x.shape[0]
            assert y.shape[1] == dims[-1]
            assert y.shape[1] == dims[0]

    def test_flow(self):
        for _ in range(25):
            in_out_dim = 2 * random.randint(1, 10)
            dims = [in_out_dim, 2 * random.randint(1, 33),
                    2 * random.randint(1, 33), in_out_dim]
            cl = CouplingLayer(dims, False)
            x = torch.rand((3, dims[0]))
            y, _ = cl(x)
            assert torch.all(y[:, :in_out_dim // 2] == x[:, :in_out_dim // 2]).item()

        for _ in range(25):
            in_out_dim = 2 * random.randint(1, 10)
            dims = [in_out_dim, 2 * random.randint(1, 33),
                    2 * random.randint(1, 33), in_out_dim]
            cl = CouplingLayer(dims, True)
            x = torch.rand((3, dims[0]))
            y, _ = cl(x)
            assert torch.all(y[:, in_out_dim // 2:] == x[:, in_out_dim // 2:]).item()


class TestRealNVP:
    def test_channels(self, device):
        dims = [[],
                [4],
                [6, 10],
                [2, 2, 2]]
        in_dims = [2, 1, 4, 4]
        rnvp = RealNVP(dims, in_dims[1:], device)

        x = torch.rand(in_dims).to(device)
        y, _ = rnvp(x)
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] == functools.reduce(operator.mul, in_dims[1:])

        x_rev = rnvp.inverse_flow(y)
        assert x_rev.shape == x.shape

        for _ in range(30):
            dims = [[2 * random.randint(1, 33)],
                    [2 * random.randint(1, 33), 2 * random.randint(1, 33)], ]
            in_dims = [2, 1, 4, 4]
            rnvp = RealNVP(dims, in_dims[1:], device)

            x = torch.rand(in_dims).to(device)
            y, _ = rnvp(x)
            assert y.shape[0] == x.shape[0]
            assert y.shape[1] == functools.reduce(operator.mul, in_dims[1:])

            x_rev = rnvp.inverse_flow(y)
            assert x_rev.shape == x.shape

    def test_indentity_flow(self, device):
        with torch.no_grad():
            dims = [[],
                    [4],
                    [6, 10],
                    [2, 2, 2]]
            in_dims = [2, 1, 4, 4]
            rnvp = RealNVP(dims, in_dims[1:], device).eval()

            x = torch.rand(in_dims).to(device)
            y, _ = rnvp(x)
            x_rev = rnvp.inverse_flow(y)
            x = torch.round(x, decimals=2)
            x_rev = torch.round(x_rev, decimals=2)
            assert torch.all(x_rev == x).item()

            for _ in range(30):
                dims = [[2 * random.randint(1, 33)]]
                in_dims = [4, 2, 8, 8]
                rnvp = RealNVP(dims, in_dims[1:], device).eval()

                x = torch.rand(in_dims).to(device)
                y, _ = rnvp(x)
                x_rev = rnvp.inverse_flow(y)
                x = torch.round(x, decimals=2)
                x_rev = torch.round(x_rev, decimals=2)
                assert torch.all(x_rev == x).item()

            for _ in range(30):
                dims = [[2 * random.randint(1, 33)]]
                in_dims = [2, 1, 16, 16]
                rnvp = RealNVP(dims, in_dims[1:], device).eval()

                x = torch.rand(in_dims).to(device)
                y, _ = rnvp(x)
                x_rev = rnvp.inverse_flow(y)
                x = torch.round(x, decimals=2)
                x_rev = torch.round(x_rev, decimals=2)
                assert torch.all(x_rev == x).item()

            for _ in range(30):
                dims = [[2 * random.randint(1, 33)]]
                in_dims = [2, 1, 28, 28]
                rnvp = RealNVP(dims, in_dims[1:], device).eval()

                x = torch.rand(in_dims).to(device)
                y, _ = rnvp(x)
                x_rev = rnvp.inverse_flow(y)
                x = torch.round(x, decimals=2)
                x_rev = torch.round(x_rev, decimals=2)
                assert torch.all(x_rev == x).item()

            for _ in range(30):
                dims = [[2 * random.randint(1, 33)],
                        [2 * random.randint(1, 33), 2 * random.randint(1, 33)]]
                in_dims = [2, 1, 4, 4]
                rnvp = RealNVP(dims, in_dims[1:], device).eval()

                x = torch.rand(in_dims).to(device)
                y, _ = rnvp(x)
                x_rev = rnvp.inverse_flow(y)
                x = torch.round(x, decimals=2)
                x_rev = torch.round(x_rev, decimals=2)
                assert torch.all(x_rev == x).item()


class TestNLLLoss:
    def test_channels(self, device):
        dims = [[],
                [4],
                [6, 10],
                [2, 2, 2]]
        in_dims = [4, 1, 28, 28]
        rnvp = RealNVP(dims, in_dims[1:], device)

        x = torch.rand(in_dims).to(device)
        y, jacobian = rnvp(x)
        nlll = NLLLoss(device)
        y0 = nlll(y, jacobian)

        d = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        for _ in range(20):
            x = d.sample((4, 28 * 28)).squeeze().to(device)
            y = nlll(x, torch.zeros(x.shape[0]).to(device))
            assert y < y0
