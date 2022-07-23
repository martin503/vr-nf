import torch
import random
from src.helpers import *

class Test_lin_inter:
    def test_smoke(self):
        wh = 2
        smp1 = torch.rand(1, wh * wh)
        smp2 = torch.rand(1, wh * wh)
        res = lin_inter(smp1, smp2, num_smp_to_gen=2)
        diffs = [torch.round(res[i + 1] - res[i], decimals=2) for i in range(res.shape[0] - 1)]
        for i in range(len(diffs) - 1):
            assert torch.all(diffs[i + 1] == diffs[i]).item()
        
        for _ in range(20):
            wh = 2 * random.randint(4, 14)
            smp1 = torch.rand(1, wh * wh)
            smp2 = torch.rand(1, wh * wh)
            res = lin_inter(smp1, smp2)
            diffs = [torch.round(res[i + 1] - res[i], decimals=2) for i in range(res.shape[0] - 1)]
            for i in range(len(diffs) - 1):
                assert torch.all(diffs[i + 1] == diffs[i]).item()


class Test_find_contrasting:
    def test_smoke(self):
        batch_size = 2
        wh = 2
        samples = torch.rand(batch_size, 1, wh, wh)
        find_contrasting(samples)
        
        for _ in range(20):
            batch_size = random.randint(2, 8)
            wh = 2 * random.randint(4, 14)
            samples = torch.rand(batch_size, 1, wh, wh)
            find_contrasting(samples)
                