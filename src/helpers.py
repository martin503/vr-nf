import os
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_model(model, loss):
    if not os.path.isdir('models/'):
        os.mkdir('models/')
    current_time = time.localtime()
    name = time.strftime('%d_%b_%Y_%H-', current_time) + 'loss_' + str(int(abs(loss)))
    torch.save(model, 'models/' + name + '.ckpt')
    return name


def plot(samples, title=None):
    length = len(samples)
    fig, ax = plt.subplots(1, length, figsize=(2*length, 2))
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    for j in range(length):
        ax[j].imshow(samples[j].cpu().numpy())
        ax[j].axis('off')
    if title is not None:
        current_time = time.localtime()
        plt.savefig('plots/' + title + time.strftime('_%d-%b-%Y-%H', current_time) + '.png')
    plt.show()


def plot_loss(values, title=None):
    plt.plot(range(1, len(values) + 1), values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if title is not None:
        current_time = time.localtime()
        plt.savefig('plots/' + title + time.strftime('_%d-%b-%Y-%H', current_time) + '.png')
    plt.show()


def lin_inter(smp1, smp2, num_smp_to_gen=10):
    assert smp1.shape == smp2.shape
    diff = (smp2 - smp1) / (num_smp_to_gen + 1)
    smps = [smp1 + diff * i for i in range(1, num_smp_to_gen + 1)]
    return torch.cat((*smps,), dim=0)


def find_contrasting(samples):
    assert len(samples.shape) == 4
    diffs = torch.zeros((samples.shape[0],) * 2)
    for i in range(len(samples) - 1):
        j = i
        while samples.shape[0] > j:
            diffs[i, j] = torch.nn.functional.l1_loss(samples[i], samples[j])
            j += 1
    ij = (diffs == torch.max(diffs)).nonzero().squeeze()
    return samples[ij[0]], samples[ij[1]]


def occlude(samples):
    assert len(samples.shape) == 4
    samples = torch.clone(samples)
    masks = torch.ones(samples.shape, dtype=torch.bool)
    wh = samples.shape[-1] // 2
    diff = wh // (samples.shape[0] - 1)
    for i in range(samples.shape[0]):
        torch.nn.init.xavier_uniform_(samples[i, :, i * diff:wh + (i + 1) * diff, i * diff:wh + (i + 1) * diff])
        masks[i, :, i * diff:wh + (i + 1) * diff, i * diff:wh + (i + 1) * diff] = False
    return samples, masks
