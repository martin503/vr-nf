# Normilizing Flows

This repo was developed during VR course at MIMUW in 2022\
Goal of the assignment was to learn how to implement and train Normalizing Flows. In summary model was meant to generate images of handwritten-like fives from random noise.
My solution got `9.5 / 10` score.\
Author: Marcin Mazur

## Code Structure

- `plots` - plots of best model results.
- `report` - report of my assignment solution (`mm418420.pdf`).
- `src` - modules used in model dev.
- `tests` - tests for modules.
- `normalizing_flows.ipynb` - notebook with final results.

## Performance

Tests should be very fast, and can be run with `pytest` command.\
Full training (`100` epochs) takes ~20min on 4GB GPU.

## Environment

Project was developed on:

- python 3.9.7
- pytorch 1.11.0

## Task description

Your task is to implement a version of Normalizing Flow for image generation. Our implementation will be based on RealNVP (<https://arxiv.org/pdf/1605.08803.pdf>) and we will be training on one class from MNIST. Your task is to read the paper in details and implement simple version of the algorithm from the paper:

1. Implement simple CouplingLayers (see RealNVP paper) with neural networks using a few fully connected layers with hidden activations of your choice. More on the CouplingLayers can be also found in <https://arxiv.org/pdf/1410.8516.pdf>. Remember to implement properly logarithm of a Jacobian determinant calculation. Implement only single scale architecture, ignore multiscale architecture with masked convolution and batch normalization. (2 points)
2. Implement RealNVP class combining many CouplingLayers with proper masking pattern (rememeber to alternate between unmodified pixels) with forward and inverse flows. (1 points)
3. Implement a loss function `nf_loss` (data log-likelihood) for the model. Hint: check `torch.distributions` (1 point)
4. Train your model to achieve good looking samples (similar to training set - similar to that appended to assignmenmt on moodle). The training process should take between 5-10 minutes. (2 points)
5. Sample from your model and pick 2 images (as visually different as possible) from your samples and plot 10 images that correspond to equally spaced linear interpolations in latent space between those images you picked. (1 point)
6. Use method from section 5.2 from <https://arxiv.org/pdf/1410.8516.pdf> with trained model and inpaint 5 sampled images with different random parts of your image occluded (50% of the image must be occluded). (2 point)
7. Write a report describing your solution, add loss plots and samples from the model. Write which hyperparameter sets worked for you and which did not. (1 point)
