import numpy as np
import scipy.stats
import statsmodels.stats.proportion
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from tqdm import tqdm
from collections import Counter


def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler,
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduler: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)

    # init delta (adv. perturbation) - FILL ME
    delta = torch.zeros((batch_size, data_tr[0][0].size(0), data_tr[0][0].size(1), data_tr[0][0].size(2)), requires_grad=True, device=device)
    # delta = torch.zeros(batch_size, data_tr[0][0].size(1),
    #                     data_tr[0][0].size(2), data_tr[0][0].size(3)).to(device)
    # delta = torch.zeros_like(data_tr, requires_grad=True, device=device)

    # total number of updates - FILL ME
    epochs_per_minibatch = int(epochs / m)

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr) / batch_size))

    for epochs in tqdm(range(epochs_per_minibatch)):
        for i, data in enumerate(loader_tr, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            for _ in range(m):
                perturbed_image = inputs + delta[:len(inputs)]
                out = model(perturbed_image)
                loss = criterion(out, labels)
                loss.backward(retain_graph=True)

                delta_grad = delta.grad.sign_()
                delta.data = torch.clamp(delta + eps * delta_grad, min=-eps, max=eps)

                optimizer.step()

                optimizer.zero_grad()
                delta.grad.zero_()

            if i * m % scheduler_step_iters == 0:
                lr_scheduler.step()

    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        batch_x = x.unsqueeze(0).expand(batch_size, -1, -1, -1)
        delta = torch.randn(n, *x.shape) * self.sigma

        predictions = []

        for i in range(0, n, batch_size):
            batch_delta = delta[i: i + batch_size]
            probabilities = self.model(batch_x[:len(batch_delta)] + batch_delta)
            predictions += torch.max(probabilities, 1)

        return dict(Counter(predictions))

    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """

        # find prediction (top class c) - FILL ME
        counts0 = self._sample_under_noise(x, n0, batch_size)
        c_max = max(counts0)
        counts = self._sample_under_noise(x, n, batch_size)

        # compute lower bound on p_c - FILL ME
        pa = statsmodels.stats.proportion.proportion_confint(counts[c_max], n, 1 - alpha)

        if pa > 0.5:
            return c_max, self.sigma * norm.ppf(pa)

        return self.ABSTAIN
        #
        # # Find prediction (top class c)
        # logits = self.model(x)
        # _, prediction = torch.max(logits, 1)
        # c = prediction.item()
        #
        # # Compute lower bound on p_c
        # perturbed_inputs = self._sample_under_noise(x, n0, batch_size)
        # perturbed_logits = self.model(perturbed_inputs)
        # perturbed_probs = torch.softmax(perturbed_logits, dim=1)
        # lower_bound, _ = torch.kthvalue(perturbed_probs[:, c], int((1 - alpha) * n0))
        #
        # # Compute certified radius
        # radius = self.sigma * torch.norm(logits[:, c] - lower_bound, p=1)
        #
        # # Return prediction and certified radius
        # return c, radius.item()
        #
        # # done
        # return c, radius


class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME

        # done
        return mask, trigger
