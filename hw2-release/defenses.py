import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from torch import optim

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=7, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
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
                               
    # total number of updates - FILL ME
    num_updates = int(np.ceil(epochs / m))
    

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    model.train()
    num_iters = 0
    for epoch in tqdm(range(num_updates)):
        for i, data in enumerate(loader_tr):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs.requires_grad = True
            delta = torch.zeros_like(inputs).to(device)

            for j in range(m):
                inputs.requires_grad = True
                optimizer.zero_grad()
                outputs = model(inputs + delta)
                loss = criterion(outputs, labels)
                loss.backward()
                # Update model
                optimizer.step()
                # Update delta
                delta += eps * torch.sign(inputs.grad)
                with torch.no_grad():
                    delta = torch.clamp(delta, -eps, eps)
                num_iters += 1
                if num_iters % scheduler_step_iters == 0:
                    lr_scheduler.step()
    
    
    # done
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
        counts = torch.zeros(4, device=x.device)
        for i in range(n):
            noise = torch.randn_like(x)
            predictions = torch.argmax(self.model(x + noise), dim=1)
            counts += torch.bincount(predictions, minlength=4).to(x.device)
        return counts
            
        
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
        counts0 = self._sample_under_noise(x,n0,batch_size)
        c_a = torch.argmax(counts0)
        counts = self._sample_under_noise(x,n,batch_size)
        p_a = proportion_confint(counts[c_a].cpu(), n, 2 * alpha, method="beta")[0]

        if p_a <= 0.5:
            c = self.ABSTAIN
            radius = 0 
        else:
            c = c_a
            radius = self.sigma * norm.ppf(p_a)

        # done
        return c, radius
        

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
        
        self.model.eval()
        # randomly initialize mask and trigger in [0,1] - FILL ME
        trigger = torch.rand(self.dim, requires_grad=ס2True, device=device)
        tmp_size = self.dim
        if len(tmp_size) == 4: 
            tmp_size = (tmp_size[2], tmp_size[3]) # 2D
        mask = torch.rand(tmp_size, requires_grad=True, device=device)

        optimizer = torch.optim.Adam([mask, trigger], lr=self.step_size)

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        for ep in range(self.niters):
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs = inputs.to(device)

                adv = mask * trigger + (1 - mask) * inputs
                targets = c_t * torch.ones_like(labels, device=device)
                
                outputs = self.model(adv)
                loss = self.loss_func(outputs, targets) + self.lambda_c * mask.abs().sum() 

                # update mask and trigger and zero gradients
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    mask.clamp_(0, 1)
                    trigger.clamp_(0, 1)
                optimizer.zero_grad()

        mask = mask.repeat(3, 1, 1)
        # done
        return mask, trigger
