import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        assert torch.all(x >= 0) and torch.all(x <= 1), "Input images should be in the range [0, 1]"
        # Start with random noise if random_start is True
        if self.rand_init:
            x_adv = x + torch.empty_like(x).uniform_(-self.eps, self.eps)
        else:
            x_adv = x.clone()
        x_adv = torch.clamp(x_adv, min=0, max=1)
        x_adv.requires_grad = True

        for i in range(self.n):
            self.model.zero_grad()

            # Forward pass
            outputs = self.model(x_adv)

            # Compute the loss
            if targeted:
                loss = -self.loss_func(outputs, y).mean()  # Minimize the negative loss for targeted attack
            else:
                loss = self.loss_func(outputs, y).mean()

            # Backward pass
            loss.backward()

            # Update the adversarial example using the gradient
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()

                # Clip the perturbation to be within the allowed epsilon-ball
                x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)

                # Clip the values to be within the valid data range
                x_adv = torch.clamp(x_adv, 0, 1)

            # Detach the adversarial example for the next iteration
            x_adv = x_adv.detach()
            x_adv.requires_grad = True

            # Early stopping condition
            if self.early_stop:
                preds = torch.argmax(self.model(x_adv), dim=1)
                if targeted:
                    if torch.all(preds == y):  # every sample hit the target
                        break
                else:
                    if torch.all(preds != y):  # every sample misclassified
                        break

        assert torch.all(x >= 0) and torch.all(x <= 1), "Input images should be in the range [0, 1]"
        assert torch.all(x_adv <= x + self.eps) and torch.all(x_adv >= x - self.eps), "Adversarial images should be within the eps-ball centered at x"

        return x_adv


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255., momentum=0.,
                 k=200, sigma=1 / 255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def nes_grad(self, x, y):
        nes_grad_est = torch.zeros_like(x)
        for i in range (self.k):
            delta = torch.randn_like(x) # (Batch_Size, Channels, Width, Height)
            plus = self.model(x + self.sigma * delta) # (Batch_Size, Classes)
            loss_plus = self.loss_func(plus, y) # (Batch_Size)
            estimate_plus = loss_plus.view(-1, 1, 1, 1) * delta # (Batch_Size, Channels, Width, Height)
            minus = self.model(x - self.sigma * delta) # (Batch_Size)
            loss_minus = self.loss_func(minus, y) # (Batch_Size, Classes)
            estimate_minus = loss_minus.view(-1, 1, 1, 1) * delta # (Batch_Size, Channels, Width, Height)
            
            nes_grad_est +=  estimate_plus - estimate_minus # (Batch_Size, Channels, Width, Height)
        return nes_grad_est / (2 * self.k * self.sigma)



    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """

        with torch.no_grad():
            assert torch.all(x >= 0) and torch.all(x <= 1), "Input images should be in the range [0, 1]"
            if self.rand_init:
                x_adv = x +torch.empty_like(x).uniform_(-self.eps, self.eps)
            else:
                x_adv = x.clone()
            x_adv = torch.clamp(x_adv, min=0, max=1)

            num_queries = torch.zeros_like(y)
            grad = torch.zeros_like(x) # For the first step
            for step in range(self.n):
                outputs = self.model(x_adv)
                _, predictions = torch.max(outputs, 1)
                if self.early_stop:
                    hit = predictions == y
                    if torch.all(hit) and targeted or torch.all(~ hit) and not targeted:
                        break
                    still_working = ~ hit if targeted else hit
                else:
                    still_working = y == y

                        
                nes_grad_est = self.nes_grad(x_adv, y)
                if targeted:
                    nes_grad_est = -nes_grad_est
                num_queries += still_working * 2 * self.k
                grad = self.momentum * grad + (1 - self.momentum) * nes_grad_est
                grad = grad * still_working.view(-1, 1, 1 ,1)
                x_adv = x_adv + self.alpha * grad.sign()

                # Clip the perturbation to be within the allowed epsilon-ball
                x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)

                # Clip the values to be within the valid data range
                x_adv = torch.clamp(x_adv, 0, 1)
            
            return x_adv, num_queries
            

class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        assert torch.all(x >= 0) and torch.all(x <= 1), "Input images should be in the range [0, 1]"
        # Start with random noise if random_start is True
        if self.rand_init:
            x_adv = x + torch.empty_like(x).uniform_(-self.eps, self.eps)
        else:
            x_adv = x.clone()
        x_adv = torch.clamp(x_adv, min=0, max=1)
        num_classes = 4
        x_adv.requires_grad = True
        for i in range(self.n):
            for model in self.models:
                model.zero_grad()
            outputs = torch.zeros(x_adv.shape[0], len(self.models), 4, device=y.device)
            for j, model in enumerate(self.models):
                outputs[:, j] = model(x_adv)
            outputs = outputs.mean(dim=1) # In the paper there were different alphas for each model - we will assume they have the same weight

            # Compute the loss
            if targeted:
                loss = -self.loss_func(outputs, y).mean()  # Minimize the negative loss for targeted attack
            else:
                loss = self.loss_func(outputs, y).mean()

            # Backward pass
            loss.backward()

            # Update the adversarial example using the gradient
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()

                # Clip the perturbation to be within the allowed epsilon-ball
                x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)

                # Clip the values to be within the valid data range
                x_adv = torch.clamp(x_adv, 0, 1)

            # Detach the adversarial example for the next iteration
            x_adv = x_adv.detach()
            x_adv.requires_grad = True

            # Early stopping condition
            if self.early_stop:
                with torch.no_grad():
                    outputs = torch.zeros(x_adv.shape[0], len(self.models), num_classes)
                    for j, model in enumerate(self.models):
                        outputs[:, j] = model(x_adv)
                    outputs = outputs.mean(dim=1)
                    if self.loss_func(outputs, y).sum() == 0:
                        break

        assert torch.all(x >= 0) and torch.all(x <= 1), "Input images should be in the range [0, 1]"
        assert torch.all(x_adv <= x + self.eps) and torch.all(x_adv >= x - self.eps), "Adversarial images should be within the eps-ball centered at x"

        return x_adv
        
