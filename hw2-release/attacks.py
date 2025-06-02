import torch
import torch.nn as nn
import torch.nn.functional as F

def carlini_wagner_loss(outputs, y, large_const=1e6):
    y = F.one_hot(y, outputs.shape[1])
    logits_y = torch.sum(torch.mul(outputs, y), 1)
    logits_max_non_y, _ = torch.max((outputs-large_const* y), 1)
    return logits_max_non_y - logits_y

class PGDAttack:

    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True, loss='ce'):
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        if loss=='ce':
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        elif loss=='cw':
            self.loss_func = carlini_wagner_loss

    def execute(self, x, y, targeted=False):

        # param to control early stopping
        allow_update = torch.ones_like(y)
        
        # init
        x_adv = torch.clone(x)
        x_adv.requires_grad = True
        if self.rand_init:
            x_adv.data = x_adv.data + self.eps*(2*torch.rand_like(x_adv)-1)
            x_adv.data = torch.clamp(x_adv, x-self.eps, x+self.eps)
            x_adv.data = torch.clamp(x_adv, 0., 1.)

        for i in range(self.n):
            # get grad
            outputs = self.model(x_adv)
            loss = torch.mean(self.loss_func(outputs, y))
            loss.backward()
            g = torch.sign(x_adv.grad)

            # early stopping
            if self.early_stop:
                g = torch.mul(g, allow_update[:, None, None, None])

            # pgd step
            if not targeted:
                x_adv.data += self.alpha*g
            else:
                x_adv.data -= self.alpha*g
            x_adv.data = torch.clamp(x_adv, x-self.eps, x+self.eps)
            x_adv.data = torch.clamp(x_adv, 0., 1.)

            # attack success rate
            with torch.no_grad():
                outputs = self.model(x_adv)
                _, preds = torch.max(outputs, 1)
                if not targeted:
                    success = preds!=y
                else:
                    success = (preds==y)
                # early stopping
                allow_update = allow_update - allow_update*success
                if self.early_stop and torch.sum(allow_update)==0:
                    break

        # done
        return x_adv

