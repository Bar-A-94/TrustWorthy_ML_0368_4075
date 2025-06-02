import argparse
import consts
import random
import numpy as np
import models
import defenses
import utils

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def evaluate_accuracy():
    # init data loader
    data_test = utils.TMLDataset('test', transform=transforms.ToTensor())
    loader_test = DataLoader(data_test,
                             batch_size=consts.BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

    
    for model_id in range(2):
        # load model
        mpath = f'trained-models/simple-cnn-part-c-{model_id}'
        model = models.SimpleCNN()
        model.load_state_dict(torch.load(mpath))
        model.eval()
        model.to(device)

        # compute and print accuracy
        acc = utils.compute_accuracy(model, loader_test, device)
        print(f'Accuracy of model {model_id}: {acc:0.4f}')

def run_neural_cleanse():
    # init data loader
    data_test = utils.TMLDataset('test', transform=transforms.ToTensor())
    loader_test = DataLoader(data_test,
                             batch_size=consts.BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

    # dictionaries for all masks and triggers
    masks, triggers = {}, {}
    for model_id in range(2):
        # dictionaries for masks and triggers for model_id
        masks[model_id], triggers[model_id] = {}, {}
        # load model
        mpath = f'trained-models/simple-cnn-part-c-{model_id}'
        model = models.SimpleCNN()
        model.load_state_dict(torch.load(mpath))
        model.eval()
        model.to(device)

        # init NeuralCleanse
        nc = defenses.NeuralCleanse(model)

        # find mask + trigger targeting each potential class
        for c_t in range(4):
            mask, trigger = nc.find_candidate_backdoor(c_t, loader_test, device)
            masks[model_id][c_t] = mask.to('cpu')
            triggers[model_id][c_t] = trigger.to('cpu')
            norm = mask.sum()
            print(f'Norm of trigger targeting class {c_t} in model {model_id}: {norm:0.4f}')

    # ask for user input
    selected_mid = int(input('Which model is backdoored (0/1)? '))
    selected_c_t = int(input('Which class is the backdoor targeting (0/1/2/3)? '))

    return selected_mid, \
        masks[selected_mid][selected_c_t], \
        triggers[selected_mid][selected_c_t], \
        selected_c_t
    
        
def evaluate_backdoor_success(model_id, mask, trigger, c_t):
    # init data loader
    data_test = utils.TMLDataset('test', transform=transforms.ToTensor())
    loader_test = DataLoader(data_test,
                             batch_size=consts.BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

    # load model
    mpath = f'trained-models/simple-cnn-part-c-{model_id}'
    model = models.SimpleCNN()
    model.load_state_dict(torch.load(mpath))
    model.eval()
    model.to(device)

    # compute and print accuracy
    sr = utils.compute_backdoor_success_rate(model,
                                             loader_test,
                                             device,
                                             mask,
                                             trigger,
                                             c_t)
    print(f'Backdoor success rate: {sr:0.4f}')
        
if __name__=='__main__':
    # evaluate the accuracy of the two models
    evaluate_accuracy()

    # run NeuralCleanse to find backdoor
    backdoored_model_id, mask, trigger, c_t = run_neural_cleanse()

    # save mask and trigger
    mask_im = mask.detach().numpy().squeeze().transpose((1, 2, 0))
    utils.save_as_im(mask_im, 'backdoor-mask.jpg')
    trigger_im = (mask*trigger).detach().numpy().squeeze().transpose((1, 2, 0))
    utils.save_as_im(trigger_im, 'backdoor-trigger.jpg')
    
    # evaluate the success rate
    mask = mask.to(device)
    trigger = trigger.to(device)
    evaluate_backdoor_success(backdoored_model_id, mask, trigger, c_t)
