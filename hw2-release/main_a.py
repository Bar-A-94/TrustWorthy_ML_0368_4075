import argparse
import consts
import random
import numpy as np
import models
import defenses
import attacks
import utils
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, help='1 for adversarial training, 0 for evaluating')
    return parser.parse_args()

def run_standard_training():
    """
    Run standard training
    """

    # load training set
    transforms_tr = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(8),
                                        transforms.RandomResizedCrop((32,32))])
    data_tr = utils.TMLDataset('train', transform=transforms_tr)

    # init model
    model = models.SimpleCNN()
    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # execute training
    t0 = time.time()
    model = utils.standard_train(model, data_tr, criterion, optimizer, \
                                 scheduler, device)
    train_time = time.time()-t0

    # move model to cpu and store it
    model.to('cpu')
    torch.save(model.state_dict(), \
               'trained-models/simple-cnn')

    # done
    return train_time


def run_free_adv_training():
    """
    Run free adversarial training
    """

    # load training set
    transforms_tr = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(8),
                                        transforms.RandomResizedCrop((32,32))])
    data_tr = utils.TMLDataset('train', transform=transforms_tr)

    # init model
    model = models.SimpleCNN()
    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # execute training
    t0 = time.time()
    model = defenses.free_adv_train(model, data_tr, criterion, optimizer, \
                                    scheduler, consts.PGD_Linf_EPS, device)
    train_time = time.time()-t0

    # move model to cpu and store it
    model.to('cpu')
    torch.save(model.state_dict(), \
               'trained-models/simple-cnn-free-adv-trained')

    # done
    return train_time


def run_evaluation():

    # load standard and adversarially trained mdoels
    trained_models = {}
    mpaths = {'standard': 'trained-models/simple-cnn',
              'adv_trained': 'trained-models/simple-cnn-free-adv-trained'}
    for mtype in mpaths:
        model = models.SimpleCNN()
        model.load_state_dict(torch.load(mpaths[mtype]))
        model.eval()
        model.to(device)
        trained_models[mtype] = model

    # load test data
    data_test = utils.TMLDataset('test', transform=transforms.ToTensor())
    loader_test = DataLoader(data_test,
                             batch_size=consts.BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

    # model accuracy
    print('Model accuracy:')
    for mtype in trained_models:
        acc = utils.compute_accuracy(trained_models[mtype],
                                     loader_test,
                                     device)
        print(f'\t- {mtype:12s}: {acc:0.4f}')

    # adversarial robustness
    print('Success rate of untargeted white-box PGD:')
    for mtype in trained_models:
        model = trained_models[mtype]
        attack = attacks.PGDAttack(model, eps=consts.PGD_Linf_EPS)
        x_adv, y = utils.run_whitebox_attack(attack, loader_test, False, device, n_classes=4)
        sr = utils.compute_attack_success(model, x_adv, y, consts.BATCH_SIZE, False, device)
        print(f'\t- {mtype:10s}: {sr:0.4f}')
        

if __name__=='__main__':
    args = parse_arguments()
    if args.train:
        print('Training standard model...')
        # t = run_standard_training()
        # print(f'Time (in seconds) to complete standard training: {t:0.4f}')
        print('Adversarially training a model...')
        t = run_free_adv_training()
        print(f'Time (in seconds) to complete free adversarial training: {t:0.4f}')
    else:
        run_evaluation()
