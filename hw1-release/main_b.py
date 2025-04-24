from utils import load_pretrained_cnn, TMLDataset, compute_accuracy, \
    run_whitebox_attack, compute_attack_success
import consts
import torch
import torchvision.transforms as transforms
from attacks import PGDAttack, PGDEnsembleAttack
import random
import numpy as np

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# load models and dataset
models = []
for i in range(3):
    model = load_pretrained_cnn(i)
    model.to(device)
    model.eval()
    models.append(model)
dataset = TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# model accuracy
for i, model in enumerate(models):
    acc = compute_accuracy(model, data_loader, device)
    print(f'Test accuracy of model {i}: {acc:0.4f}')

# init attacks
attacks = [PGDAttack(models[i], eps=8 / 255., early_stop=False) for i in range(3)]

# untargeted attacks
transfer_success = np.zeros((3, 3))
for i in range(3):  # src
    x_adv, y = run_whitebox_attack(attacks[i], data_loader, False, device)
    for j in range(3):  # target
        sr = compute_attack_success(models[j], x_adv, y, consts.BATCH_SIZE, False, device)
        transfer_success[i, j] = sr
print('Untargeted attacks\' transferability:')
with np.printoptions(precision=4):
    print(transfer_success)

# targeted attacks
transfer_success = np.zeros((3, 3))
for i in range(3):  # src
    x_adv, y = run_whitebox_attack(attacks[i], data_loader, True, device)
    for j in range(3):  # target
        sr = compute_attack_success(models[j], x_adv, y, consts.BATCH_SIZE, True, device)
        transfer_success[i, j] = sr
print('Targeted attacks\' transferability:')
with np.printoptions(precision=4):
    print(transfer_success)

# ensemble attacks against model 0
attack = PGDEnsembleAttack(models[1:], eps=8 / 255., early_stop=False)

# untargeted attacks
x_adv, y = run_whitebox_attack(attack, data_loader, False, device)
sr_untarg = compute_attack_success(models[0], x_adv, y, consts.BATCH_SIZE, False, device)

# targeted attacks
x_adv, y = run_whitebox_attack(attack, data_loader, True, device)
sr_targ = compute_attack_success(models[0], x_adv, y, consts.BATCH_SIZE, True, device)

# results
print('Ensemble attacks\' transferability from models 1+2 to model 0:')
print(f'\t- untargeted attack: {sr_untarg:0.4f}')
print(f'\t- targeted attack: {sr_targ:0.4f}')
