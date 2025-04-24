from utils import load_pretrained_cnn, TMLDataset, compute_accuracy, \
    compute_attack_success, run_whitebox_attack, run_blackbox_attack
import consts
import torch
import torchvision.transforms as transforms
from attacks import NESBBoxPGDAttack, PGDAttack
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# load model and dataset
model = load_pretrained_cnn(0)
model.to(device)
model.eval()
dataset = TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# test accuracy
acc = compute_accuracy(model, data_loader, device)
print(f'The test accuracy of the model is: {acc:0.4f}')

# init attacks
wb_attack = PGDAttack(model)
bb_attack = NESBBoxPGDAttack(model)

# execute white-box
print('White-box attack:')
for targeted in [False, True]:
    x_adv, y = run_whitebox_attack(wb_attack, data_loader, targeted, device)
    sr = compute_attack_success(model, x_adv, y, consts.BATCH_SIZE, targeted, device)
    if targeted:
        print(f'\t- targeted success rate: {sr:0.4f}')
    else:
        print(f'\t- untargeted success rate: {sr:0.4f}')

# excecute targeted and untargeted black-box attacks w/ and wo/ momentum
n_queries_all = []
for momentum in [0, 0.9]:
    for targeted in [False, True]:
        bb_attack.momentum = momentum
        x_adv, y, n_queries = run_blackbox_attack(bb_attack, data_loader, targeted, device)
        sr = compute_attack_success(model, x_adv, y, consts.BATCH_SIZE, targeted, device)
        median_queries = torch.median(n_queries)
        if targeted:
            print(f'Targeted black-box attack (momentum={momentum:0.2f}):')
        else:
            print(f'Untargeted black-box attack (momentum={momentum:0.2f}):')
        print(f'\t- success rate: {sr:0.4f}\n\t- median(# queries): {median_queries}')
        n_queries_all.append(n_queries.detach().to('cpu'))

# box-plot # queries wo/ and w/ momentum for untargeted attacks
plt.figure()
plt.boxplot([n_queries_all[0], n_queries_all[2]])
plt.xticks(range(1, 3), ['0.0', '0.9'])
plt.title('untargeted')
plt.xlabel('momentum')
plt.ylabel('# queries')
plt.savefig('bbox-n_queries_untargeted.jpg')

# box-plot # queries wo/ and w/ momentum for targeted attacks
plt.figure()
plt.boxplot([n_queries_all[1], n_queries_all[3]])
plt.xticks(range(1, 3), ['0.0', '0.9'])
plt.title('targeted')
plt.xlabel('momentum')
plt.ylabel('# queries')
plt.savefig('bbox-n_queries_targeted.jpg')
