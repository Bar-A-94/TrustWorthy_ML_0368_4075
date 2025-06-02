import argparse
import consts
import random
import numpy as np
import models
import defenses
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set_theme()

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def run_evaluation(sigma):

    # load model
    trained_models = {}
    mpath = f'trained-models/simple-cnn-sigma-{sigma:0.4f}'
    model = models.SimpleCNN()
    model.load_state_dict(torch.load(mpath))
    model.eval()
    model.to(device)

    # load test data
    data_test = utils.TMLDataset('test', transform=transforms.ToTensor())
    loader_test = DataLoader(data_test,
                             batch_size=1,
                             shuffle=True,
                             num_workers=2)

    # init smoothed model
    smoothed_model = defenses.SmoothedModel(model, sigma)

    # find certified radius per sample
    cert_radii = []
    for (x,y) in tqdm(loader_test):
        x, y = x.to(device), y.to(device)
        pred, radius = smoothed_model.certify(x,
                                              consts.RS_N0,
                                              consts.RS_N,
                                              consts.RS_ALPHA,
                                              consts.BATCH_SIZE)
        cert_radii.append(radius if pred==y else 0.)
    
    # done
    return cert_radii
        

def plot_radii(radii):
    x = []
    y = []  # accuracy
    # derive x and y from the certified radii - FILL ME
    radii = np.array(radii)
    uniq_radii = sorted(list(set(radii)))  # radius
    for radius in uniq_radii:
        certified_samples = radii[np.logical_and(radii >= radius, radii > 0)]
        y.append(len(certified_samples) / len(radii))
        x.append(radius)
    # plot
    plt.plot(x, y)

if __name__=='__main__':
    sigmas = [0.05, 0.20]
    radii = {}
    for sigma in sigmas:
        print(f'Certifying L2 radii with sigma={sigma:0.4f}')
        radii[sigma] = run_evaluation(sigma)

    # plot
    plt.figure()
    for sigma in sigmas:
        plot_radii(radii[sigma])
    plt.xlabel('certified L2 radius')
    plt.ylabel('accuracy')
    plt.legend(sigmas, title='sigma')
    plt.savefig('randomized-smoothing-acc-vs-radius.pdf')
