import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class TMLDataset(Dataset):
    def __init__(self, part, fpath='dataset-full.npz', transform=None):
        # init
        with gzip.open(fpath, 'rb') as fin:
            data_tr = np.load(fin, allow_pickle=True)
            data_test = np.load(fin, allow_pickle=True)
        if part=='train':
            self.data = data_tr
        elif part=='test':
            self.data = data_test
        else:
            raise ValueError(f'Unknown dataset part {part}')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
   
def standard_train(model, data_tr, criterion, optimizer, lr_scheduler, device,
                   epochs=100, batch_size=128, dl_nw=10):
    """
    Standard model training.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - device: device used for training
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

    # train
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        for i, data in enumerate(loader_tr, 0):
            # get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize - FILL ME
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                    
        # update scheduler - FILL ME
        lr_scheduler.step()
        

    # done
    return model

def compute_accuracy(model, data_loader, device):
    count_correct = 0
    count_all = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            count_correct += torch.sum(y==preds).to('cpu')
            count_all += len(x)
    return count_correct/float(count_all)

def compute_backdoor_success_rate(model, data_loader, device,
                                  mask, trigger, c_t):
    count_success = 0
    count_all = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data[0], data[1]
            x = x[y!=c_t]
            if len(x)<1:
                continue
            x = data[0].to(device)
            x = x*(1-mask) + mask*trigger
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            count_success += torch.sum(c_t==preds.to('cpu')).item()
            count_all += len(x)
    return count_success/float(count_all)

def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    x_adv_all, y_all = [], []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted:
            y = (y + torch.randint(low=1, high=n_classes, size=(len(y),), device=device))%n_classes
        x_adv = attack.execute(x, y, targeted=targeted)
        x_adv_all.append(x_adv)
        y_all.append(y)
    return torch.cat(x_adv_all), torch.cat(y_all)

def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    count_success = 0
    x_adv.to(device)
    y.to(device)
    with torch.no_grad():
        for i in range(0, len(x_adv), batch_size):
            x_batch = x_adv[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            if not targeted:
                count_success += torch.sum(y_batch!=preds).detach()
            else:
                count_success += torch.sum(y_batch==preds).detach()
    return count_success/float(len(x_adv))

def save_as_im(x, outpath):
    """
    Used to store a numpy array (with values in [0,1] as an image).
    Outpath should specify the extension (e.g., end with ".jpg").
    """
    im = Image.fromarray((x*255.).astype(np.uint8)).convert('RGB')
    im.save(outpath)
