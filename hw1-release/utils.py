import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random



def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    correct = 0
    total = 0
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        outputs = model(x)
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == y).sum().item()
        total += x.shape[0]
    return correct/total
        


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    all_x = []
    all_y = []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted:
            t = (y + torch.randint(1, n_classes, y.shape, device=y.device)) % n_classes
            x_adv = attack.execute(x, t, targeted)
            all_y.append(t)
        else:
            x_adv = attack.execute(x, y, targeted)
            all_y.append(y)
        all_x.append(x_adv)
    x_adv = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)
    return x_adv, y



def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    all_x = []
    all_y = []
    all_num_queries = []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted:
            t = (y + torch.randint(1, n_classes, y.shape, device=y.device)) % n_classes
            x_adv, num_queries = attack.execute(x, t, targeted)
            all_y.append(t)
        else:
            x_adv, num_queries = attack.execute(x, y, targeted)
            all_y.append(y)
        all_x.append(x_adv)
        all_num_queries.append(num_queries)
    x_adv = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)
    num_queries = torch.cat(all_num_queries, dim=0)
    return x_adv, y, num_queries


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    correct = 0
    x_adv = x_adv.to(device)
    y = y.to(device)
    outputs = model(x_adv)
    _, predictions = torch.max(outputs, dim=1)
    if targeted:
        correct += (predictions == y).sum().item()
    else:
        correct += (predictions != y).sum().item()
    return correct/x_adv.shape[0]



def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    # Ensure the weight is in float32 format
    w = torch.tensor(w, dtype=torch.float32).item()
    # Convert the float32 number to its binary representation
    binary_repr = binary(w)

    # Choose a random bit to flip
    bit_idx = random.randint(0, 31)
    # Flip the chosen bit
    flipped_bit = '0' if binary_repr[bit_idx] == '1' else '1'
    flipped_binary_repr = binary_repr[:bit_idx] + flipped_bit + binary_repr[bit_idx + 1:]

    # Convert the binary representation back to float32
    flipped_w = float32(flipped_binary_repr)

    return flipped_w, bit_idx

def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    # Convert float32 to bytes
    bytes_val = struct.pack('!f', num)
    
    # Convert bytes to binary string
    binary_str = ''.join(format(byte, '08b') for byte in bytes_val)
    
    return binary_str


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    # Convert binary string to bytes
    binary_bytes = int(binary, 2).to_bytes(4, byteorder='big')
    
    # Convert bytes back to float32 using struct
    float_val = struct.unpack('!f', binary_bytes)[0]
    
    return float_val


def random_bit_flip(w):
#     """
#     This functoin receives a weight in float32 format, picks a
#     random bit to flip in it, flips the bit, and returns:
#     1- The weight with the bit flipped
#     2- The index of the flipped bit in {0, 1, ..., 31}
#     """
    binary_w = binary(w)
    # Pick a random bit to flip
    bf_idx = np.random.randint(0, 32)
    # Flip the bit
    flipped_bit = '1' if binary_w[bf_idx] == '0' else '0'
    binary_w = binary_w[:bf_idx] + flipped_bit + binary_w[bf_idx + 1:]
    # Convert back to float32
    flipped_w = float32(binary_w)
    return flipped_w, bf_idx
