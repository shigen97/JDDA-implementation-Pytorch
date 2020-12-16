import os
import pickle
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

def load_mnist(image_dir, split='train'):
    print('Loading MNIST dataset.')

    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join('data', image_dir, image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f, encoding='bytes')
    keys = list(mnist.keys())
    mnist[str(keys[0])] = mnist.pop(keys[0])
    mnist[str(keys[1])] = mnist.pop(keys[1])
    images = mnist["b'X'"] / 127.5 - 1
    labels = mnist["b'y'"]
    labels = np.squeeze(labels).astype(int)
    return images, labels


def load_svhn(image_dir, split='train'):
    print ('Loading SVHN dataset.')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join('data', image_dir, image_file)
    svhn = loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
    # ~ images= resize_images(images)
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels


def load_usps(image_dir,split='train'):
    print('Loading USPS dataset.')
    image_file='USPS_train.pkl' if split=='train' else 'USPS_test.pkl'
    image_dir=os.path.join('data', image_dir,image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f, encoding='bytes')
    keys = list(usps.keys())
    usps[str(keys[0])] = usps.pop(keys[0])
    usps[str(keys[1])] = usps.pop(keys[1])
    images = usps["b'data'"]

    labels = usps["b'label'"]
    labels=np.squeeze(labels).astype(int)
    return images,labels



def load_syn(image_dir,split='train'):
    print('load syn dataset')
    image_file='synth_train_32x32.mat' if split=='train' else 'synth_test_32x32.mat'
    image_dir=os.path.join('data', image_dir,image_file)
    syn = loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2]) / 127.5 - 1
    labels = syn['y'].reshape(-1)
    return images,labels


def load_mnistm(image_dir,split='train'):
    print('Loading mnistm dataset.')
    image_file='mnistm_train.pkl' if split=='train' else 'mnistm_test.pkl'
    image_dir=os.path.join('data', image_dir,image_file)
    with open(image_dir, 'rb') as f:
        mnistm = pickle.load(f, encoding='bytes')
    keys = list(mnistm.keys())
    mnistm[str(keys[0])] = mnistm.pop(keys[0])
    mnistm[str(keys[1])] = mnistm.pop(keys[1])
    images = mnistm["b'data'"]

    labels = mnistm["b'label'"]
    labels=np.squeeze(labels).astype(int)
    return images, labels


def get_dataloader(dataset_name, batch_size, split='train'):
    assert dataset_name in ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
    load_data_func = {
        'mnist': load_mnist,
        'mnistm': load_mnistm,
        'svhn': load_svhn,
        'syn': load_syn,
        'usps': load_usps
    }
    x, y = load_data_func[dataset_name](dataset_name, split)
    x, y = torch.from_numpy(x), torch.from_numpy(y).long()
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 3)
    x = torch.transpose(x, 3, 2)
    x = torch.transpose(x, 2, 1)
    input_transform = transforms.Compose([
    transforms.Grayscale(1)
    ])
    if x.shape[1] == 3:
        x = input_transform(x)
    x = x.float()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return dataloader

