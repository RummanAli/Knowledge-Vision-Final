
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
import pandas as pd
import numpy as np
import keras
import tensorflow
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sklearn
from PIL import Image


class KIGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,knowledge_res,knowledge_dense, batch_size=100, dim=(224,224), n_channels=3,
                 n_classes=23, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.knowledge_res = knowledge_res
        self.knowledge_dense = knowledge_dense
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        k_res = np.empty((self.batch_size,23), dtype=float)
        k_dense = np.empty((self.batch_size,23), dtype=float)
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            k_res[i] = self.knowledge_res[ID]
            k_dense[i] = self.knowledge_dense[ID]
            y[i] = self.labels[ID]
            
        return [k_dense,k_res], tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

        
class g1020KIGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=2, dim=(299,299), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        k = np.empty((self.batch_size,2))
        k2 = np.empty((self.batch_size,2))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = Image.open('./g1020-polygons/' + ID[:-4] + '.png')
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask = img.split()[3])
            #X[i,] = np.array(background.resize((299, 299), Image.ANTIALIAS))

            # Store class
            y[i] = self.labels[ID]
            k[i] = np.load('./runs/g1020/resnet/'+str(ID[:-4])+'.npy',allow_pickle= True)
            k2[i]= np.load('./runs/g1020/densenet/'+str(ID[:-4])+'.npy',allow_pickle= True)
        return [k2,k], tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)


class G1020Dataset(Dataset):


    def __init__(self, list_IDs, labels,transforms, batch_size=16, dim=(299,299), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transforms
        self.n_channels = n_channels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.list_IDs[idx]
        img = Image.open('./g1020-polygons/' + img_name[:-4] + '.png')
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask = img.split()[3])
        label = self.labels[img_name]
        sample = {'image': background, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample['image'],sample['label']
