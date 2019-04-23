from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import copy
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Paramètres du réseau
BATCH_SIZE = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
Création du Dataset qui sera utilisé dans Pytorch.
'''
class SignaturesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.signatures_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.signatures_csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.signatures_csv.iloc[idx, 0])
        image = io.imread(img_name)
        genuine = self.signatures_csv.iloc[idx,2] #== 'True')
        
        if self.transform:
            image = self.transform(image)
        sampletuple = (image, int(genuine))
        return sampletuple


"""
Redimensionne les images.
ATTENTION ! Pour le moment le ratio n'est pas conservé, les images peuvent être déformée...
A changer.
output_size: tuple ou int.
Si tuple, la taille de l'image est matchée. Première valeur = height, deuxième valeur = width.
Si int, le plus petit côté match la valeur, et le ratio est conservé.
"""
class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


"""
Rogne de façon aléatoire l'image.
Je ne sais pas encore si on va se servir de cette transformation mais je la laisse là au cas où.
Argument : output_size (tuple ou int), taille du rognage. Si int un rognage carré est effectué.
"""
class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                      left: left + new_w]
        return image


"""Convert ndarrays in sample to Tensors."""
class ToTensor(object):

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

dataset = SignaturesDataset(csv_file='data/allSignatures.csv', root_dir='data/',
    transform=transforms.Compose([Rescale((IMG_HEIGHT, IMG_WIDTH)), ToTensor(), normalize]))

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
training_size = int(dataset_size*0.9)
valid_size = int(dataset_size*0.1)
shuffle_dataset = True
if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, val_indices = indices[valid_size:], indices[:valid_size]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                dataloader = train_dataloader
                dataset_size = training_size
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = valid_dataloader
                dataset_size = valid_size

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f} Correct: {} out of {}'.format(
                phase, epoch_loss, epoch_acc,running_corrects,dataset_size))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model)
    return model

losses = []
accuracies = []
best_precision = 0



vgg16 = models.vgg16(pretrained=True)
# Load the pretrained model from pytorch
# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[-1].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(vgg16.parameters(), lr = 0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

vgg16 = vgg16.to(device)  # if you have access to a gpu
model = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)