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
from g1020_preprocessor import get_loader
from dataloader import KIGenerator,G1020Dataset,g1020KIGenerator
import argparse
import pickle
import tensorflow as tf
import gdown
import zipfile
from results_visualize import report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Add, Reshape,Conv2D,MaxPooling2D,ZeroPadding2D,experimental,AveragePooling2D,Concatenate,Dropout,Activation
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Knowledge Incorporation')
parser.add_argument('--data_dir', default='archive', type=str)
parser.add_argument('--dataset', default='archive', type=str)
parser.add_argument('--model','-m', default='resnet', type=str)
parser.add_argument('--num_classes','-v', default=23, type=int)
parser.add_argument('--batch_size','-e', default=16, type=int)
parser.add_argument('--input_size','-i', default=16, type=int)
args = parser.parse_args()  

model = args.model
data_dir = args.data_dir
num_classes = args.num_classes
batch_size = args.batch_size
dataset = args.dataset
input_size = args.input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs=20):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #print(dataloaders[phase].classes())
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def load_model(model_path,model):
    if model == 'densenet':
        dn = models.densenet169(pretrained=False)
        num_ftrs = dn.classifier.in_features
        dn.classifier = nn.Linear(num_ftrs, num_classes)
    elif model == 'resnet':
        dn = models.resnet152(pretrained=True)
        num_ftrs = dn.fc.in_features
        dn.fc = nn.Linear(num_ftrs, num_classes)

    dn.load_state_dict(torch.load(model_path+'/'+dataset+ '/'+ model + '/best.pth'))
    return dn

def save_outputs_g1020(dn,fold,data_path,directory):
    outputs = []
    for phase in ['train' , 'test']:
      for images in fold["validation" if phase == 'test' else 'train']:
        image_path = os.path.join(data_path,images[:-4]+'.png')
        img = Image.open(image_path)
        img.load()
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask = img.split()[3])
        img = data_transforms[phase](background)
        img = torch.unsqueeze(img,axis = 0)
        dn.eval()
        out = dn(img).detach().cpu().numpy()
        np.save(os.path.join(directory,images[:-4]+'.npy'),out)

def save_outputs(model_ft,data_dir,save_dir):
    model_ft = model_ft.to(device)
    k1 = {}
    train_dir = data_dir + '/train'
    img_list = os.listdir(train_dir)
    img_list.sort()
    for i,path in enumerate(img_list):
        for img in os.listdir(os.path.join(train_dir,path)):
            final = os.path.join(train_dir,path,img)
            img = Image.open(final)
            img = data_transforms['train'](img)
            img = torch.unsqueeze(img,axis = 0)
            img = img.to(device)
            output = model_ft(img)
            k1[final] = output.detach().cpu().numpy()

    an_file = open(save_dir + "/train_outputs.pkl", "wb")
    pickle.dump(k1, an_file)
    an_file.close()

    k2 = {}
    test_dir = data_dir + '/test'

    img_list = os.listdir(test_dir)
    img_list.sort()
    for i,path in enumerate(img_list):
        for img in os.listdir(os.path.join(test_dir,path)):
            final = os.path.join(test_dir,path,img)
            img = Image.open(final)
            img = data_transforms['test'](img)
            img = torch.unsqueeze(img,axis = 0)
            img = img.to(device)
            output = model_ft(img)
            k2[final] = output.detach().cpu().numpy()

    an_file = open(save_dir + "/test_outputs.pkl", "wb")
    pickle.dump(k2, an_file)
    an_file.close()

def get_data_IDs(data_dir):
    train_dir = './archive/train'
    test_dir = './archive/test'
    img_list = os.listdir(train_dir)
    img_list.sort()
    list_IDs = []
    labels = {}
    for i,path in enumerate(img_list):
        for img in os.listdir(os.path.join(train_dir,path)):
            final = os.path.join(train_dir,path,img)
            list_IDs.append(final)
            labels[final] = i

            
    img_list = os.listdir(test_dir)
    img_list.sort()
    list_IDs2 = []
    labels2 = {}
    for i,path in enumerate(img_list):
        for img in os.listdir(os.path.join(test_dir,path)):
            final = os.path.join(test_dir,path,img)
            list_IDs2.append(final)
            labels2[final] = i
    return list_IDs,list_IDs2,labels,labels2

def get_outputs_final_g1020(model_path,fold):
    model = tf.keras.models.load_model(model_path)
    y_true = []
    outputs = []
    know1 = []
    know2 = []
    for x in fold['test']:
        y_true.append(labels[str(x)])
        output = model([np.load(os.path.join("./runs",dataset,'resnet',x[:-4]+'.npy')),np.load(os.path.join("./runs",dataset,'densenet',x[:-4]+'.npy'))])
        outputs.append(np.argmax(output,axis = -1))
        know1.append(np.load(os.path.join("./runs",dataset,'resnet',x[:-4]+'.npy')))
        know2.append(np.load(os.path.join("./runs",dataset,'densenet',x[:-4]+'.npy')))

    return y_true,outputs,know1,know2 
def get_outputs(model_path):
    model = tf.keras.models.load_model(model_path)
    test_dir = './archive/test'
    y_true = []
    outputs = []
    know1 = []
    know2 = []
    a_file = open(os.path.join('./runs',dataset,"resnet","test_outputs.pkl"), "rb")
    test_res_dict = pickle.load(a_file)
    a_file = open(os.path.join('./runs',dataset,"densenet","test_outputs.pkl"), "rb")
    test_dense_dict = pickle.load(a_file)
    img_list = os.listdir(test_dir)
    img_list.sort()
    for i,path in enumerate(img_list):
        for img in os.listdir(os.path.join(test_dir,path)):
            final = os.path.join(test_dir,path,img)
            list_IDs2.append(final)
            y_true.append(i)
            k1 = test_res_dict[final]
            k2 = test_dense_dict[final]
            know1.append(k1)
            know2.append(k2)
            output = model([k1,k2])
            output = np.argmax(output,axis = -1)
            outputs.append(output)
    return y_true,outputs,know1,know2
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
url = data_dir
gdown.download(url, quiet=True,fuzzy = True)

if dataset == "g1020":
    with zipfile.ZipFile('g1020_polygons.zip', 'r') as zip_ref:
      zip_ref.extractall('./')
    folds,labels = get_loader('./g1020-polygons')
    print(folds)
    image_datasets = {x: G1020Dataset(folds[4]['train'],labels, data_transforms[x]) for x in ['train', 'test']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
else:
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
      zip_ref.extractall('./archive/')
    image_datasets = {x: datasets.ImageFolder(os.path.join('./archive', x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}

for model in ["resnet","densenet"]:
    if model == "densenet":
        model_ft = models.densenet169(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model == "resnet":
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=20)
    path = os.path.join("./runs",dataset,model)

    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model_ft.state_dict(),path+'/best.pth')
    if dataset == "g1020":
        save_outputs_g1020(load_model('./runs',model),folds[4],'./g1020-polygons',path)
    else:
        save_outputs(load_model('./runs',model),'./archive',path)


def knowledge_incorporated_model(num_classes):#pretrained_model):    
    inputA = Input((num_classes))
    inputB = Input((num_classes))
    inputA = Activation('softmax')(inputA)
    inputB = Activation('softmax')(inputB)
    l1    = Add()([inputA,inputB])
    l1    = Dense(32)(l1)
    l2    = Activation('relu')(l1)
    l4    = Dense(64)(l2)
    l5    = Activation('relu')(l4)
    l6    = Dense(num_classes)(l5)
    l7    = Activation('softmax')(l6)
    l8    = Add()([l7,inputB])
    l9    = Activation('softmax')(l8)
    return Model(inputs=[inputA, inputB], outputs=l9)


if dataset == "g1020":
    training_generator = g1020KIGenerator(folds[4]['train'], labels)
    validation_generator = g1020KIGenerator(folds[4]['validation'], labels)
else:
    list_IDs,list_IDs2,labels,labels2 = get_data_IDs(data_dir)
    a_file = open(os.path.join('./runs',dataset,"resnet","train_outputs.pkl"), "rb")
    train_res_dict = pickle.load(a_file)
    a_file = open(os.path.join('./runs',dataset,"resnet","test_outputs.pkl"), "rb")
    test_res_dict = pickle.load(a_file)
    a_file = open(os.path.join('./runs',dataset,"densenet","train_outputs.pkl"), "rb")
    train_dense_dict = pickle.load(a_file)
    a_file = open(os.path.join('./runs',dataset,"densenet","test_outputs.pkl"), "rb")
    test_dense_dict = pickle.load(a_file)
    train_generator = KIGenerator(list_IDs, labels,train_res_dict,train_dense_dict)
    validation_generator = KIGenerator(list_IDs2, labels2,test_res_dict,test_dense_dict)   
opt_rms = tf.keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
final_model = knowledge_incorporated_model(num_classes)
final_model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
history_final = final_model.fit_generator(train_generator,validation_data=validation_generator,epochs = 20)
tf.keras.models.save_model(final_model,'./runs/'+dataset+'/g1020/kI.pth')
if dataset == "g1020":
    y_true,preds,know1,know2 = get_outputs_final_g1020('./runs/'+dataset+'/g1020/kI.pth',folds[4]['validation'])
else:
    y_true,preds,know1,know2 = get_outputs('./runs/'+dataset+'/g1020/kI.pth')
report(y_true,preds,know1,know2)


#g1020 = '/content/drive/MyDrive/g1020-polygons'
#other = '/content'