# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:08:56 2021

@author: Zhao Ji Wang
"""

import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def predict_image(img, model):
    xb = img
    yb = model(xb)
    yb = F.softmax(yb, dim = 1)
    print(yb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

#Load in own test image
img_path="My_Digit.png"
image = Image.open(img_path)
image = image.convert("L")
image = ToTensor()(image).unsqueeze(0) # unsqueeze to add artificial first dimension
# image = Variable(image)
print("Image Shape: ", image.shape)


batch_size = 128



dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor(), download = True) 
train_ds, val_ds = random_split(dataset, [50000, 10000])

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

# Defining Model

input_size = 28*28
num_classes = 10

# Logistic regression model
model = MnistModel(); 

model.load_state_dict(torch.load('mnist-logistic.pth'))

# print(model.state_dict())

print("Predicted: ", predict_image(image, model)) 

# history1 = fit(20, 0.001, model, train_loader, val_loader)

torch.save(model.state_dict(), 'mnist-logistic.pth')

