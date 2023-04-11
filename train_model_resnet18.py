#!/usr/bin/env python
# coding: utf-8

# # Collision Avoidance - Train Model (ResNet18)
# 
# Welcome to this host side Jupyter Notebook!  This should look familiar if you ran through the notebooks that run on the robot.  In this notebook we'll train our image classifier to detect two classes
# ``free`` and ``blocked``, which we'll use for avoiding collisions.  For this, we'll use a popular deep learning library *PyTorch*

# In[1]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

#!unzip -q dataset.zip


dataset = datasets.ImageFolder(
    'dataset',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)




train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])



train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


# ### Define the neural network
# 
# Now, we define the neural network we'll be training.  The *torchvision* package provides a collection of pre-trained models that we can use.
# 
# In a process called *transfer learning*, we can repurpose a pre-trained model (trained on millions of images) for a new task that has possibly much less data available.
# 
# Important features that were learned in the original training of the pre-trained model are re-usable for the new task.  We'll use the ``resnet18`` model.

# In[6]:


model = models.resnet18(pretrained=True)


# The ``resnet18`` model was originally trained for a dataset that had 1000 class labels, but our dataset only has two class labels!  We'll replace
# the final layer with a new, untrained layer that has only two outputs.  

# In[7]:


model.fc = torch.nn.Linear(512, 2)


# Finally, we transfer our model for execution on the GPU

# In[8]:


device = torch.device('cuda')
model = model.to(device)


# ### Train the neural network
# 
# Using the code below we will train the neural network for 30 epochs, saving the best performing model after each epoch.
# 
# > An epoch is a full run through our data.

# In[9]:


NUM_EPOCHS = 30
BEST_MODEL_PATH = 'best_model_resnet18.pth'
best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print('Epoch:\t Validation Accuracy')

for epoch in range(NUM_EPOCHS):
    
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print(f'{epoch+1:>5}:\t {test_accuracy:>.5f}')
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy


# Once that is finished, you should see a file ``best_model_resnet18.pth`` in the Jupyter Lab file browser.  Select ``Right click`` -> ``Download`` to download the model to your workstation

# In[ ]:




