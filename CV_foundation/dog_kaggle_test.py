import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
# from pytorch_lightning import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



# 读取csv文件
comp_df = pd.read_csv('../data/dog/labels.csv')
test_df = pd.read_csv('../data/dog/sample_submission.csv')

# 训练集和测试集有多少张图像?
print('Training set: {}, Test set: {}'.format(comp_df.shape[0],test_df.shape[0]))

"""创建自定义数据集类和转换器"""
class img_dataset(Dataset):
    def __init__(self, dataframe, transform=None, test=False):
        self.dataframe = dataframe
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        x = Image.open(self.dataframe.iloc[index, 0])
        if self.transform:
            x = self.transform(x)
        if self.test:
            return x
        else:
            y = self.dataframe.iloc[index, 1]
            return x, y

    def __len__(self):
        return self.dataframe.shape[0]

# Creat transfomers
train_transformer = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomRotation(15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transformer = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# Print the result of 1 epoch
def print_epoch_result(train_loss, train_acc, val_loss, val_acc):
    print('loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}'.format(train_loss,
                                                                                train_acc,
                                                                                val_loss,
                                                                                val_acc))


# Main Training function
def train_model(model, cost_function, optimizer, train_loader, val_loader, num_epochs=5, device='cuda'):
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    def calculate_accuracy(y_hat, y):
        _, preds = torch.max(y_hat, 1)
        return torch.sum(preds == y).item() / len(y)

    for epoch in range(num_epochs):
        """
        On epoch start
        """
        print('-' * 15)
        print('Start training {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 15)

        # Training
        train_sub_losses = []
        correct_train = 0
        total_train = 0
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = cost_function(y_hat, y)
            loss.backward()
            optimizer.step()
            train_sub_losses.append(loss.item())
            correct_train += (y_hat.argmax(1) == y).type(torch.float).sum().item()
            total_train += y.size(0)

        # Validation
        val_sub_losses = []
        correct_val = 0
        total_val = 0
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = cost_function(y_hat, y)
            val_sub_losses.append(loss.item())
            correct_val += (y_hat.argmax(1) == y).type(torch.float).sum().item()
            total_val += y.size(0)

        """
        On epoch end
        """
        train_losses.append(np.mean(train_sub_losses))
        val_losses.append(np.mean(val_sub_losses))

        train_epoch_acc = correct_train / total_train
        val_epoch_acc = correct_val / total_val
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {np.mean(train_sub_losses):.4f} - "
              f"Train Acc: {train_epoch_acc:.4f} - "
              f"Val Loss: {np.mean(val_sub_losses):.4f} - "
              f"Val Acc: {val_epoch_acc:.4f}")

    print('Finish Training.')
    return train_losses, train_acc, val_losses, val_acc




if __name__ == '__main__':
    # Setting up gpu
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    # Parameters for dataset
    training_samples = comp_df.shape[
        0]  # Use small number first to test whether the model is doing well, then change back to full dataset
    test_size = 0.05
    batch_size = 64

    # Reduce the number of samples
    sample_df = comp_df.sample(training_samples)

    # Split the comp_df into training set and validation set
    x_train, x_val, _, _ = train_test_split(sample_df, sample_df, test_size=test_size)

    # Create dataloaders form datasets
    train_set = img_dataset(x_train, transform=train_transformer)
    val_set = img_dataset(x_val, transform=val_transformer)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # How many images in training set and val set?
    print('Training set: {}, Validation set: {}'.format(x_train.shape[0], x_val.shape[0]))


    # Use resnet-50 as a base model
    class net(torch.nn.Module):
        def __init__(self, base_model, base_out_features, num_classes):
            super(net, self).__init__()
            self.base_model = base_model
            self.linear1 = torch.nn.Linear(base_out_features, 512)
            self.output = torch.nn.Linear(512, num_classes)

        def forward(self, x):
            x = F.relu(self.base_model(x))
            x = F.relu(self.linear1(x))
            x = self.output(x)
            return x


    res = torchvision.models.resnet34(pretrained=True)
    for param in res.parameters():
        param.requires_grad = False

    model_final = net(base_model=res, base_out_features=res.fc.out_features, num_classes=120)
    model_final = model_final.to(device)

    # Cost function and optimzier
    cost_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for param in model_final.parameters() if param.requires_grad], lr=0.0003)

    # Learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)

    # Epoch
    EPOCHS = 30

    # Start Training
    train_losses, train_acc, val_losses, val_acc = train_model(model=model_final,
                                                               cost_function=cost_function,
                                                               optimizer=optimizer,
                                                               train_loader=train_loader,
                                                               val_loader=val_loader,
                                                               num_epochs=EPOCHS,
                                                               device = device)
