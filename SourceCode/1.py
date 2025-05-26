import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split training set into training (40,000) and validation (10,000)
train_size = int(0.8 * len(trainset))  # 80% for training
val_size = len(trainset) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

# Create DataLoaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Training function with validation
def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_train_loss / len(trainloader)
        epoch_train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(valloader)
        epoch_val_acc = 100 * correct_val / total_val
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.3f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {epoch_val_loss:.3f}, Val Acc: {epoch_val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

# Plot learning curves
def plot_learning_curves(mlp_losses, mlp_val_losses, mlp_accs, mlp_val_accs, cnn_losses, cnn_val_losses, cnn_accs, cnn_val_accs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp_losses, label='MLP Train Loss')
    plt.plot(mlp_val_losses, label='MLP Val Loss', linestyle='--')
    plt.plot(cnn_losses, label='CNN Train Loss')
    plt.plot(cnn_val_losses, label='CNN Val Loss', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_accs, label='MLP Train Accuracy')
    plt.plot(mlp_val_accs, label='MLP Val Accuracy', linestyle='--')
    plt.plot(cnn_accs, label='CNN Train Accuracy')
    plt.plot(cnn_val_accs, label='CNN Val Accuracy', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(labels, preds, title, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

# Main execution
def main():
    # Initialize models
    mlp = MLP().to(device)
    cnn = CNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    
    # Train models
    print("Training MLP...")
    mlp_losses, mlp_val_losses, mlp_accs, mlp_val_accs = train_model(mlp, trainloader, valloader, criterion, mlp_optimizer)
    
    print("\nTraining CNN...")
    cnn_losses, cnn_val_losses, cnn_accs, cnn_val_accs = train_model(cnn, trainloader, valloader, criterion, cnn_optimizer)
    
    # Evaluate models
    mlp_accuracy, mlp_preds, mlp_labels = evaluate_model(mlp, testloader)
    cnn_accuracy, cnn_preds, cnn_labels = evaluate_model(cnn, testloader)
    
    print(f'\nMLP Test Accuracy: {mlp_accuracy:.2f}%')
    print(f'CNN Test Accuracy: {cnn_accuracy:.2f}%')
    
    # Plot results
    plot_learning_curves(mlp_losses, mlp_val_losses, mlp_accs, mlp_val_accs, cnn_losses, cnn_val_losses, cnn_accs, cnn_val_accs)
    plot_confusion_matrix(mlp_labels, mlp_preds, 'MLP Confusion Matrix', 'mlp_confusion_matrix.png')
    plot_confusion_matrix(cnn_labels, cnn_preds, 'CNN Confusion Matrix', 'cnn_confusion_matrix.png')

if __name__ == '__main__':
    main()
