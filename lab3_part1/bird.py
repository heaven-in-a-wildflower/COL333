import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
from collections import Counter

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

# Paths and Training Parameters
if __name__ == "__main__":
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3]

class birdClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(birdClassifier, self).__init__()
        
        # Feature extraction with improved blocks and attention mechanisms
        self.features = nn.Sequential(
            # Initial block with increased channels
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),  # Replace ReLU with GELU for better gradient flow
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),

            # Block 1 with residual connection and channel attention
            ResidualBlock(64, 128, use_attention=True),
            nn.MaxPool2d(2, 2),

            # Double Residual block with increased channels
            ResidualBlock(128, 256, use_attention=True),
            ResidualBlock(256, 256, use_attention=True),
            nn.MaxPool2d(2, 2),

            # Enhanced Inception block with squeeze-and-excitation
            InceptionBlock(256, [128, 256, 64, 64]),
            SEBlock(512),  # Increased channels after inception
            nn.MaxPool2d(2, 2),

            # Spatial attention block
            SpatialAttention(),

            # Deep feature extraction with dilated convolutions
            nn.Conv2d(512, 1024, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            DropBlock2d(block_size=7, drop_prob=0.1),  # Reduced drop probability
            nn.MaxPool2d(2, 2),
        )
        
        # Multiple pooling branches for better feature aggregation
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Enhanced classifier with deeper architecture
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Reduced dropout for better feature preservation
            nn.Linear(2048, 1024),  # Double input features due to dual pooling
            nn.GELU(),
            nn.BatchNorm1d(1024),  # Added BatchNorm for stability
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Concatenate average and max pooling features
        avg_pooled = self.avg_pool(x).view(x.size(0), -1)
        max_pooled = self.max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        x = self.classifier(x)
        return x

class DropBlock2d(nn.Module):
    def __init__(self, block_size, drop_prob):
        super(DropBlock2d, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        block_mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        return x * (1 - block_mask)
        
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        scale = x.mean(dim=[2, 3], keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.use_attention = use_attention
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        if use_attention:
            self.se = SEBlock(out_channels)
        
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.use_attention:
            x = self.se(x)
        x += residual
        return self.gelu(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size=1)
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_list[1], kernel_size=1),
            nn.Conv2d(out_channels_list[1], out_channels_list[1], kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_list[2], kernel_size=1),
            nn.Conv2d(out_channels_list[2], out_channels_list[2], kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels_list[3], kernel_size=1)
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, dim=1)
# Training utilities
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))

# Custom Dataset class to handle file path based labels
class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.class_counts = Counter()
        
        # Walk through directory structure
        for class_idx in range(10):  # Assuming 10 classes numbered 0-9
            class_dir = os.path.join(root_dir, str(class_idx))
            if os.path.exists(class_dir):
                self.class_to_idx[str(class_idx)] = class_idx
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
                        self.class_counts[class_idx] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')  # Explicitly convert to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

mean = [0.4838482737541199, 0.49302709102630615, 0.41043317317962646]
std = [0.23783758282661438, 0.23537121713161469, 0.25884684920310974]
# Step 2: Create the final transform with the calculated statistics
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=100):
    train_losses, train_accuracies = [], []
    previous_loss = float('inf')  # Initialize previous loss to a large value
    lr_decrease_factor = 0.5  # Factor by which to decrease the learning rate
    best_train_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i,(images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), "bird.pth")
        
        # Check for increase in loss and adjust learning rate if necessary
        if train_loss > previous_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decrease_factor  # Decrease learning rate

        previous_loss = train_loss  # Update previous loss
        
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}]:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_accuracy:.4f}")
        
# Main Execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = birdClassifier(num_classes=10).to(device)

# Number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params}")

criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def test_model(model, test_loader, device, output_csv='bird.csv'):
    model.eval()
    results = []
    
    print("Testing...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                print(f'Processed {i+1} test samples')
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])
            
if trainStatus == "train":
    full_train_dataset = BirdDataset(root_dir=dataPath, transform=train_transform)
    full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)
    train_model(model, full_train_loader, criterion, optimizer, scheduler, num_epochs=15)
    
else:
    model.load_state_dict(torch.load(modelPath))
    test_dataset = BirdDataset(root_dir=dataPath,transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_model(model, test_loader, device)
