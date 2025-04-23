import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # 1st Convolution Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        # Max Pool (after 1st conv): kernel=4, stride=4
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        # 2nd Convolution Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        # Max Pool (after 2nd conv): kernel=2, stride=2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd Convolution Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               stride=1, padding=1)
        # Max Pool (after 3rd conv): kernel=2, stride=2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classification head
        # We'll flatten the output of pool3, then use a linear layer
        self.fc = nn.Linear(in_features=128 *  (224 // 4 // 2 // 2) * (224 // 4 // 2 // 2),
                            out_features=num_classes)
        # - Input image is 224x224
        # - After pool1 with kernel=4,stride=4 => 224/4=56
        # - After pool2 with kernel=2,stride=2 => 56/2=28
        # - After pool3 with kernel=2,stride=2 => 28/2=14
        # So final feature map is (128, 14, 14). Flatten => 128*14*14

    def forward(self, x):
        # 1st conv + ReLU + max pool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 2nd conv + ReLU + max pool
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 3rd conv + ReLU + max pool
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.fc(x)
        return x


# model_class.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Load a pretrained ResNet-18 from torchvision.
        resnet = models.resnet18(pretrained=True)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        
        # Instead of storing the model as a single attribute,
        # assign each layer to self so that the state dict keys match.
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
