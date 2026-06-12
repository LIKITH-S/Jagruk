import torch.nn as nn
from torchvision import models

class DamageClassifier(nn.Module):
    """
    Damage classification model based on pretrained ResNet50.
    Modified for 4 damage severity classes: no-damage, minor, major, destroyed.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(DamageClassifier, self).__init__()
        
        # Load the pretrained ResNet50
        if pretrained:
            self.resnet = models.resnet50(weights='DEFAULT')
        else:
            self.resnet = models.resnet50()
            
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
