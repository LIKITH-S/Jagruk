import os
import torch
import torchvision.transforms as transforms
import pytest
from src.cv_service.models.architecture import DamageClassifier

def test_damage_classifier_forward():
    """Verify forward pass of the modified ResNet50."""
    num_classes = 4
    model = DamageClassifier(num_classes=num_classes, pretrained=False)
    
    # Batch of 2 images, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    # Assert output size is exactly (2, 4)
    assert output.size() == (2, 4)
    assert isinstance(output, torch.Tensor)
