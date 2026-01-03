import torch
import torch.nn as nn
import os
from torchvision import models

def get_resnet_model(checkpoint_path=None):
    # Load ResNet18 with default pre-trained weights
    model = models.resnet18(weights='DEFAULT')
    
    # Modify the final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    # --- UPDATED LOADING LOGIC ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Check if it's a dictionary checkpoint or just weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Extract only the weights from the dictionary
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Load as plain weights (backward compatibility)
            model.load_state_dict(checkpoint)
    
    return model