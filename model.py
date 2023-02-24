import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms, models
import torch.nn.functional as F



# HyperParameters
batch_size = 1
input_size = (224, 224)
num_classes = 2


# Define a transform to preprocess the data
transform = transforms.Compose([    
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
# Define the model
def get_model():
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft
    
def initialize(path):
    global model
    model = get_model().to(device)

    # Load the trained model weights
    ## TODO:  !!IMPORTANT, remove the map_location='cpu' when running on GPU!!
    model.load_state_dict(torch.load(path, map_location='cpu'))



def evaluate(images):
    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        # Move the data to the correct device


        inputs = transform(images).to(device)

        # Forward pass
        outputs = model(inputs.unsqueeze(0))

        # Apply softmax to the outputs
        probs = F.softmax(outputs, dim=1)

        # Get the predictions and the confidence
        _, predicted = torch.max(outputs.data, 1)
        confidence, _ = torch.max(probs.data, 1)

        return predicted, confidence

        # Update the correct and total counts, and add the confidence to the correct/incorrect confidence
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()
        #correct_confidence += confidence[predicted == labels].sum().item()
        #incorrect_confidence += confidence[predicted != labels].sum().item()
