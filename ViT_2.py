#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:21:58 2024

@author: harisushehu
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification
import os
from PIL import Image
import requests



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load the Vision Transformer (ViT) model
def load_vit_model(weights_choice, num_classes):
    """
    Loads a Vision Transformer model with the specified weights.
    """
    if weights_choice == 'vit_base_patch16_224':
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    elif weights_choice == 'vit_large_patch16_224':
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    elif weights_choice == 'vit_small_patch16_224':
        # This example assumes you have a valid model name for the `vit_small_patch16_224` model
        model_name = 'vit-large-patch16-224'  # Change this to the correct identifier if needed
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

    else:
        raise ValueError(f'Unknown weights choice: {weights_choice}')
    return model


# Function to train the model
def train_model(model, train_loader, val_loader, dataset_sizes, device, save_path='best_model.pth', num_epochs=10,
                lr=1e-4, verbose=1):
    """
    Trains the model and evaluates its performance.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes['train']
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved model with validation accuracy: {val_acc:.4f}')

    return best_acc


# Data loading and preprocessing
def prepare_data(data_dir, batch_size=16, validation_split=0.1, img_size=(224, 224)):
    """
    Prepares training and validation data loaders.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(dataset.classes)

    # Split dataset into train and validation sets
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=validation_split,
                                          stratify=dataset.targets)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    dataset_sizes = {'train': len(train_idx), 'val': len(val_idx)}
    return train_loader, val_loader, dataset_sizes, num_classes


# Main function
if __name__ == '__main__':
    # Define data directories
    base_dir = './output/train/train'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    categories = ["angular leaf spot", "bean rus", "healthy"]

    # Prepare data
    train_loader, val_loader, dataset_sizes, num_classes = prepare_data(train_dir)

    # Load model (ViT)
    model = load_vit_model(weights_choice='vit_base_patch16_224', num_classes=num_classes) # vit_small_patch16_224, vit_base_patch16_224
    model.to(device)

    # Train the model
    best_acc = train_model(model, train_loader, val_loader, dataset_sizes, device, num_epochs=10, lr=1e-4)
    print(f'Best Validation Accuracy: {best_acc:.4f}')
