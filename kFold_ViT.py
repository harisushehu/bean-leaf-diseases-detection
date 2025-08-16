#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:55:36 2025

@author: harisushehu
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from transformers import ViTForImageClassification
import os
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time
import json

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
        model_name = 'vit-large-patch16-224'
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

    else:
        raise ValueError(f'Unknown weights choice: {weights_choice}')
    return model

def plot_metrics(train_losses, val_accuracies, fold, save_dir='figures'):
    """
    Plot training loss and validation accuracy and save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title(f'Training Loss (Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title(f'Validation Accuracy (Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fold_{fold + 1}_metrics.png'))
    plt.close()

def plot_confusion_matrix(cm, classes, fold, save_dir='confusion_matrices'):
    """
    Plot confusion matrix and save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Fold {fold + 1})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fold_{fold + 1}_confusion_matrix.png'))
    plt.close()

# Function to train the model
def train_model(model, train_loader, val_loader, dataset_sizes, device, num_epochs=10,
                lr=1e-4, verbose=1, fold=0):
    """
    Trains the model and evaluates its performance.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes['train']
        train_losses.append(epoch_loss)
        
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
        val_accuracies.append(val_acc)
        
        if verbose:
            epoch_time = time() - epoch_start_time
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s')

        if val_acc > best_acc:
            best_acc = val_acc

    # Plot metrics for this fold
    plot_metrics(train_losses, val_accuracies, fold)
    
    # Calculate additional metrics
    val_precision = precision_score(val_labels, val_preds, average='weighted')
    val_recall = recall_score(val_labels, val_preds, average='weighted')
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    
    return {
        'accuracy': best_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'confusion_matrix': cm,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

# Data loading and preprocessing
def prepare_data(data_dir, batch_size=16, img_size=(224, 224)):
    """
    Prepares dataset with transformations.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(dataset.classes)
    
    return dataset, num_classes

# Main function
if __name__ == '__main__':
    # Define data directories
    base_dir = './output/train/train'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    categories = ["angular leaf spot", "bean rus", "healthy"]
    
    # Prepare data
    dataset, num_classes = prepare_data(train_dir)
    
    # Initialize k-fold cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Results storage
    results = []
    class_names = dataset.classes
    
    # Load model (ViT)
    model = load_vit_model(weights_choice='vit_base_patch16_224', num_classes=num_classes)
    model.to(device)
    
    # K-fold Cross Validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)
        
        # Define data loaders for training and validation
        train_loader = DataLoader(train_subsampler, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=16, shuffle=False)
        
        dataset_sizes = {'train': len(train_ids), 'val': len(val_ids)}
        
        # Train the model for this fold
        fold_results = train_model(model, train_loader, val_loader, dataset_sizes, device, 
                                 num_epochs=10, lr=1e-4, fold=fold)
        
        # Save confusion matrix plot
        plot_confusion_matrix(fold_results['confusion_matrix'], class_names, fold)
        
        # Store results
        results.append({
            'fold': fold + 1,
            'accuracy': fold_results['accuracy'],
            'precision': fold_results['precision'],
            'recall': fold_results['recall'],
            'f1': fold_results['f1'],
            'confusion_matrix': fold_results['confusion_matrix'].tolist()
        })
        
        # Print fold results
        print(f'Fold {fold + 1} Results:')
        print(f'Accuracy: {fold_results["accuracy"]:.4f}')
        print(f'Precision: {fold_results["precision"]:.4f}')
        print(f'Recall: {fold_results["recall"]:.4f}')
        print(f'F1 Score: {fold_results["f1"]:.4f}')
        print('--------------------------------')
    
    # Calculate average metrics
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    
    # Add average results
    results.append({
        'fold': 'Average',
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'confusion_matrix': None
    })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('cross_validation_results.csv', index=False)
    
    # Save detailed results to JSON
    with open('detailed_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final results
    print('\nFINAL RESULTS:')
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')