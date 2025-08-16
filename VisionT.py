#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:29:14 2024

@author: harisushehu
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
from sklearn.metrics import accuracy_score

class BeanLeafDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []

        class_names = os.listdir(image_folder)
        for label, class_name in enumerate(class_names):
            class_folder = os.path.join(image_folder, class_name)
            # Ensure that we only iterate through directories
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    # Only append image files
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Add other extensions if needed
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            # Use the "pixel_values" tensor output from the processor
            image = self.transform(images=image)["pixel_values"][0]  # Extract the tensor
        return torch.tensor(image), label


# Load the model
def load_vit_model(weights_choice, num_classes):
    model = ViTForImageClassification.from_pretrained(
        weights_choice,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    return model

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=5, lr=1e-4):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Convert inputs to tensors if they are lists
            if isinstance(inputs, list):
                inputs = torch.stack(inputs)  # Stack list into a single tensor

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs)  # Convert lists to tensors if needed

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved model with validation accuracy: {best_acc:.4f}')

    return best_acc

# Example usage
def main():
    # Define paths
    train_dir = './output/train/train'
    val_dir = './output/train/test'
    batch_size = 16
    num_epochs = 5
    lr = 1e-4
    
    # Create datasets and loaders
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    transform = processor.preprocess
    
    train_dataset = BeanLeafDataset(train_dir, transform=transform)
    val_dataset = BeanLeafDataset(val_dir, transform=transform)
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load and train model
    model = load_vit_model('google/vit-base-patch16-224', num_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=lr)
    print(f'Best Validation Accuracy: {best_acc:.4f}')

if __name__ == "__main__":
    main()
