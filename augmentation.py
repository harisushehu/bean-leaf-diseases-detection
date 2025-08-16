#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:06:24 2024

@author: harisushehu
"""

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Load an example image (replace 'example.jpg' with the path to your image)
image_path = './data/train/train/1/angular_leaf_spot_train.21.jpg'
original_image = Image.open(image_path).convert('RGB')

# Define the augmentations
transformations = {
    "Original": original_image,
    "Shift Left (0.1)": transforms.functional.affine(original_image, angle=0, translate=(-10, 0), scale=1, shear=0),
    "Rotate (15)": transforms.functional.rotate(original_image, angle=15),
    "Height Shift (0.1)": transforms.functional.affine(original_image, angle=0, translate=(0, -10), scale=1, shear=0),
    "Horizontal Flip": transforms.functional.hflip(original_image),
    "Color Jitter (0.2)": transforms.ColorJitter(brightness=0.2)(original_image)
}

# Function to display images
def show_images(transformations):
    plt.figure(figsize=(12, 8))
    for i, (name, img) in enumerate(transformations.items()):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Show the images
show_images(transformations)
