import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
import joblib
from tqdm import tqdm # <-- ADDED for the progress bar

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define local dataset path
dataset_path = "sample_images/YELLOW-RUST-19"  # Adjust to your local path
os.makedirs("models", exist_ok=True)

# Data augmentation and normalization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(root=dataset_path, transform=train_transforms)
print(f"Classes: {dataset.classes}")
print(f"Total images: {len(dataset)}")

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 64
# Note: Set num_workers=0 on Windows to avoid potential issues with multiprocessing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Initialize ViT model
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, len(dataset.classes))  # Use discovered number of classes
model = model.to(device)

# Set up loss function, optimizer, and scheduler
# Adjust weights if your dataset is imbalanced
weights = torch.tensor([1.0, 1.0, 1.2, 1.5, 1.0, 1.1]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=2e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# Training loop
num_epochs = 15
print("\n--- Starting Model Training ---")

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    # Wrap the train_loader with tqdm for a progress bar
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    for inputs, labels in train_progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        # Update the progress bar with the current loss
        train_progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader.dataset)
    scheduler.step()

    # --- Validation Phase ---
    model.eval()
    correct = 0
    total = 0
    # Wrap the test_loader with tqdm for a progress bar
    val_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
    with torch.no_grad():
        for inputs, labels in val_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_accuracy = 100 * correct / total
    
    # Print summary for the epoch
    print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n")

print("--- Finished Model Training ---")

# Save the model's state dictionary as .pkl
model_path = "models/wheat_yellow_rust_model.pkl"
joblib.dump(model.state_dict(), model_path)
print(f"\nModel state dictionary saved at {model_path}")