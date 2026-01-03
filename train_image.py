import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.image_utils import get_resnet_model
import os

# --- CONFIGURATION ---
DATA_DIR = "data/image/raw"
MODEL_SAVE_PATH = "models/image_model.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def train_model():
    # 1. Data Augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # 3. Initialize Model and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    
    # Create model and optimizer first
    model = get_resnet_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    start_epoch = 0
    
    # --- RESUME LOGIC ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 Checkpoint found! Loading to resume...")
        # Load the full dictionary
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        
        try:
            # Load states into existing objects
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"✅ Resuming from Epoch {start_epoch + 1}")
        except KeyError:
            # Fallback if your previous .pth file was only model weights
            print("⚠️ Old format detected. Loading weights only and starting from Epoch 1.")
            model.load_state_dict(checkpoint) # Original version just saved state_dict

    print(f"🚀 Training on: {device}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Validation Step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"⭐ Epoch [{epoch+1}/{EPOCHS}] Completed - Val Accuracy: {accuracy:.2f}%")
        
        # --- SAVE CHECKPOINT DICTIONARY ---
        # Save all states needed to fully resume
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }
        torch.save(checkpoint_data, MODEL_SAVE_PATH)
        print(f"💾 Checkpoint saved for Epoch {epoch + 1}")

    print(f"✅ Training Session Finished.")

if __name__ == "__main__":
    train_model()