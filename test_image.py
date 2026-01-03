import torch
from PIL import Image
from torchvision import transforms
from src.image_utils import get_resnet_model

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet_model(checkpoint_path="models/image_model.pth").to(device)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        prob = output.item()
    label = "FAKE" if prob > 0.5 else "REAL"
    print(f"Prediction: {label} (Confidence: {prob:.4f})")

# Replace with a path to a real image you have
predict("dr.jpg")