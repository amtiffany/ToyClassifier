import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Define model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 2. Load the model
model = SimpleCNN()
model.load_state_dict(torch.load("binary_cnn_model.pth"))
model.eval()

# 3. Define image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 4. Folder to classify
image_folder = "input_images"  # CHANGE THIS
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 5. Predict on each image
print(f"🔍 Predicting on {len(image_files)} image(s) in: {image_folder}\n")

for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 64, 64]

    with torch.no_grad():
        output = model(input_tensor)
        prediction = int(output.item() > 0.5)

    print(f"{filename}: Predicted Class {prediction}")
