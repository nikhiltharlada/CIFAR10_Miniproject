import torch
import os
from PIL import Image
from torchvision import transforms
from src.model import CIFARCNN
IMAGE_NAME = "aeroplane1.jpg" #change the image name 
MODEL_PATH = "cifar10_model.pth"

IMAGE_PATH = os.path.join("testing_images", IMAGE_NAME)
classes = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFARCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

if os.path.exists(IMAGE_PATH):
    img = Image.open(IMAGE_PATH).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_index = torch.max(output, 1)
        result = classes[predicted_index.item()]

    print(f"Selected Image: {IMAGE_NAME}")
    print(f"Prediction: {result.upper()}")
else:
    print(f"Error: File '{IMAGE_NAME}' not found in folder 'testing_images'")