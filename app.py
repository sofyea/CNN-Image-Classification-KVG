import torch
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import ssl
from model import SimpleCNN

# Pintas SSL untuk muat turun gambar ujian
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Senarai Kelas CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Muat Naik Model yang Dah Dilatih
model = SimpleCNN()
model.load_state_dict(torch.load('cifar10_cnn.pth'))
model.eval()

print("Model CNN berjaya dimuat naik dari 'cifar10_cnn.pth'!")

# 3. Preprocessing Imej Ujian
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 4. Muat Turun Gambar Contoh (Kucing)
img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
urllib.request.urlretrieve(img_url, "test_image.jpg")

# 5. Lakukan Ramalan (Prediction)
image = Image.open("test_image.jpg")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

print(f"\n==================================================")
print(f"Hasil Ramalan Model CNN: {classes[predicted[0]].upper()}")
print(f"==================================================")
