import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN

# 1. Tetapkan device (GPU jika ada, jika tidak guna CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Preprocessing Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Memuatkan dataset untuk latihan...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 3. Panggil Model, Loss Function, dan Optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Proses Latihan (Training Loop)
epochs = 2
print("Mula melatih model CNN...")

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # Cetak status setiap 200 batch
            print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {running_loss / 200:.3f}')
            running_loss = 0.0

# 5. Simpan Model yang Dah Dilatih
torch.save(model.state_dict(), 'cifar10_cnn.pth')
print("\nBERJAYA! Model telah dilatih dan disimpan sebagai 'cifar10_cnn.pth'.")
