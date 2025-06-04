import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

#  資料預處理
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset, _ = torch.utils.data.random_split(trainset, [5000, len(trainset)-5000])  # 只取前 5000 筆資料
trainloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

#  建立簡單 CNN 模型
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.model(x)

net = MiniCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

#  模型訓練
for epoch in range(3):
    total_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
