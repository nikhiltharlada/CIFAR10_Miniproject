import torch
from torch import nn
from src.model import CIFARCNN
from src.data_setup import get_loaders
from src.engine import train_step, test_step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs=18
train_loader, test_loader = get_loaders(batch_size=64)
model = CIFARCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(epochs):
    loss = train_step(model, train_loader, loss_fn, optimizer, device)
    accuracy = test_step(model, test_loader, device)
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")
torch.save(model.state_dict(), "cifar10_model.pth")
print("Project Complete! Model Saved.")
