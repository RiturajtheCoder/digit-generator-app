# digit_gan_training.ipynb

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(mnist, batch_size=128, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], 1)
        return self.model(x).view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(794, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat([img.view(img.size(0), -1), labels], 1)
        return self.model(x)

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(device)

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# Training loop
epochs = 10
for epoch in range(epochs):
    for imgs, labels in dataloader:
        batch_size = imgs.size(0)
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        z = torch.randn(batch_size, 100).to(device)
        gen_labels = one_hot(torch.randint(0, 10, (batch_size,)))

        # Train Discriminator
        real_imgs = imgs.to(device)
        real_labels = one_hot(labels)
        optimizer_D.zero_grad()
        d_real = D(real_imgs, real_labels)
        d_fake = D(G(z, gen_labels).detach(), gen_labels)
        loss_D = loss_fn(d_real, real) + loss_fn(d_fake, fake)
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        gen_imgs = G(z, gen_labels)
        d_fake = D(gen_imgs, gen_labels)
        loss_G = loss_fn(d_fake, real)
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

# Save the model
torch.save(G.state_dict(), "generator.pth")
