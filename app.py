# app.py

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generator model must match training architecture
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

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

# Load model
device = torch.device("cpu")
G = Generator()
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

st.title("🖌️ Handwritten Digit Generator")
digit = st.selectbox("Select a digit (0-9)", list(range(10)))

if st.button("Generate Images"):
    z = torch.randn(5, 100)
    labels = one_hot(torch.tensor([digit]*5), 10)
    with torch.no_grad():
        images = G(z, labels).squeeze().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(images[i], cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
