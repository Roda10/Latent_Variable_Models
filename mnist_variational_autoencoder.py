import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting the hyperparameters' values
input_size = 784
latent_dim = 32
batch_size = 128

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load MNIST
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Defining the VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_dim=latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_size),
            nn.Sigmoid()  # Pixel values between 0-1
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss function(ELBO)
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, input_size), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss = vae_loss(recon_batch, data, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {train_loss/len(train_loader.dataset):.4f}')

# Reconstruct Samples

def show_reconstructions(model, n=10):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(train_loader))
        data = data.to(device)
        recon, _, _ = model(data)
        
        fig, axes = plt.subplots(2, n, figsize=(20, 4))
        for i in range(n):
            axes[0,i].imshow(data[i].cpu().view(28,28), cmap='gray')
            axes[0,i].axis('off')
            axes[1,i].imshow(recon[i].cpu().view(28,28), cmap='gray')
            axes[1,i].axis('off')
        plt.show()

show_reconstructions(model)

# Generate new digits
def generate_samples(model, latent_dim=20):
    model.eval()
    with torch.no_grad():
        # Sample from standard normal
        z = torch.randn(64, latent_dim).to(device)
        samples = model.decode(z)
        
        fig = plt.figure(figsize=(8,8))
        for i in range(64):
            plt.subplot(8,8,i+1)
            plt.imshow(samples[i].cpu().view(28,28), cmap='gray')
            plt.axis('off')
        plt.show()

generate_samples(model)

# Project test set into latent space
def plot_latent_space(model):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            latents.append(mu.cpu())
            labels.append(label)
    
    latents = torch.cat(latents).numpy()
    labels = torch.cat(labels).numpy()
    
    plt.figure(figsize=(10,8))
    plt.scatter(latents[:,0], latents[:,1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.show()

plot_latent_space(model)