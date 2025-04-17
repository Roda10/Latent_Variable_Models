import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(40)

# Hyperparameters
input_size = 784  # 28x28 images flattened
hidden_size = 128  # Hidden layer size
latent_size = 32  # Latent space size
num_epochs = 20
batch_size = 128
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.ToTensor()  # Convert images to tensors and normalize to [0,1]
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: input_size -> hidden_size -> latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        # Decoder: latent_size -> hidden_size -> input_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Output in [0,1] to match input
        )
    
    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten input
        z = self.encoder(x)  # Encode to latent space
        x_hat = self.decoder(z)  # Decode to reconstruct
        return x_hat

# Initialize model, loss function, and optimizer
model = Autoencoder()
criterion = nn.BCELoss()  # Binary cross-entropy for pixel values in [0,1]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        img, _ = data  # Ignore labels
        img = img.to(torch.device('cpu'))  # Ensure data is on CPU (or GPU if available)
        
        # Forward pass
        output = model(img)
        loss = criterion(output, img.view(-1, input_size))
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
    
    # Print average loss per epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Testing and visualization
model.eval()
with torch.no_grad():
    # Get some test images
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    
    # Reconstruct images
    reconstructed = model(images)
    
    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        # Original images
        axes[0, i].imshow(images[i].squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].view(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.savefig('mnist_ae_reconstructions.png')
    plt.close()

print("Training complete! Reconstructions saved as 'mnist_ae_reconstructions.png'.")