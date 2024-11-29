import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os
from tqdm import tqdm

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Custom Dataset
class MNISTDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] / 255.0  # Normalization

# ==============================================VAE===================================================
# Define VAE Model
class VAE(nn.Module):
    def __init__(self,h_dim=324,latent_size=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, h_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(h_dim, latent_size)
        self.fc_logvar = nn.Linear(h_dim, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sampling(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        h = self.encoder(x.view(-1, 28*28))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def sampling(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
        
# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Training Function
def train_vae(train_loader, model, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")

# def train_vae(train_loader, model, optimizer, epochs):
#     # Initialize the cosine annealing scheduler
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)

#     model.train()
#     for epoch in range(epochs):
#         train_loss = 0
#         for data in train_loader:
#             data = data.to(device)
#             optimizer.zero_grad()
#             recon_batch, mu, logvar = model(data)
#             loss = loss_function(recon_batch, data, mu, logvar)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
        
#         # Step the scheduler at the end of each epoch
#         scheduler.step()
        
#         # Print the average loss for the epoch and the current learning rate
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}, LR: {current_lr:.6f}")
        
def show_reconstruction(model, val_loader, n=15, output_image_path="reconstruction.png"):
    model.eval()
    data = next(iter(val_loader))
    
    data = data.to(device)
    with torch.no_grad():
        recon_data, _, _ = model(data)
    
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        # Original images
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(recon_data[i].cpu().view(28, 28).detach().numpy(), cmap='gray')
        axes[1, i].axis('off')
    
    # Save the figure
    plt.savefig(output_image_path)
    plt.show()
    print(f"Reconstruction images saved to {output_image_path}")

def eval_reconstruction(original_images, reconstructed_images):
    mse = np.mean((original_images - reconstructed_images) ** 2)
    
    ssim_vals = []
    for i in range(len(original_images)):
        ssim_vals.append(ssim(original_images[i], reconstructed_images[i], data_range=reconstructed_images[i].max() - reconstructed_images[i].min()))

    mean_ssim = np.mean(ssim_vals)
    return mean_ssim,1 - mse

def extract_latent_vectors(model, loader):
    model.eval()
    latents = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu().numpy())
    return np.concatenate(latents)

def plot_2d_manifold(model, latent_dim=2, n=20, digit_size=28, output_path="generated_manifold.png"):
    """Generate and visualize images from sampled latent vectors"""
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Create grid of points
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    model.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                # Sample point from latent space
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                
                # Generate image from sampled point
                x_decoded = model.decode(z_sample)
                digit = x_decoded.cpu().view(digit_size, digit_size).numpy()
                
                # Place the generated image in the grid
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')
    plt.axis('off')
    plt.savefig(output_path)
    plt.show()
    print(f"Generated manifold saved to {output_path}")

def visualize_latent_space(model, loader):
    model.eval()
    latents = extract_latent_vectors(model, loader)
    plt.figure(figsize=(8, 8))
    plt.title("VAE Latent Space Distribution")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.scatter(latents[:, 0], latents[:, 1], alpha=0.5, s=2)
    plt.grid(True)
    plt.colorbar()
    plt.savefig("latent_space.png")
    plt.show()
    print(f"Latent space visualization saved to latent_space.png")
    
    return latents

# ==============================================GMM===================================================

class GaussMixModel:
    def __init__(self, n_classes=3, max_iter=100, tol=1e-4):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.covs = None
        self.weights = None
        self.class_mapping = {1: 0, 4: 1, 8: 2}  # Map original classes to indices
    
    def initialize_from_validation(self, init_latents, init_labels):
        self.means = np.zeros((self.n_classes, init_latents.shape[1]))
        self.covs = np.zeros((self.n_classes, init_latents.shape[1], init_latents.shape[1]))        
        for orig_class, i in self.class_mapping.items():
            class_latents = init_latents[init_labels == orig_class]
            self.means[i] = np.mean(class_latents, axis=0)
            self.covs[i] = np.dot((class_latents - self.means[i]).T, class_latents - self.means[i]) / len(class_latents) + np.eye(init_latents.shape[1]) * 1e-6        
        # Initialize weights as uniform
        self.weights = np.ones(self.n_classes) / self.n_classes
    
    def _gaussian(self, X, mean, cov):
        return np.exp(-0.5 * (X.shape[1] * np.log(2 * np.pi) + 
                            np.linalg.slogdet(cov + np.eye(X.shape[1]) * 1e-6)[1] + 
                            np.sum((X - mean).dot(np.linalg.inv(cov + np.eye(X.shape[1]) * 1e-6)) * (X - mean), axis=1)))

    
    def _log_likelihood(self, X):
        return np.sum(np.log(np.sum([self.weights[k] * self._gaussian(X, self.means[k], self.covs[k]) for k in range(self.n_classes)], axis=0) + 1e-10))

    def _e_step(self, X):
        return (resp := np.array([self.weights[k] * self._gaussian(X, self.means[k], self.covs[k]) for k in range(self.n_classes)]).T) / resp.sum(axis=1, keepdims=True)

    def _m_step(self, X, resp):
        self.weights, self.means, self.covs = (N := resp.sum(axis=0)) / X.shape[0], resp.T.dot(X) / N[:, np.newaxis], np.array([(resp[:, k:k+1] * (diff := X - m)).T.dot(diff) / N[k] + np.eye(X.shape[1]) * 1e-6 for k, m in enumerate(self.means)])

    def fit(self, X, val_latents, val_labels):
        self.initialize_from_validation(val_latents, val_labels)
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            if abs((ll := self._log_likelihood(X)) - prev_ll) < self.tol: break
            prev_ll = ll
            self._m_step(X, self._e_step(X))

    def predict(self, X):
        resp = self._e_step(X)
        cluster_indices = np.argmax(resp, axis=1)
        inverse_mapping = dict(map(reversed, self.class_mapping.items()))
        return np.array([inverse_mapping.get(idx, -1) for idx in cluster_indices])

def plot_gmm(latents, gmm_means, gmm_covs, resps):
    plt.figure(figsize=(8, 8))
    plt.title("GMM Clustering in Latent Space")
    plt.scatter(latents[:, 0], latents[:, 1], c=np.argmax(resps, axis=1), cmap='viridis', alpha=0.9, s=15, marker='o')
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    for i, (mean, cov) in enumerate(zip(gmm_means, gmm_covs)):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='blue', lw=2, facecolor='none')
        plt.gca().add_patch(ellipse)
        plt.plot(mean[0], mean[1], 'ro')
    plt.colorbar(label="Cluster Responsibility")
    plt.grid(True)
    plt.savefig("gmm.png")
    plt.show()

def prediction_vae_and_gmm(model, gmm, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            # Get latent vectors
            data = data.to(device)
            mu, _ = model.encode(data)
            latents = mu.cpu().numpy()
            batch_preds = gmm.predict(latents)
            predictions.extend(batch_preds)
    return np.array(predictions)

# ==============================================EXE===================================================
if __name__ == "__main__":
    args = sys.argv
    print(args[2])
    if args[3] == "train":
        # Load data arrays from the .npz files
        train_data = np.load(args[1])['data']
        val_data = np.load(args[2])['data']
        val_labels = np.load(args[2])['labels']
        
        train_dataset = MNISTDataset(train_data)
        val_dataset = MNISTDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        model = VAE().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3,weight_decay=5e-4)
        
        # Train VAE
        train_vae(train_loader, model, optimizer, epochs=500)
        val_latents = extract_latent_vectors(model, val_loader)

        print("Visualizing raw latent space...")
        train_latents = visualize_latent_space(model, train_loader)
        
        gmm = GaussMixModel()
        gmm.fit(train_latents, val_latents, val_labels)

        # Plot GMM clustering results
        resps = gmm._e_step(train_latents)
        plot_gmm(train_latents, gmm.means, gmm.covs, resps)

        # Save model
        torch.save(model.state_dict(), args[4])
        with open(args[5], 'wb') as f:
            pickle.dump(gmm, f)

        plot_2d_manifold(model, latent_dim=2, n=20)

        # ==============================================================
        # # Load datasets
        # complete_data = np.load(sys.argv[1])['data']
        # complete_labels = np.load(sys.argv[1])['labels']
        # val_data = np.load(sys.argv[2])['data']
        # val_labels = np.load(sys.argv[2])['labels']

        # # Create full training dataset with labels
        # complete_dataset = MNISTDataset(complete_data)
        
        # # Split into train (80%) and test (20%)
        # train_size = int(0.8 * len(complete_dataset))
        # test_size = len(complete_dataset) - train_size
        # train_dataset, test_dataset = random_split(
        #     complete_dataset,
        #     [train_size, test_size],
        #     generator=torch.Generator().manual_seed(42)
        # )

        # # Extract indices for train and test datasets
        # train_indices = train_dataset.indices
        # test_indices = test_dataset.indices

        # # Split labels using the indices
        # # train_labels = complete_labels[train_indices]
        # test_labels = complete_labels[test_indices]

        # # Create initializer dataset with labels
        # val_dataset = MNISTDataset(val_data)
        
        # # Create data loaders
        # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        # val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # model = VAE().to(device)
        # optimizer = optim.AdamW(model.parameters(), lr=1e-3,weight_decay=5e-4)
        
        # # Train VAE
        # train_vae(train_loader, model, optimizer, epochs=500)
        # val_latents = extract_latent_vectors(model, val_loader)

        # print("Visualizing raw latent space...")
        # train_latents = visualize_latent_space(model, train_loader)

        # # Reconstruct images
        # reconstructed_images = []
        # original_images = []
        # model.eval()
        
        # with torch.no_grad():
        #     for data in test_loader:
        #         data = data.to(device)
        #         recon_batch, _, _ = model(data)
        #         reconstructed_images.extend(recon_batch.cpu().numpy().reshape(-1, 28, 28))
        #         original_images.extend(data.cpu().numpy().reshape(-1, 28, 28))
        
        # reconstructed_images = np.array(reconstructed_images)
        # original_images = np.array(original_images)
        
        # # eval reconstruction
        # _ssim,_1_minus_mse = eval_reconstruction(original_images, reconstructed_images)
        # print(f"Reconstruction Evaluation: SSIM: {_ssim:.4f}, 1-MSE: {_1_minus_mse:.4f}")
        
        # # Fit GMM
        # gmm = GaussMixModel()
        # gmm.fit(train_latents, val_latents, val_labels)

        # # Get predictions
        # predictions = prediction_vae_and_gmm(model, gmm, test_loader)
        
        # # Calculate metrics
        # metrics = eval_gmm_performance(test_labels, predictions)
        
        # print("Test Set Performance:")
        # print(f"Accuracy: {metrics['accuracy']:.4f}")
        # print(f"F1 Score (Micro): {metrics['f1_micro']:.4f}")
        # print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        # print(f"Average F1: {metrics['f1_avg']:.4f}")

        # # Plot GMM clustering results
        # resps = gmm._e_step(train_latents)
        # plot_gmm(train_latents, gmm.means, gmm.covs, resps)

        # # Save model
        # torch.save(model.state_dict(), args[4])
        # with open(args[5], 'wb') as f:
        #     pickle.dump(gmm, f)

        # plot_2d_manifold(model, latent_dim=2, n=20)

    elif args[2] == "test_reconstruction":
        val_data = np.load(args[1])['data']
        val_dataset = MNISTDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        vae = VAE().to(device)
        vae.load_state_dict(torch.load(args[3]))
        reconstructed_images = []
        original_images = []
        vae.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon_batch, _, _ = vae(data)
                reconstructed_images.extend(recon_batch.cpu().numpy().reshape(-1, 28, 28))
                original_images.extend(data.cpu().numpy().reshape(-1, 28, 28))
        reconstructed_images = np.array(reconstructed_images)
        original_images = np.array(original_images)
        np.savez('vae_reconstructed.npz', data=reconstructed_images)
        print("Reconstr images saved to vae_reconstructed.npz")
        _ssim,_1_minus_mse = eval_reconstruction(original_images, reconstructed_images)
        print(f"Reconstruction Evaluation: SSIM: {_ssim:.4f}, 1-MSE: {_1_minus_mse:.4f}")
        show_reconstruction(vae, val_loader)

    elif args[2] == "test_classifier":
        test_data = np.load(args[1])
        test_images = test_data['data']
        test_labels = test_data['labels']         
        test_dataset = MNISTDataset(test_images)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        vae = VAE().to(device)
        vae.load_state_dict(torch.load(args[3]))
        with open(args[4], 'rb') as f:
            gmm = pickle.load(f)
        predictions = prediction_vae_and_gmm(vae, gmm, test_loader)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        output_file='vae.csv'
        df = pd.DataFrame({'Predicted_Label': predictions})
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")