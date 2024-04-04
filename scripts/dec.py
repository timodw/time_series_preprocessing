import torch
from time import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class DenseEncoder(torch.nn.Module):
    
    def __init__(self, input_dim, latent_dim,
                 layer_sizes=[512, 128]):
        super(DenseEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layer_sizes = [input_dim] + layer_sizes + [latent_dim]
        
        layers = []
        for i in range(1, len(self.layer_sizes)):
            layers.append(torch.nn.Linear(in_features=self.layer_sizes[i-1], out_features=self.layer_sizes[i]))
            if i != len(self.layer_sizes) - 1:
                layers.append(torch.nn.LeakyReLU(.2))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, X):
        return self.model(X)
    
    
class DenseDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, output_dim,
                 layer_sizes=[128, 512]):
        super(DenseDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layer_sizes = [latent_dim] + layer_sizes + [output_dim]
        
        layers = []
        for i in range(1, len(self.layer_sizes)):
            layers.append(torch.nn.Linear(in_features=self.layer_sizes[i-1], out_features=self.layer_sizes[i]))
            if i != len(self.layer_sizes) - 1:
                layers.append(torch.nn.LeakyReLU(.2))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, z):
        return self.model(z)
    

class ClusteringLayer(torch.nn.Module):
    
    def __init__(self, n_clusters, latent_dim, cluster_centers, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.cluster_centers = torch.nn.Parameter(cluster_centers)
        #self.dropout = torch.nn.Dropout(.2)
        
    def forward(self, z):
        #z = self.dropout(z)
        squared_norm = torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = (1.0 + squared_norm / self.alpha)**(-(self.alpha + 1) / 2)
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()
        return t_dist
    
    
class DEC(torch.nn.Module):
    
    def __init__(self, n_clusters, latent_dim, encoder, cluster_centers, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.clustering_layer = ClusteringLayer(self.n_clusters, self. latent_dim, cluster_centers, alpha)
        
    def target_distribution(self, q):
        weight = q**2 / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
        
    def forward(self, X):
        z = self.encoder(X)
        return self.clustering_layer(z)
    

def pretraining(encoder, decoder,
                X_train, X_val, device='cpu',
                batch_size=128, epochs=50,
                optimizer='adam', lr=1E-3,
                loss_fn='mse', with_noise=False, verbose=True, with_evaluation=False):
    if verbose:
        print(f"Training using device: {device}")
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    training_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).to(torch.float32))
    
    training_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                      batch_size=batch_size,
                                                      drop_last=True, shuffle=True)
    
    if optimizer == 'adam':
        optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=2E-5)
    else:
        optim = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    
    if loss_fn == 'mse':
        loss_fn = torch.nn.functional.mse_loss
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        epoch_training_losses = []
        
        t_train_start = time()
        for X in iter(training_dataloader):
            X = X[0].to(device)
            
            optim.zero_grad()
            if with_noise:
                X_noise = X + torch.randn(X.size()).to(device) * .2
                z = encoder(X_noise)
            else:
                z = encoder(X)
            X_hat = decoder(z)
            loss = loss_fn(X, X_hat)
            loss.backward()
            optim.step()
            
            epoch_training_losses.append(loss.detach().item())
        t_train_end = time()
        
        if verbose:
            print(f"Epoch {epoch}\nTraining time: {t_train_end - t_train_start:.2f}s; Training loss: {np.mean(epoch_training_losses):.5f};")


def train_dec(encoder,
              X_train, X_val,
              n_clusters=3, latent_dim=10, tol=1E-3, lr=1E-3,
              device='cpu', cluster_init='kmeans', verbose=True):
    if verbose:
        print(f"Training using device: {device}")
    
    encoder = encoder.to(device)
    X_train = torch.from_numpy(X_train).to(torch.float32).to(device)
    
    # cluster init
    z_train = encoder(X_train)
    if cluster_init == 'kmeans':
        if verbose:
            print('Initializing using KMeans!')
        kmeans = KMeans(n_clusters=n_clusters, random_state=86).fit(z_train.detach().cpu().numpy())
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
        current_cluster_assignment = kmeans.predict(z_train.detach().cpu().numpy())
    elif cluster_init == 'gmm':
        if verbose:
            print('Initializing using GMM!')
        gmm = GaussianMixture(n_components=n_clusters, random_state=86).fit(z_train.detach().cpu().numpy())
        cluster_centers = torch.from_numpy(gmm.means_)
        current_cluster_assignment = gmm.predict(z_train.detach().cpu().numpy())
    prev_cluster_assignment = np.empty_like(current_cluster_assignment)
    
    dec_model = DEC(n_clusters, latent_dim, encoder, cluster_centers).to(device)
    
    optim = torch.optim.Adam(dec_model.parameters(), lr=lr, weight_decay=1E-3)
    loss_fn = torch.nn.KLDivLoss(reduction='sum')
    
    epoch = 0
    while (current_cluster_assignment != prev_cluster_assignment).sum() / len(current_cluster_assignment) > tol:
        dec_model.train()
        
        t_train_start = time()
            
        output = dec_model(X_train)
        target = dec_model.target_distribution(output).detach()

        loss = loss_fn(output.log(), target) / output.size()[0]
        optim.zero_grad()
        loss.backward()
        optim.step()

        prev_cluster_assignment = current_cluster_assignment
        current_cluster_assignment = np.argmax(output.detach().cpu().numpy(), axis=-1)
        
        t_train_end = time()
        
        if verbose:
            print(f"Epoch {epoch};\nTraining time: {t_train_end - t_train_start:.2f}s; Training loss: {loss.detach().item():.5f};")
            print(f"Percentage changed: {(current_cluster_assignment != prev_cluster_assignment).sum() / len(current_cluster_assignment) * 100:.4f}%\n")
        
        epoch += 1
    return dec_model