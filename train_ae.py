import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder
from data import get_training_and_validation_data


def training_loop(autoencoder, train_loader, val_loader,
                  criterion, optimizer, num_epochs, checkpoint='ae.pth'):

    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        # Training
        autoencoder.train()
        train_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_time = time.time() - start_time
        train_loss /= len(train_loader.dataset)

        # Validation
        autoencoder.eval()
        val_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_time = time.time() - start_time
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), checkpoint) 

        # Print epoch results
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Training Time: {train_time:.2f}s; Training Loss: {train_loss:.4f}")
        print(f"Validation Time: {val_time:.2f}s; Validation Loss: {val_loss:.4f}")
        print("-" * 50)


if __name__ == '__main__':
    root_data_path = Path('processed_data')
    dataset_id = 'K6WZL7BWHQ'
    X_train, y_train, X_val, y_val = get_training_and_validation_data(root_data_path, dataset_id)

    X_mean, X_std = X_train.mean(), X_train.std()
    X_train -= X_mean
    X_train /= X_std
    X_val -= X_mean
    X_val /= X_std

    input_dim = X_train.shape[1]
    hidden_dims = [256, 128]
    latent_dim = 64
    autoencoder = Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation='lrelu',
        latent_activation='linear',
        negative_slope=0.01
    )

    batch_size = 512
    train_tensor = torch.tensor(X_train, dtype=torch.float32).squeeze()
    val_tensor = torch.tensor(X_val, dtype=torch.float32).squeeze()
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1E-3)
    num_epochs = 30

    training_loop(autoencoder, train_loader, val_loader, criterion, optimizer, num_epochs)
    autoencoder.load_state_dict(torch.load('ae.pth'))