import time
from pathlib import Path
import tempfile
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

import ray
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from autoencoder import Autoencoder
from data import get_training_and_validation_data


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 50

DATA_ROOT = Path('/home/timodw/IDLab/time_series_preprocessing/processed_data')
DATASET_ID = 'RTAGXFQJ4T' # With FFT


def train_autoencoder(config, verbose=False, ray_tune=True):
    X_train, _, X_val, _ = get_training_and_validation_data(DATA_ROOT, DATASET_ID)

    X_mean, X_std = X_train.mean(), X_train.std()
    X_train -= X_mean
    X_train /= X_std
    X_val -= X_mean
    X_val /= X_std
    
    batch_size = config['batch_size']
    train_tensor = torch.tensor(X_train, dtype=torch.float32).squeeze()
    val_tensor = torch.tensor(X_val, dtype=torch.float32).squeeze()
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    autoencoder = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dims=[config['l1'], config['l2']],
        latent_dim=config['latent_dim'],
        activation=config['activation'],
        latent_activation=config['latent_activation'],
        negative_slope=config['negative_slope']
    ).to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config['lr'])

    if ray_tune:
        checkpoint = ray.train.get_checkpoint()
    else:
        checkpoint = None
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
            start_epoch = checkpoint_dict['epoch'] + 1
            autoencoder.load_state_dict(checkpoint_dict['model_state'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state'])
    else:
        start_epoch = 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Training
        autoencoder.train()
        train_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            inputs = batch[0].to(DEVICE)
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
                inputs = batch[0].to(DEVICE)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_time = time.time() - start_time
        val_loss /= len(val_loader.dataset)

        if ray_tune:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(checkpoint_data, os.path.join(tempdir, 'checkpoint.pt'))
                    ray.train.report({'loss': val_loss}, checkpoint=Checkpoint.from_directory(tempdir))

        if verbose:
            print(f"Epoch {epoch}/{NUM_EPOCHS}")
            print(f"Training Time: {train_time:.2f}s; Training Loss: {train_loss:.4f}")
            print(f"Validation Time: {val_time:.2f}s; Validation Loss: {val_loss:.4f}")
            print("-" * 50)


if __name__ == '__main__':
    config = {
        'l1': 256,
        'l2': 128,
        'latent_dim': 64,
        'activation': 'lrelu',
        'latent_activation': 'lrelu',
        'negative_slope': 0.025,
        'batch_size': 32,
        'lr': 1E-4
    }

    train_autoencoder(config, verbose=True, ray_tune=False)

    # search_space = {
    #     "l1": tune.choice([64, 128, 256]),
    #     "l2": tune.choice([32, 64, 128]),
    #     "latent_dim": tune.choice([16, 32, 64]),
    #     "activation": tune.choice(["lrelu", "relu", "sigmoid", "tanh", "linear"]),
    #     "latent_activation": tune.choice(["lrelu", "relu", "sigmoid", "tanh", "linear"]),
    #     "negative_slope": tune.uniform(0.01, 0.1),
    #     "batch_size": tune.choice([16, 32, 64]),
    #     "lr": tune.loguniform(1e-4, 1e-2)
    # }

    # scheduler = ASHAScheduler(
    #     metric='loss',
    #     mode='min',
    #     max_t=50,
    #     grace_period=1,
    #     reduction_factor=2
    # )

    # result = tune.run(
    #     train_autoencoder,
    #     resources_per_trial={'cpu': 4, 'gpu': 1},
    #     config=search_space,
    #     num_samples=50,
    #     scheduler=scheduler,
    #     verbose=1
    # )

    # print('Best configuration:', result.get_best_config(metric='loss', mode='min'))
