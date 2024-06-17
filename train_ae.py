import time
from pathlib import Path
import tempfile
import os
import numpy as np
import argparse
from hashlib import sha1
import json
from base64 import b32encode
from functools import partial

import torch
from torch.utils.data import DataLoader, TensorDataset

import ray
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from autoencoder import Autoencoder, VariationalAutoencoder, CategoricalAutoencoder, ConvolutionalCategoricalAutoencoder
from autoencoder import vae_loss, gumbel_elbo_loss
from data import get_training_and_validation_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 200
MODEL = 'ae'

DATA_ROOT = Path('/home/timodw/IDLab/time_series_preprocessing/processed_data')
# DATA_ROOT = Path('/Users/timodewaele/Developer/IDLab/time_series_preprocessing/processed_data')
# DATASET_ID = 'RTAGXFQJ4T' # With FFT
DATASET_ID = 'K6WZL7BWHQ' # Standard


def train_autoencoder(config, verbose=False, ray_tune=True, checkpoint_folder=None):
    if 'classes' in config:
        classes = config['classes']
    else:
        classes = None
    X_train, _, X_val, _ = get_training_and_validation_data(DATA_ROOT, DATASET_ID, classes=classes, balanced=True)

    # X_mean, X_std = X_train.mean(), X_train.std()
    # X_train -= X_mean
    # X_train /= X_std
    # X_val -= X_mean
    # X_val /= X_std

    X_min, X_max = X_train.min(), X_train.max()
    X_train -= X_min
    X_train /= (X_max - X_min)
    X_val -= X_min
    X_val /= (X_max - X_min)
    
    batch_size = config['batch_size']
    train_tensor = torch.tensor(X_train, dtype=torch.float32).squeeze()
    val_tensor = torch.tensor(X_val, dtype=torch.float32).squeeze()
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if MODEL == 'vae':
        model = VariationalAutoencoder
    elif MODEL == 'cae':
        model = CategoricalAutoencoder
    elif MODEL == 'convcae':
        model = ConvolutionalCategoricalAutoencoder
    else:
        model = Autoencoder

    autoencoder = model(input_dim=train_tensor.size(1), **config).to(DEVICE)
    print(autoencoder)

    if MODEL == 'vae':
        criterion = partial(vae_loss, kl_weight=config['kl_weight'])
    elif MODEL == 'cae' or MODEL == 'convcae':
        criterion = partial(gumbel_elbo_loss, kl_weight=config['kl_weight'])
    else:
        criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

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

    best_val_loss = float('inf')
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Training
        autoencoder.train()
        train_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            inputs = batch[0].to(DEVICE)
            optimizer.zero_grad()
            if MODEL == 'vae':
                outputs, mu, logvar = autoencoder(inputs)
                loss = criterion(outputs, inputs, mu, logvar)
            elif MODEL == 'cae' or MODEL == 'convcae':
                outputs, p = autoencoder(inputs)
                loss = criterion(outputs, inputs, p)
            else:
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
                if MODEL == 'vae':
                    outputs, mu, logvar = autoencoder(inputs)
                    loss = criterion(outputs, inputs, mu, logvar)
                elif MODEL == 'cae' or MODEL == 'convcae':
                    outputs, p = autoencoder(inputs)
                    loss = criterion(outputs, inputs, p)
                else:
                    outputs = autoencoder(inputs)
                    loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_time = time.time() - start_time
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_folder:
                torch.save(autoencoder.state_dict(), checkpoint_folder / 'ae.pth') 

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    
    if not args.tune:
        config = {
            'hidden_dims': [512, 128, 64],
            'latent_dim': 16,
            'activation': 'lrelu',
            'latent_activation': 'linear',
            'negative_slope': 0.025,
            'batch_size': 512,
            'lr': 1E-4,
            'classes': [1, 2, 3],
            # 'kl_weight': 2E-3,
            # 'temperature': .5,
            # 'dropout': .0,
            'weight_decay': .0
        }
        # config = {
        #     'hidden_size': 512,
        #     'hidden_channels': [8, 16, 32],
        #     'strides': [1, 2, 4],
        #     'kernel_sizes': [2, 4, 6],
        #     'decoder_output_padding': [0, 1, 0],
        #     'conv_output': 768,
        #     'latent_dim': 32,
        #     'input_channels': 1,
        #     'activation': 'lrelu',
        #     'negative_slope': 0.025,
        #     'batch_size': 64,
        #     'lr': 1E-4,
        #     'kl_weight': 1E-3,
        #     'temperature': .5,
        #     'weight_decay': 0.
        # }
        training_parameters = {
            'dataset_id': DATASET_ID,
            'model': MODEL
        }
        config_json = json.dumps({**config, **training_parameters}, sort_keys=True, indent=4)
        config_hash = b32encode(sha1(repr(config_json).encode()).digest()).decode()[:10]
        output_folder = Path('models') / config_hash
        output_folder.mkdir(exist_ok=True, parents=True)
        (output_folder / 'config.json').write_text(config_json + '\n')
        print(f"Training model configuration {config_hash}")

        train_autoencoder(config, verbose=True, ray_tune=False, checkpoint_folder=output_folder)
        print(f"Finnished training configuration {config_hash}")
    else:
        search_space = {
            "l1": tune.choice([64, 128, 256]),
            "l2": tune.choice([32, 64, 128]),
            "latent_dim": tune.choice([16, 32, 64]),
            "activation": tune.choice(["lrelu", "relu", "sigmoid", "tanh", "linear"]),
            "latent_activation": tune.choice(["lrelu", "relu", "sigmoid", "tanh", "linear"]),
            "negative_slope": tune.uniform(0.01, 0.1),
            "batch_size": tune.choice([16, 32, 64]),
            "lr": tune.loguniform(1e-4, 1e-2)
        }

        scheduler = ASHAScheduler(
            metric='loss',
            mode='min',
            max_t=50,
            grace_period=1,
            reduction_factor=2
        )

        result = tune.run(
            train_autoencoder,
            resources_per_trial={'cpu': 4, 'gpu': 1},
            config=search_space,
            num_samples=50,
            scheduler=scheduler,
            verbose=1
        )

        print('Best configuration:', result.get_best_config(metric='loss', mode='min'))
