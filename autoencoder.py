import torch

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='lrelu', latent_activation='lrelu', negative_slope=0.01):
        super(Autoencoder, self).__init__()
        
        self.activation = self._get_activation_function(activation, negative_slope)
        self.latent_activation = self._get_activation_function(latent_activation, negative_slope)

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self.activation)
            prev_dim = hidden_dim
        encoder_layers.append(torch.nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(self.latent_activation)
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(self.activation)
            prev_dim = hidden_dim
        decoder_layers.append(torch.nn.Linear(prev_dim, input_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def _get_activation_function(self, activation, negative_slope):
        if activation == 'lrelu':
            return torch.nn.LeakyReLU(negative_slope=negative_slope)
        elif activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation == 'tanh':
            return torch.nn.Tanh()
        elif activation == 'linear':
            return torch.nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    input_dim = 200
    hidden_dims = [256, 128]
    latent_dim = 64

    autoencoder = Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation='lrelu',
        latent_activation='tanh',
        negative_slope=0.01
    )

    print(autoencoder)
