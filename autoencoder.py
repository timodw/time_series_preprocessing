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


def vae_loss(recon_x, x, mu, logvar, kl_weight=1E-3):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kl_weight * KLD


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='lrelu', latent_activation='lrelu', negative_slope=0.01):
        super(VariationalAutoencoder, self).__init__()
        
        self.activation = self._get_activation_function(activation, negative_slope)
        self.latent_activation = self._get_activation_function(latent_activation, negative_slope)
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self.activation)
            prev_dim = hidden_dim
        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.fc_mu = torch.nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(prev_dim, latent_dim)

        # Decoder
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

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def gumbel_elbo_loss(X_pred, X_true, p, kl_weight=1E-3, num_cat=4):
    rec_loss = torch.nn.functional.mse_loss(X_pred, X_true)
    logits = torch.nn.functional.log_softmax(p, dim=-1)
    kl = torch.nn.functional.kl_div(logits, torch.ones_like(logits) / p.size()[-1])
    return rec_loss + kl_weight * kl

class CategoricalAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='lrelu', latent_activation='lrelu', negative_slope=0.01):
        super(CategoricalAutoencoder, self).__init__()
        
        self.activation = self._get_activation_function(activation, negative_slope)
        self.latent_activation = self._get_activation_function(latent_activation, negative_slope)
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self.activation)
            prev_dim = hidden_dim
        encoder_layers.append(torch.nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(self.latent_activation)
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder
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

    def encode(self, x):
        p = self.encoder(x)
        return p

    def reparameterize(self, p, temperature=0.5, epsilon=1E-7):
        g = torch.rand_like(p)
        g = -torch.log(-torch.log(g + epsilon) + epsilon)

        z = torch.nn.functional.softmax((p + g) / temperature, dim=-1)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        p = self.encode(x)
        z = self.reparameterize(p)
        return self.decode(z), p


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
