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


def gumbel_elbo_loss(X_pred, X_true, p, kl_weight=1E-3):
    rec_loss = torch.nn.functional.mse_loss(X_pred, X_true)
    logits = torch.nn.functional.log_softmax(p, dim=-1)
    kl = torch.nn.functional.kl_div(logits, torch.ones_like(logits) / p.size()[-1])
    return rec_loss + kl_weight * kl



class CategoricalAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim,
                 activation='lrelu', latent_activation='lrelu',
                 negative_slope=0.01, temperature=.5):
        super(CategoricalAutoencoder, self).__init__()
        
        self.activation = self._get_activation_function(activation, negative_slope)
        self.latent_activation = self._get_activation_function(latent_activation, negative_slope)
        self.latent_dim = latent_dim
        self.temperature = temperature

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
        z = self.reparameterize(p, temperature=self.temperature)
        return self.decode(z), p


class RepeatLayer(torch.nn.Module):
    def __init__(self, repeats, dim):
        super(RepeatLayer, self).__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x):
        x = x.repeat_interleave(self.repeats, dim=self.dim)
        return x


class UnsqueezeLayer(torch.nn.Module):
    def __init__(self, dim):
        super(UnsqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class ConvolutionalCategoricalAutoencoder(torch.nn.Module):
    def __init__(self, input_channels,
                 hidden_channels, strides, kernel_sizes, latent_dim,
                 activation='lrelu', latent_activation='lrelu',
                 negative_slope=0.01, temperature=0.5, **kwargs):
        super(ConvolutionalCategoricalAutoencoder, self).__init__()

        self.activation = self._get_activation_function(activation, negative_slope)
        self.latent_activation = self._get_activation_function(latent_activation, negative_slope)
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Encoder
        encoder_layers = []
        prev_channels = input_channels
        for hidden_channel, stride, kernel_size in zip(hidden_channels, strides, kernel_sizes):
            encoder_layers.append(torch.nn.Conv1d(prev_channels, hidden_channel, kernel_size, stride=stride, padding=1))
            encoder_layers.append(self.activation)
            prev_channels = hidden_channel
        encoder_layers.append(torch.nn.AdaptiveMaxPool1d(1))
        encoder_layers.append(torch.nn.Flatten())
        encoder_layers.append(torch.nn.Linear(prev_channels, latent_dim))
        encoder_layers.append(self.latent_activation)
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        reversed_hidden_channels = list(reversed(hidden_channels))
        reversed_strides = list(reversed(strides))
        reversed_kernel_sizes = list(reversed(kernel_sizes))
        decoder_layers.append(torch.nn.Linear(latent_dim, reversed_hidden_channels[0]))
        decoder_layers.append(UnsqueezeLayer(dim=-1))
        decoder_layers.append(RepeatLayer(7, dim=-1))
        prev_channels = reversed_hidden_channels[0]
        for hidden_channel, stride, kernel_size in zip(reversed_hidden_channels[1:], reversed_strides[:-1], reversed_kernel_sizes[:-1]):
            decoder_layers.append(torch.nn.ConvTranspose1d(prev_channels, hidden_channel, kernel_size, stride=stride, padding=1, output_padding=1))
            decoder_layers.append(self.activation)
            prev_channels = hidden_channel
        decoder_layers.append(torch.nn.ConvTranspose1d(prev_channels, input_channels, reversed_kernel_sizes[-1], stride=reversed_strides[-1], padding=2, output_padding=1))
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
        x = x.unsqueeze(1)
        p = self.encoder(x)
        return p

    def reparameterize(self, p, temperature=0.5, epsilon=1E-7):
        g = torch.rand_like(p)
        g = -torch.log(-torch.log(g + epsilon) + epsilon)

        z = torch.nn.functional.softmax((p + g) / temperature, dim=-1)
        return z

    def decode(self, z):
        out = self.decoder(z)
        return out.squeeze()

    def forward(self, x):
        p = self.encode(x)
        z = self.reparameterize(p, temperature=self.temperature)
        return self.decode(z), p


if __name__ == '__main__':
    input_dim = 200
    hidden_channels = [16, 32, 64]
    strides = [3, 3, 3]
    kernel_sizes = [5, 5, 5]
    latent_dim = 16

    autoencoder = ConvolutionalCategoricalAutoencoder(
        input_channels=1,
        input_length=input_dim,
        hidden_channels=hidden_channels,
        strides=strides,
        kernel_sizes=kernel_sizes,
        latent_dim=latent_dim,
        activation='lrelu',
        latent_activation='linear',
        negative_slope=0.01
    )

    print(autoencoder)
