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
    logits = torch.log_softmax(p, dim=-1)
    kl = torch.nn.functional.kl_div(logits, torch.ones_like(logits) / p.size()[-1])
    return rec_loss + kl_weight * kl


class CategoricalAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim,
                 activation='lrelu', negative_slope=0.01, temperature=.5,
                 dropout=.5, **kwargs):
        super(CategoricalAutoencoder, self).__init__()
        
        self.activation = self._get_activation_function(activation, negative_slope)
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        encoder_layers.append(torch.nn.Dropout1d(dropout))
        for hidden_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self.activation)
            prev_dim = hidden_dim
        encoder_layers.append(torch.nn.Linear(prev_dim, latent_dim))
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
    def __init__(self, input_channels, hidden_size,
                 hidden_channels, strides, kernel_sizes, latent_dim,
                 decoder_output_padding, conv_output,
                 activation='lrelu',
                 negative_slope=0.01, temperature=0.5, **kwargs):
        super(ConvolutionalCategoricalAutoencoder, self).__init__()

        self.activation = self._get_activation_function(activation, negative_slope)
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Encoder
        encoder_layers = []
        prev_channels = input_channels
        for hidden_channel, stride, kernel_size in zip(hidden_channels, strides, kernel_sizes):
            encoder_layers.append(torch.nn.Conv1d(prev_channels, hidden_channel, kernel_size, stride=stride))
            encoder_layers.append(self.activation)
            prev_channels = hidden_channel
        encoder_layers.append(torch.nn.Flatten())
        encoder_layers.append(torch.nn.Linear(conv_output, hidden_size))
        encoder_layers.append(self.activation)
        encoder_layers.append(torch.nn.Linear(hidden_size, latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        reversed_hidden_channels = list(reversed(hidden_channels))
        reversed_strides = list(reversed(strides))
        reversed_kernel_sizes = list(reversed(kernel_sizes))
        decoder_layers.append(torch.nn.Linear(latent_dim, hidden_size))
        decoder_layers.append(self.activation)
        decoder_layers.append(torch.nn.Linear(hidden_size, conv_output))
        decoder_layers.append(self.activation)
        decoder_layers.append(torch.nn.Unflatten(1, (reversed_hidden_channels[0], -1)))
        prev_channels = reversed_hidden_channels[0]
        for hidden_channel, stride, kernel_size, output_padding in zip(reversed_hidden_channels[1:], reversed_strides[:-1], reversed_kernel_sizes[:-1], decoder_output_padding[:-1]):
            decoder_layers.append(torch.nn.ConvTranspose1d(prev_channels, hidden_channel, kernel_size, stride=stride, output_padding=output_padding))
            decoder_layers.append(self.activation)
            prev_channels = hidden_channel
        decoder_layers.append(torch.nn.ConvTranspose1d(prev_channels, input_channels, reversed_kernel_sizes[-1], stride=reversed_strides[-1], output_padding=decoder_output_padding[-1]))
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
        x_hat = self.decode(z)
        return x_hat, p


if __name__ == '__main__':
    input_dim = 200
    hidden_channels = [8, 16, 32]
    strides = [1, 2, 4]
    kernel_sizes = [2, 4, 6]
    decoder_output_padding = [0, 1, 0]
    latent_dim = 128
    conv_out = latent_dim * 24

    autoencoder = ConvolutionalCategoricalAutoencoder(
        input_channels=1,
        hidden_channels=hidden_channels,
        strides=strides,
        kernel_sizes=kernel_sizes,
        decoder_output_padding=decoder_output_padding,
        conv_output=conv_out,
        latent_dim=latent_dim,
        activation='lrelu',
        negative_slope=0.01
    )

    print(autoencoder)
    print(autoencoder(torch.randn(64, 200))[0].size())
