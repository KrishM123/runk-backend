import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, encoder_dims, decoder_dims):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            *[
                layer 
                for i in range(1, len(encoder_dims) - 1)
                for layer in (nn.Linear(encoder_dims[i-1], encoder_dims[i]), nn.ReLU())
            ],
            nn.Linear(encoder_dims[-2], encoder_dims[-1])
        )
        # Decoder
        self.decoder = nn.Sequential(
            *[
                layer 
                for i in range(1, len(decoder_dims) - 1)
                for layer in (nn.Linear(decoder_dims[i-1], decoder_dims[i]), nn.ReLU())
            ],
            nn.Linear(decoder_dims[-2], decoder_dims[-1])
        )
    def decode(self, x):
        return self.decoder(x)
    def encode(self, x):
        return self.encoder(x)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x