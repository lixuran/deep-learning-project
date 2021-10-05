import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def flatten(x):
    return to_var(x.view(x.size(0), -1))
class UnFlatten(nn.Module):
    def __init__(self, size, dim=1):
        super(UnFlatten, self).__init__()
        self.size = size
        self.dim = dim

    def forward(self, input):
        return input.view(input.size(0), self.size, self.dim, self.dim)  # todo : this is not good


class Encoder(nn.Module):
    def __init__(self, input_size, conv_layers, fc_layers, z_dim):
        # fc_layers doesnt include the last layer.
        # work with both conv and fc layers. :)
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.prev_fc_size = -1
        self.last_img_dim = -1
        layers = []
        prev_channels = 1
        img_dim = input_size
        if conv_layers != None:

            for n_channels, kernel_size, stride, padding in conv_layers:
                layers += [
                    nn.Conv2d(int(prev_channels), int(n_channels), int(kernel_size), stride=int(stride),
                              padding=int(padding)),
                    nn.LeakyReLU(0.2),  # todo: should also try normal relu
                ]
                prev_channels = n_channels
                # img_dim = img_dim // stride #todo: this need rework might be wrong
                img_dim_new = np.floor((img_dim + padding * 2 - kernel_size) / stride + 1)
                assert img_dim_new == img_dim // stride  # for now assume img dim shrink to half every time
                img_dim = img_dim_new
        layers += [nn.Flatten()]
        #print("test1")
        self.prev_fc_size = prev_channels * img_dim * img_dim

        self.last_img_dim = img_dim
        for i, fc_size in enumerate(fc_layers):
            #print(self.prev_fc_size, fc_size)
            layers += [nn.Linear(int(self.prev_fc_size), int(fc_size))]
            # if i + 1 < len(fc_layers):
            layers += [nn.LeakyReLU(0.2)]
            self.prev_fc_size = fc_size
        layers += [nn.Linear(int(self.prev_fc_size), int(z_dim * 2))]
        self.layers = nn.Sequential(*layers)
        #print("test2")

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, h_dim, z_dim=20):
        super(Decoder, self).__init__()
        self.h_dim = h_dim

        self.linear1 = nn.Linear(z_dim, h_dim)
        # self.hidden_layers=conv_layers.reverse()

        self.layers = nn.Sequential(  # only works for minst image 1*28*28
            UnFlatten(h_dim),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.linear1(x)
        out = self.layers(z)
        assert out.shape[2] == 28
        return out


class VAE_CNN(nn.Module):
    def __init__(self, conv_layers=None, fc_layers=None, input_size=28, z_dim=20,KL_weight=-0.5):
        super(VAE_CNN, self).__init__()
        # requires fc
        # self.encoder = nn.Sequential(
        #     nn.Linear(image_size, h_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(h_dim, z_dim*2)
        # )
        self.encoder = Encoder(input_size, conv_layers, fc_layers, z_dim)
        self.weight = KL_weight
        self.decoder = Decoder(self.encoder.prev_fc_size, z_dim)

    def loss_fn(self, recon_x, x, mu, logvar):
        # todo: probably flatten recon_x if its not already flat

        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = self.weight * torch.mean(
            1 + logvar - mu ** 2 - logvar.exp())  # todo: check out why this is the case also for conv use mean?

        # KLD = weight * torch.sum(1 + logvar - mu**2 -  logvar.exp())#todo: check out why this is the case also for conv use mean?
        return BCE + KLD  # todo: bce might be wrong ?
    def loss_fn_each(self, recon_x, x, mu, logvar):
        # todo: probably flatten recon_x if its not already flat

        BCE = F.binary_cross_entropy(recon_x, x, size_average=False,reduce=False)

        assert len(BCE.shape) >= 2
        BCE = torch.sum(BCE, [i for i in range(1,len(BCE.shape))])
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = self.weight * torch.mean(
            1 + logvar - mu ** 2 - logvar.exp(),1)  # todo: check out why this is the case also for conv use mean?
        #print("BCE ",BCE.shape)
        #print("KLD ",KLD.shape)

        # KLD = weight * torch.sum(1 + logvar - mu**2 -  logvar.exp())#todo: check out why this is the case also for conv use mean?
        return BCE + KLD  # todo: bce might be wrong ?
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar