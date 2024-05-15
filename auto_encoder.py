import os
import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = torch.nn.MultiheadAttention(channels, num_heads=1, batch_first=True)
        self.ln = torch.nn.LayerNorm([channels])
        self.ff_self = torch.nn.Sequential(
            torch.nn.LayerNorm([channels]),
            torch.nn.Linear(channels, channels),
            torch.nn.GELU(),
            torch.nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)
    

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_1 = torch.nn.GroupNorm(32, in_channels)
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = torch.nn.GroupNorm(32, out_channels)
        self.conv_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = torch.nn.Identity()
        else:
            self.residual_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x

        x = self.group_norm_1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_1(x)

        x = self.group_norm_2(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

    
class Encoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            # output: 64 x 64 x 64
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 64),
            # output: 128 x 32 x 32
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 128),
            # output: 256 x 16 x 16
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 256),
            SelfAttention(256),
            ResidualBlock(256, 256),
            torch.nn.GroupNorm(32, 256),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 8, kernel_size=3, padding=1),
            torch.nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = torch.nn.functional.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        noise = torch.randn(*mean.size()).float().cuda()
        x = mean + stdev * noise

        x *= 0.18215
        return mean, log_variance, x


class Decoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(4, 4, kernel_size=1, padding=0),
            torch.nn.Conv2d(4, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 256),
            SelfAttention(256),
            ResidualBlock(256, 256),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 64),
            torch.nn.GroupNorm(32, 64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x

# load weights
encoder = Encoder().float().cuda()
decoder = Decoder().float().cuda()

encoder.load_state_dict(torch.load(os.path.join("diffusion_vae", "encoder_ckpt.pt")))
decoder.load_state_dict(torch.load(os.path.join("diffusion_vae", "decoder_ckpt.pt")))

encoder.eval()
decoder.eval()

def encode_images(images: torch.Tensor):
    images = images.cuda()
    _, _, z = encoder(images)
    return z

def decode_images(x: torch.Tensor):
    x = decoder(x)
    return x