import torch
import torchaudio
from .ComplexLayers import ComplexSequential, ComplexConv1d, ComplexBatchNorm2d, complex_LeakyReLU, ComplexConvTranspose1d

def enc(in_channels, out_channels, kernel, stride):
    return (ComplexSequential(
        ComplexConv1d(in_channels, out_channels, kernel, stride=stride),
        ComplexBatchNorm2d(out_channels),
        complex_LeakyReLU()
        ))

def dec(in_channels, out_channels, kernel, stride, output_padding):
    return (ComplexSequential(
        ComplexConvTranspose1d(in_channels, out_channels, kernel, stride=stride, output_padding=output_padding),
        ComplexBatchNorm2d(out_channels),
        complex_LeakyReLU()
        ))


class ComplexUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = enc(1, 32, (7, 5), (2,2))
        self.enc2 = enc(32, 64, (7, 5), (2,2))
        self.enc3 = enc(64, 64, (5, 3), (2,2))
        self.enc4 = enc(64, 64, (5, 3), (2,2))
        self.enc5 = enc(64, 64, (5, 3), (2,1))

        self.dec1 = ComplexConvTranspose1d(32+32, 1, (7, 5), (2,2), output_padding=(0,1)) #out?
        self.dec2 = dec(64+64, 32, (7, 5), (2,2), (1, 0))
        self.dec3 = dec(64+64, 64, (5, 3), (2,2), (1, 1))
        self.dec4 = dec(64+64, 64, (5, 3), (2,2), (1, 1))
        self.dec5 = dec(64, 64, (5, 3), (2,1), (1, 0))

        self.spec = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=256, power=None)
    def forward(self, input):
        length = input.shape[1]

        spec = self.spec(input)
        input_r, input_i = spec[:,:,:,0], spec[:,:,:,1]
        input_r, input_i = torch.unsqueeze(input_r, 1), torch.unsqueeze(input_i, 1)

        enc1_r, enc1_i = self.enc1(input_r, input_i)
        enc2_r, enc2_i = self.enc2(enc1_r, enc1_i)
        enc3_r, enc3_i = self.enc3(enc2_r, enc2_i)
        enc4_r, enc4_i = self.enc4(enc3_r, enc3_i)
        enc5_r, enc5_i = self.enc5(enc4_r, enc4_i)

        x_r, x_i = self.dec5(enc5_r, enc5_i)
        #print(x_r.shape, x_i.shape, enc4_r.shape, enc4_i.shape)
        x_r, x_i = self.dec4(torch.cat([x_r, enc4_r], dim=1), torch.cat([x_i, enc4_i], dim=1))
        #print(x_r.shape, x_i.shape, enc3_r.shape, enc3_i.shape)
        x_r, x_i = self.dec3(torch.cat([x_r, enc3_r], dim=1), torch.cat([x_i, enc3_i], dim=1))
        #print(x_r.shape, x_i.shape, enc2_r.shape, enc2_i.shape)
        x_r, x_i = self.dec2(torch.cat([x_r, enc2_r], dim=1), torch.cat([x_i, enc2_i], dim=1))
        #print(x_r.shape, x_i.shape, enc1_r.shape, enc1_i.shape)
        x_r, x_i = self.dec1(torch.cat([x_r, enc1_r], dim=1), torch.cat([x_i, enc1_i], dim=1))

        abs_x = torch.sqrt(x_r**2 + x_i**2)
        M_mag = torch.tanh(abs_x)
        M_ph_r, M_ph_i = x_r / abs_x, x_i / abs_x
        M_r, M_i = M_mag * M_ph_r, M_mag * M_ph_i
        #print(M_r.shape, M_i.shape, input_r.shape, input_i.shape)
        output_r, output_i = input_r*M_r - input_i*M_i, M_r*input_i + input_r*M_i
        output = torch.stack([output_r, output_i], dim=-1)

        output = torchaudio.functional.istft(output, n_fft=1024, hop_length=256, win_length=1024, length=length)
        output = torch.squeeze(output)
        return output
