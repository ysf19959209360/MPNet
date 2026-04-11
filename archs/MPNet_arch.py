import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)







def _haar_filters(device):
    h = torch.tensor([[1., 1.],
                      [1., 1.]], device=device) / 2.0
    g = torch.tensor([[1., -1.],
                      [1., -1.]], device=device) / 2.0
    ht = torch.tensor([[1., 1.],
                       [-1., -1.]], device=device) / 2.0
    gt = torch.tensor([[1., -1.],
                       [-1., 1.]], device=device) / 2.0
    k = torch.stack([h, g, ht, gt], dim=0).unsqueeze(1)  # [4,1,2,2]
    return k

class DWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # x: [B,C,H,W]
        B, C, H, W = x.shape
        pad_h = (H % 2)
        pad_w = (W % 2)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H = x.shape[-2]; W = x.shape[-1]

        k = _haar_filters(x.device)     # [4,1,2,2]
        k = k.repeat(C, 1, 1, 1)        # [4*C,1,2,2]

        y = F.conv2d(x, k, stride=2, groups=C)   # [B,4C,H/2,W/2]
        LL, LH, HL, HH = torch.chunk(y, 4, dim=1)
        return LL, LH, HL, HH

class IWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        x = torch.cat([LL, LH, HL, HH], dim=1)  # [B,4C,H,W]
        x = x.view(B, 4, C, H, W)
        # [B, C, 2H, 2W]
        out = torch.zeros(B, C, H*2, W*2, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = x[:, 0]
        out[:, :, 0::2, 1::2] = x[:, 1]
        out[:, :, 1::2, 0::2] = x[:, 2]
        out[:, :, 1::2, 1::2] = x[:, 3]
        return out


class SpitialUnit(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False) 
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))  

    def forward(self, x):
        x = x.permute(0,3,1,2).contiguous()
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = residual + self.gamma * x
        return out.permute(0,2,3,1).contiguous() 


class FourierUnit(nn.Module):
    def __init__(self, dim, fft_norm='ortho'):
        super().__init__()
        self.fft_norm = fft_norm

        self.amp_low = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        )
        self.amp_high = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        )
        self.phase_refine = nn.Conv2d(dim, dim, 1, bias=False)

        self.phase_scale = nn.Parameter(torch.tensor(0.01))  

    def radial_mask(self, Hf, Wf, device, dtype, cutoff=0.25, sharpness=20.0):
        yy = torch.linspace(0, 1, Hf, device=device, dtype=dtype).view(Hf, 1)
        xx = torch.linspace(0, 1, Wf, device=device, dtype=dtype).view(1, Wf)
        rr = torch.sqrt(yy ** 2 + xx ** 2)
        low = torch.sigmoid((cutoff - rr) * sharpness)
        high = 1.0 - low
        return low[None, None], high[None, None]

    def forward(self, x):
        x = x.permute(0,3,1,2).contiguous() # BCHW
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x, norm=self.fft_norm)
        amp = torch.abs(X)
        phase = torch.angle(X)
        Hf, Wf = amp.shape[-2], amp.shape[-1]
        low_mask, high_mask = self.radial_mask(Hf, Wf, x.device, x.dtype)
        amp_low = self.amp_low(amp * low_mask)
        amp_high = self.amp_high(amp * high_mask)
        amp_out = amp_low + amp_high
        phase_out = phase + self.phase_scale * self.phase_refine(phase)
        real = amp_out * torch.cos(phase_out)
        imag = amp_out * torch.sin(phase_out)
        Y = torch.complex(real, imag)
        out = torch.fft.irfft2(Y, s=(H, W), norm=self.fft_norm)
        return out.permute(0,2,3,1).contiguous()

class MLP(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        in_channels = 3 * dim
        hidden_channels = in_channels * expansion

        self.norm = nn.BatchNorm2d(in_channels)

        self.project_in = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                                groups=hidden_channels, bias=False)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_channels, in_channels, 1, 1, 0, bias=False)

        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.project_out(x)
        return identity + self.res_scale * x

class WaveletUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        self.mlp = MLP(dim, expansion=2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        LL, LH, HL, HH = self.dwt(x)

        high = torch.cat([LH, HL, HH], dim=1)
        high_enh = self.mlp(high)

        LH_out, HL_out, HH_out = torch.chunk(high_enh, 3, dim=1)
        out = self.iwt(LL, LH_out, HL_out, HH_out)
        return out.permute(0, 2, 3, 1).contiguous()

class MPGSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dim_reduced=16):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        inner = dim_head * heads

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)

        if dim_reduced is None:
            dim_reduced = dim

        self.reduce = nn.Linear(inner, dim_reduced, bias=False)
        self.expand = nn.Linear(dim_reduced, inner, bias=False)
        self.dim_reduced = dim_reduced

        self.q_spatial = SpitialUnit(dim_reduced)
        self.k_freq = FourierUnit(dim_reduced)
        self.v_wavelet = WaveletUnit(dim_reduced)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(inner, dim, bias=True)

    def forward(self, x_in):  # NHWC
        b,h,w,c = x_in.shape
        x = x_in.reshape(b,h*w,c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q_dim = self.reduce(q_inp).view(b,h,w,self.dim_reduced)
        k_dim = self.reduce(k_inp).view(b,h,w,self.dim_reduced)
        v_dim = self.reduce(v_inp).view(b,h,w,self.dim_reduced)

        q_mix = self.q_spatial(q_dim)
        k_mix = self.k_freq(k_dim)
        v_mix = self.v_wavelet(v_dim)

        q_new = self.expand(q_mix.view(b,h*w,self.dim_reduced))
        k_new = self.expand(k_mix.view(b,h*w,self.dim_reduced))
        v_new = self.expand(v_mix.view(b,h*w,self.dim_reduced))

        def pack(t): return t.view(b,h*w,self.num_heads,self.dim_head).permute(0,2,1,3)
        q,k,v = map(pack,(q_new,k_new,v_new))

        q = q.transpose(-2,-1)
        k = k.transpose(-2,-1)
        v = v.transpose(-2,-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (k @ q.transpose(-2,-1))
        attn = (attn * self.rescale).softmax(dim=-1)
        x = attn @ v
        x = x.permute(0,3,1,2).reshape(b,h*w,self.num_heads*self.dim_head)
        return self.proj(x).view(b,h,w,c)
    
class GFFN(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = dim * mult
        self.pw_in  = nn.Conv2d(dim, hidden * 2, 1, 1, bias=False)   
        self.dw     = nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False, groups=hidden)
        self.act    = nn.GELU()
        self.pw_out = nn.Conv2d(hidden, dim, 1, 1, bias=False)

    def forward(self, x):          # [B,H,W,C]
        x = x.permute(0,3,1,2).contiguous()
        u, g = torch.chunk(self.pw_in(x), 2, dim=1)  # [B,hidden,H,W]×2
        x = self.act(u) * g                          
        x = self.act(self.dw(x))
        x = self.pw_out(x)
        return x.permute(0,2,3,1)

class MPAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, MPGSA(dim=dim, dim_head=dim_head, heads=heads)),
                PreNorm(dim, GFFN(dim=dim))
            ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                MPAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        self.bottleneck = MPAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                MPAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        fea = self.embedding(x)

        fea_encoder = []
        for (MPAB, FeaDownSample) in self.encoder_layers:
            fea = MPAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        fea = self.bottleneck(fea)

        for i, (Feat_upsampler, Feat_fution, MPAB_Blcok) in enumerate(self.decoder_layers):
            fea = Feat_upsampler(fea)
            fea = Feat_fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = MPAB_Blcok(fea)

        out = self.mapping(fea) + x

        return out

class MPNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 2]):
        super(MPNet, self).__init__()
        self.denoiser = Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks
        )

    def forward(self, x):
        out = self.denoiser(x)
        return out


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    import time
    model = MPNet(n_feat=40,num_blocks=[1,2,2]).cuda()
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params: {n_param / 1_000_000:.2f}M')


