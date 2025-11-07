import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_model(args, parent=False):
    return Dehaze()

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Focal(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=3, focal_factor=2, bias=True):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor

        self.act = nn.GELU()
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                ) 

            self.kernel_sizes.append(kernel_size)      

    def forward(self, ctx, gates):   
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        out = ctx_all

        return out

class CrossFocalModule(nn.Module):
    def __init__(self, dim, pools_sizes=[2,1]):
        super(CrossFocalModule, self).__init__()
        self.pools_sizes = pools_sizes
        self.focal_level = 3
        pools, focals = [], []
        self.f = nn.Conv2d(dim, 2*dim + (self.focal_level+1), 1, 1, 0)
        self.h = nn.Conv2d(dim,   dim, 1, 1, 0)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

        for size in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=size, stride=size))
            focals.append(Focal(dim))

        self.pools = nn.ModuleList(pools)
        self.focals = nn.ModuleList(focals)

    def forward(self, x):
        C = x.shape[1]
        x_size = x.size()

        # pre linear projection c-> 2c + 1
        x = self.f(x)
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)

        if len(self.pools_sizes) == 1:
            # single modulator aggregation
            y = self.focals[0](ctx, self.gates)
        else:
            # cross  modulator aggregation [2, 1]  [3, 2, 1]
            for i in range(len(self.pools_sizes)):
                # lowest scale
                if i == 0:
                    feas = self.pools[i](ctx)
                    gates = self.pools[i](self.gates)
                    y = self.focals[i](feas, gates)
                # highest scale
                elif i == len(self.pools_sizes) - 1:
                    feas = ctx + y_up
                    gates = self.gates
                    y = self.focals[i](feas, gates)
                # middle scales
                else:
                    feas = self.pools[i](ctx) + y_up
                    gates = self.pools[i](self.gates)
                    y = self.focals[i](feas, gates)
                # upsample to fuse with the new ctx 
                if i != len(self.pools_sizes)-1:
                    y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True) # for later layers 

        # ========================================
        cross_modulator = y
        # ========================================

        self.modulator = self.h(cross_modulator)
        
        x_out = q*self.modulator

        # post linear porjection
        x_out = self.proj(x_out)

        return x_out

class SpaBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(SpaBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        if self.in_size != self.out_size:
            self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.layer_1 = nn.Sequential(*[
            LayerNorm2d(out_size),
            CrossFocalModule(out_size)
        ])
        self.layer_2 = nn.Sequential(*[
            nn.Conv2d(out_size, out_size,3,1,1),
            nn.GELU(),
            nn.Conv2d(out_size, out_size,3,1,1),
            nn.GELU(),
        ])
        
    def forward(self, x):
        if self.in_size != self.out_size:
            x = self.identity(x)
        x = self.layer_1(x) + x
        x = self.layer_2(x) + x
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class FreBlock(nn.Module):
    def __init__(self, dim, use_FFT_PHASE=True, use_FFT_AMP=True):
        super(FreBlock, self).__init__()
        self.use_FFT_PHASE = use_FFT_PHASE
        self.use_FFT_AMP = use_FFT_AMP
        self.norm = LayerNorm2d(dim)

        self.process_spa = nn.Sequential(
            nn.Conv2d(dim, dim,1,1,0),
            nn.GELU(),
            nn.Conv2d(dim, dim,1,1,0)
        )

        self.cha_pool = nn.Sequential(*[ChannelPool(), nn.Conv2d(2, 1, 3, 1, 1)])
        self.process_cha = nn.Sequential(
            nn.Conv2d(1, 1, 1, 1,0),
            nn.GELU(),
            nn.Conv2d(1, 1, 1, 1,0)
        )

    def forward(self, x):
        x = self.norm(x)
        b,c,h,w = x.size()
        x_fft = torch.fft.rfft2(x, norm='backward')
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        if self.use_FFT_AMP:
            x_amp = self.process_spa(x_amp)
        elif self.use_FFT_PHASE:
            x_pha = self.process_spa(x_pha)
        x_fft_out = torch.fft.irfft2(x_amp*torch.exp(1j*x_pha), norm='backward')

        x_z = self.cha_pool(x) # b 1 h w
        x_z_fft = torch.fft.rfft(x_z, norm='backward')
        x_z_amp = torch.abs(x_z_fft)
        x_z_pha = torch.angle(x_z_fft)

        if self.use_FFT_AMP:
            x_z_amp = self.process_cha(x_z_amp)
        elif self.use_FFT_PHASE:
            x_z_pha = self.process_cha(x_z_pha)
        x_z_fft_out = torch.fft.irfft(x_z_amp*torch.exp(1j*x_z_pha), norm='backward')

        out = x_fft_out * x_z_fft_out
        
        return out

class BasicBlock(nn.Module):
    # encoder 阶段：the same number of channel in input and output
    # decoder 阶段：the various channel in input and output 
    def __init__(self, in_size, out_size, downsample=False, use_csff=False, use_FFT_PHASE=False, use_FFT_AMP=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.use_csff = use_csff
        self.use_FFT_PHASE = use_FFT_PHASE
        self.use_FFT_AMP = use_FFT_AMP

        self.spatial_process = SpaBlock(in_size, out_size)
        self.fre_process = FreBlock(out_size, use_FFT_PHASE, use_FFT_AMP)
        self.fusion = nn.Conv2d(out_size*2, out_size, 1, 1, 0)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x, enc=None, dec=None):
        spa_out = self.spatial_process(x)
        fre_out = self.fre_process(spa_out)
        out = self.fusion(torch.cat([spa_out, fre_out], dim=1))

        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

# for cross-scale interaction (CSI) module 
# ======================================BEGIN======================================
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=True):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
        )
        
    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=True):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

# ======================================END======================================
class CM(nn.Module):
    def __init__(self, dim, focal_level):
        super(CM, self).__init__()
        self.focal_level = focal_level
        self.f = nn.Conv2d(dim, dim + self.focal_level, 1, 1, 0, bias=True)
        self.h = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.norm = LayerNorm2d(dim)

    def forward(self, x, x_list):
        x0 = x 
        C = x.shape[1]
        x = self.f(x)
        q, self.gates = torch.split(x, (C, self.focal_level), 1)
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = x_list[l]
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.norm(x_out)

        return x_out + x0

# Cross-scale Interaction
class CSI(nn.Module):
    def __init__(self, wf=20, depth=4):
        super(CSI, self).__init__()
        self.depth = depth
        self.wf = wf
        self.conv_fea = nn.ModuleList()
        for i in range(depth - 1):
            self.conv_fea.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))            

        self.resize_fea = nn.ModuleList()
        self.fusion_fea = nn.ModuleList()
        self.modulation = nn.ModuleList()

        for i in range(self.depth - 1):
            self.resize_fea.append(nn.ModuleList())

        for i in range(self.depth - 1):
            self.resize_fea[i] = nn.ModuleList()
            self.modulation.append(CM( (2**i)*wf, self.depth-1) )
            for j in range(self.depth - 1):
                if i < j:
                    self.resize_fea[i].append(DownSample(in_channels=(2**i)*wf, scale_factor=2**(j-i), chan_factor=2, kernel_size=3))
                elif i == j:
                    self.resize_fea[i].append(None)
                else:
                    self.resize_fea[i].append(UpSample(in_channels=(2**i)*wf, scale_factor=2**(i-j), chan_factor=2, kernel_size=3))

            self.fusion_fea.append(nn.Conv2d((2**i)*wf*(depth-1), (2**i)*wf, 1, 1, 0))

    def forward(self, x):
        feas = []

        for i in range(self.depth - 1):
            feas.append(self.conv_fea[i](x[i]))

        for i in range(self.depth - 1):
            cross_feas = [feas[i]]
            for j in range(self.depth-1):
                if i != j:
                    resize_fea = self.resize_fea[j][i](feas[j])
                    y = torch.cat([feas[i], resize_fea], dim=1)
                    cross_feas.append(resize_fea)
                    feas[i] = y
            feas[i] = self.fusion_fea[i](feas[i]) # cross-scale fusion = concat + 1×1 conv
            feas[i] = self.modulation[i](feas[i], cross_feas) # modulation

        return feas


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, use_FFT_AMP=False, use_FFT_PHASE=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        # self.up = nn.Sequential(nn.Conv2d(in_size, out_size*4, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelShuffle(2))
        self.conv_block = BasicBlock(in_size, out_size, downsample=False, use_FFT_AMP=use_FFT_AMP, use_FFT_PHASE=use_FFT_PHASE)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class FirstNet(nn.Module):
    def __init__(self, in_chn=3, wf=20, depth=4):
        super(FirstNet, self).__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.first = nn.Conv2d(in_chn, wf, 3, 1, 1)

        prev_channels = wf
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.encoder.append(BasicBlock(prev_channels, (2**i) * wf, downsample=downsample, use_FFT_AMP=True, use_FFT_PHASE=False))
            prev_channels = (2**i) * wf

        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder.append(UNetUpBlock(prev_channels, (2**i)*wf, use_FFT_AMP=True, use_FFT_PHASE=False))
            prev_channels = (2**i)*wf

        self.csi_enc = CSI(wf)
        self.csi_dec = CSI(wf)

    def forward(self, x):
        image = x
        x1 = self.first(image)
        encs = []
        decs = []
        for i, enc in enumerate(self.encoder):
            if (i+1) < self.depth:
                x1, x1_up = enc(x1)
                encs.append(x1_up)
            else:
                x1 = enc(x1)

        encs_refine = self.csi_enc(encs)

        for i, dec in enumerate(self.decoder):
            x1 = dec(x1, encs_refine[-i-1])
            decs.append(x1)        
        decs.reverse()
        decs_refine = self.csi_dec(decs)

        return x1, encs_refine, decs_refine

class SecondNet(nn.Module):
    def __init__(self, in_chn=3, wf=20, depth=4):
        super(SecondNet, self).__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.first = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.cat = nn.Conv2d(2*wf, wf, 1, 1, 0)

        prev_channels = wf
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.encoder.append(BasicBlock(prev_channels, (2**i) * wf, downsample, use_csff=downsample, use_FFT_AMP=False, use_FFT_PHASE=True))
            prev_channels = (2**i) * wf

        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder.append(UNetUpBlock(prev_channels, (2**i)*wf, use_FFT_AMP=False, use_FFT_PHASE=True))
            prev_channels = (2**i)*wf

        self.csi_enc = CSI(wf)
        self.last = nn.Conv2d(prev_channels, in_chn, 3, 1, 1, bias=True)

    def forward(self, out_1_feature, x, enc_first, dec_first):
        image = x
        x = self.first(x)
        x = self.cat(torch.cat([out_1_feature, x], dim=1))
        encs = []
        for i, enc in enumerate(self.encoder):
            if (i+1) < self.depth:
                x, x_up = enc(x, enc_first[i], dec_first[i])
                encs.append(x_up)
            else:
                x = enc(x)
        
        encs_refine = self.csi_enc(encs)

        for i, dec in enumerate(self.decoder):
            x = dec(x, encs_refine[-i-1])

        out = self.last(x)

        return out + image

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feat, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, n_feat,3, 1, 1)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1 + x
        return x1, img

class Dehaze(nn.Module):
    def __init__(self, base_channel=20):
        super(Dehaze, self).__init__()

        self.first = FirstNet(wf=base_channel)
        self.second = SecondNet(wf=base_channel)
        self.sam = SAM(base_channel)
        
    def forward(self, x):
        image = x
        out_1, encs_first, decs_first = self.first(x)

        out_1_feature, out_1 = self.sam(out_1, image)

        # frequency features exchange
        # ============================================================================================
        out_1_fft = torch.fft.rfft2(out_1, norm='backward')
        out_1_amp = torch.abs(out_1_fft)
        
        out_1_phase = torch.angle(out_1_fft)

        image_fft = torch.fft.rfft2(image, norm='backward')
        image_phase = torch.angle(image_fft)
        image_inverse = torch.fft.irfft2(out_1_amp*torch.exp(1j*image_phase), norm='backward')
        # ============================================================================================

        out_2 = self.second(out_1_feature, image_inverse, encs_first, decs_first)
        
        return [out_1, out_1_amp, out_1_phase, out_2, image_phase]

if __name__ == '__main__':
    model = Dehaze()
    x = torch.randn(1,3,256,256)
    # y = model(x)

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9))
    # Params and FLOPs are 3.423651M/18.385677952G
