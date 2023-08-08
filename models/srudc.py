import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from .norm_layer import *
from einops import rearrange


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ConvLayer(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
		)

		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.proj = nn.Conv2d(dim, dim, 1)

		self.CA = CALayer(dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		out = self.CA(out)
		return out


class BasicBlock(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
		super().__init__()
		self.norm = norm_layer(dim)
		self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)

	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x


class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)
		# self.chaConv = nn.Conv2d(dim, dim*height, 1, bias=False)
		#
		# self.qk = nn.Conv2d(dim*height, dim * height * 2, kernel_size=1, bias=False)
		# self.v_conv = nn.Conv2d(dim * height, dim * height, kernel_size=1, bias=False)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats * attn, dim=1)

		# in_feats_2c = torch.cat(in_feats, dim=1)
		# in_feats_2c = in_feats_2c.view(B, self.height, C, H, W)
		#
		# feats_sum = torch.sum(in_feats_2c, dim=1)
		#
		# feats_cha = self.chaConv(feats_sum)
		#
		# qk = self.qk(feats_cha)
		# q, k = qk.chunk(2, dim=1)
		# v = self.v_conv(torch.cat(in_feats, dim=1))
		#
		# q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.height)
		# k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.height)
		# v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.height)
		#
		# q = torch.nn.functional.normalize(q, dim=-1)
		# k = torch.nn.functional.normalize(k, dim=-1)
		#
		# attn = q @ k.transpose(-2, -1)
		# attn = attn.softmax(dim=-1)
		#
		# out = (attn @ v)
		#
		# out = rearrange(out, 'b head c (h w) -> b head c h w', head=self.height, h=H, w=W)
		# out = torch.sum(out, dim=1)
		return out


class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.2, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.2, inplace=True))
        return x[0] * (scale + 1) + shift
########## From Restormer
##########################################################################
## Layer Norm

def to_3d(x):
	return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
	return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(BiasFree_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(WithBias_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		mu = x.mean(-1, keepdim=True)
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
	def __init__(self, dim, LayerNorm_type):
		super(LayerNorm, self).__init__()
		if LayerNorm_type == 'BiasFree':
			self.body = BiasFree_LayerNorm(dim)
		else:
			self.body = WithBias_LayerNorm(dim)

	def forward(self, x):
		h, w = x.shape[-2:]
		return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
	def __init__(self, dim, ffn_expansion_factor, bias):
		super(FeedForward, self).__init__()

		hidden_features = int(dim * ffn_expansion_factor)

		self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

		self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
								groups=hidden_features * 2, bias=bias)

		self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

	def forward(self, x):
		x = self.project_in(x)
		x1, x2 = self.dwconv(x).chunk(2, dim=1)
		x = F.gelu(x1) * x2
		x = self.project_out(x)
		return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
	def __init__(self, dim, num_heads, bias):
		super(Attention, self).__init__()
		self.num_heads = num_heads
		self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

		self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
		self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
									padding=1, groups=dim * 3, bias=bias)
		self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

	def forward(self, x):
		b, c, h, w = x.shape

		qkv = self.qkv_dwconv(self.qkv(x))
		q, k, v = qkv.chunk(3, dim=1)

		q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
		k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
		v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

		q = torch.nn.functional.normalize(q, dim=-1)
		k = torch.nn.functional.normalize(k, dim=-1)

		attn = (q @ k.transpose(-2, -1)) * self.temperature
		attn = attn.softmax(dim=-1)

		out = (attn @ v)

		out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

		out = self.project_out(out)
		return out


##########################################################################
class TransformerBlock(nn.Module):
	def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
		super(TransformerBlock, self).__init__()

		self.norm1 = LayerNorm(dim, LayerNorm_type)
		self.attn = Attention(dim, num_heads, bias)
		self.norm2 = LayerNorm(dim, LayerNorm_type)
		self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.ffn(self.norm2(x))

		return x


class SRUDC(nn.Module):
	def __init__(self, kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(SRUDC, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths)
		half_num = stage_num // 2
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)
		self.condconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()
		#fuse conditional features into main branch
		self.fusions_cond = nn.ModuleList()


		#backbone of condition branch
		self.layers_cond1 = nn.Sequential(*[TransformerBlock(dim=embed_dims[0], num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(2)])
		self.downs_cond1_2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])  ## From Level 1 to Level 2
		self.layers_cond2 = nn.Sequential(*[TransformerBlock(dim=embed_dims[1], num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(3)])

		self.downs_cond2_3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])  ## From Level 2 to Level 3
		self.layers_cond3 = nn.Sequential(*[TransformerBlock(dim=embed_dims[2], num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(4)])

		self.avg_pool_cond4 = nn.AdaptiveAvgPool2d(1)

		self.layers_cond5 = nn.Sequential(nn.Conv2d(embed_dims[2], embed_dims[2] // 4, 1, padding=0, bias=False),
										  nn.Sigmoid(),
										  nn.Conv2d(embed_dims[2] // 4, 2, 1, padding=0, bias=False),
										  nn.Sigmoid())

		self.downs_cond = nn.ModuleList()
		self.ups = nn.ModuleList()

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.downs_cond.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			# self.fusions.append(fusion_layer(embed_dims[i]))
			self.fusions.append(nn.Conv2d(embed_dims[i] * 2, embed_dims[i], 1, bias=False))
			self.fusions_cond.append(SFTLayer(in_nc=embed_dims[i], out_nc=embed_dims[i], nf=embed_dims[i] // 2))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)

	def forward(self, x):
		feat = self.inconv(x)
		feat_cond = self.condconv(x)

		skip_cond = []
		cond_enc_level1 = self.layers_cond1(feat_cond)
		skip_cond.append(cond_enc_level1)

		cond_enc_level2 = self.downs_cond1_2(cond_enc_level1)
		cond_enc_level2 = self.layers_cond2(cond_enc_level2)
		skip_cond.append(cond_enc_level2)

		cond_enc_level3 = self.downs_cond2_3(cond_enc_level2)
		cond_enc_level3 = self.layers_cond3(cond_enc_level3)
		skip_cond.append(cond_enc_level3)

		cond_level4 = self.avg_pool_cond4(cond_enc_level3)
		cond = self.layers_cond5(cond_level4)

		skips = []

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			# feat_cond = self.layers_cond[i](feat_cond)
			feat_cond = skip_cond[i]
			# feat = self.fusions_cond[i]([feat, feat_cond])
			feat = self.fusions_cond[i]((feat, feat_cond))
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)
			#feat_cond = self.downs_cond[i](feat_cond)

		feat = self.layers[self.half_num](feat)

		for i in range(self.half_num-1, -1, -1):
			feat = self.ups[i](feat)
			# feat = self.fusions[i]([feat, skips[i]])
			feat = self.fusions[i](torch.cat((feat, skips[i]), dim=1))
			feat = self.layers[self.stage_num-i-1](feat)

		x = self.outconv(feat) + x

		return x, cond


__all__ = ['SRUDC', 'SRUDC_f', 'SRUDC_l']

def SRUDC_f():
	return SRUDC(kernel_size=5, base_dim=24, depths=[16, 16, 16, 32, 16, 16, 16], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def SRUDC_l():
	return SRUDC(kernel_size=5, base_dim=16, depths=[16, 16, 16, 32, 16, 16, 16], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
