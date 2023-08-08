import os
import argparse
import torch
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from DISTS_pytorch import DISTS
import lpips

from utils import AverageMeter, write_img, chw_to_hwc, pad_img
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SRUDC_f', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--test_set', default='test', type=str, help='test dataset name')
parser.add_argument('--exp', default='UDC', type=str, help='experiment setting')
args = parser.parse_args()


def SSIM_def(img1, img2):
	ssim_val = ssim(img1, img2, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
	return ssim_val


def PSNR_def(img1, img2):
	mse = np.mean((img1 - img2) ** 2)
	if mse == 0:
		return 100
	PIXEL_MAX = 1
	return 10 * math.log10(PIXEL_MAX / mse)

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	eDISTS = AverageMeter()
	eLPIPS = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	loss_fn_alex = lpips.LPIPS(net='alex').cuda()  # best forward scores
	dists = DISTS().cuda()

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			H, W = input.shape[2:]
			input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
			output, _ = network(input)
			output = output.clamp_(-1, 1)
			output = output[:, :, :H, :W]

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			# for DISTS [0, 1]
			dists_cur = dists(target, output)
			eDISTS.update(dists_cur)
			# for LIPIPS [-1, 1]
			x_tensor = target * 2.0 - 1
			output_tensor = output * 2.0 - 1
			lpips_cur = loss_fn_alex(x_tensor, output_tensor)
			eLPIPS.update(lpips_cur)

			# psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()
			#
			# _, _, H, W = output.size()
			# down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			# ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
			# 				F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
			# 				data_range=1, size_average=False).item()
		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		tar_img = chw_to_hwc(target.detach().cpu().squeeze(0).numpy())

		psnr_val = PSNR_def(tar_img, out_img)
		ssim_val = SSIM_def(tar_img, out_img)

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))
		print('LPIPS : %f, DISTS : %f'%(lpips_cur, dists_cur))

		f_result.write('%s, %.02f, %.03f, %f, %f\n'%(filename, psnr_val, ssim_val, lpips_cur, dists_cur))

		# out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()
	print('avgPSNR : %.02f, avgSSIM : %.03f, avgLPIPS : %f, avgDISTS : %f' % (PSNR.avg, SSIM.avg, eLPIPS.avg, eDISTS.avg))

	os.rename(os.path.join(result_dir, 'results.csv'), 
			  os.path.join(result_dir, '%.03f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


def main():
	network = eval(args.model)()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.test_set)
	test_dataset = PairLoader(dataset_dir, 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.test_set, args.exp, args.model)
	test(test_loader, network, result_dir)


if __name__ == '__main__':
	main()
