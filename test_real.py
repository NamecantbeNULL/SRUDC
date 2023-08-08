import os
import cv2
import argparse
import torch
from utils import hwc_to_chw, read_img
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import AverageMeter, write_img, chw_to_hwc, pad_img
# from datasets.loader import PairLoader, SingleLoader
from torch.utils.data import Dataset
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_d', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='/data/doublebin/data_mixed/test/LQ_real_UAV', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--test_set', default='real', type=str, help='test dataset name')
parser.add_argument('--exp', default='reside-in', type=str, help='experiment setting')
args = parser.parse_args()


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'source': hwc_to_chw(img), 'filename': img_name}

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		filename = batch['filename'][0]

		with torch.no_grad():
			H, W = input.shape[2:]
			input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
			output, _ = network(input)
			output = output.clamp_(-1, 1)
			output = output[:, :, :H, :W]

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())

		print('Test: [{0}]'.format(idx))

		# out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)


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

	# dataset_dir = os.path.join(args.data_dir, args.test_set)
	dataset_dir = args.data_dir
	# test_dataset = PairLoader(dataset_dir, 'test')
	test_dataset = SingleLoader(dataset_dir)
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.test_set, args.exp, args.model)
	test(test_loader, network, result_dir)


if __name__ == '__main__':
	main()
