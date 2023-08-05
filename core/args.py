from argparse import ArgumentParser, Namespace
import yaml
import os
from os.path import splitext
import torch
import random
import numpy as np
import sys


class ArgParser:
	def __init__(self, task):
		parser = ArgumentParser()
		parser.add_argument("--config", default='default.yaml')
		parser.add_argument("--start_step", type=int, default=0, help="resume training from this step")
		parser.add_argument("--gpu_id", type=int, default=0)
		parser.add_argument("--seed", type=int, default=42, help="random seed")
		parser.add_argument("--out", default=None, help="redirect stdout to the specified file")
		self.parser = parser
		self.task = task

	def parse_args(self, verbose=False, use_random_seed=True):
		args = self.parser.parse_args()
		args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

		config_path = f"./configs/{self.task}/{args.config}"
		config = yaml.safe_load(open(f"{config_path}", 'r'))
		args = vars(args)
		args.update(config)
		args = Namespace(**args)
		args.checkpoint_dir = f"./saved_model/{self.task}/{splitext(os.path.basename(args.config))[0]}"
		os.makedirs(args.checkpoint_dir, exist_ok=True)
		args.train_record_path = f"{args.checkpoint_dir}/train_record.pkl"
		args.test_result_path = f"{args.checkpoint_dir}/test_result.pkl"
		if args.out is not None:
			sys.stdout = open(f"{args.checkpoint_dir}/{args.out}", 'w')
			sys.stderr = open(f"{args.checkpoint_dir}/{args.out}", 'w')

		if use_random_seed:
			random.seed(args.seed)
			np.random.seed(args.seed)
			torch.manual_seed(args.seed)

		if verbose:
			self.print_args(args)
		return args

	@staticmethod
	def print_args(args):
		print(f"Configurations\n{'=' * 50}", flush=True)
		[print(k, ':', v, flush=True) for k, v in vars(args).items()]
		print('=' * 50, flush=True)


if __name__ == '__main__':
	sys.argv.extend(['--config', 'default.yaml'])
	arg_parser = ArgParser(task='cavity')
	args = arg_parser.parse_args(verbose=True)
