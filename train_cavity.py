import numpy as np
import pickle
import torch
import torch.optim as optim
import random
import time

from core.args import ArgParser
from flow_model.sg_autoencoder import CavityAutoencoder
from mesh_graph_modules import get_selector, get_reverser
from core.dataset import load_cavity_data
from core.metrics import ScoreMeter, Recorder


def train(args, model, dataloader, save=False):
	recorder = Recorder(['train_loss', 'train_root_mse', 'train_re_mse'])
	score_meter = ScoreMeter()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.ExponentialLR(
		optimizer, gamma=np.exp(np.log(args.lr_decay_extent) / args.lr_decay_len))
	start = time.time()
	for i in range(args.n_train_steps):
		if i % len(dataloader) == 0:
			random.shuffle(dataloader)
		inputs = dataloader[i % len(dataloader)]
		train_loss, _, ro_mse, re_mse = model(inputs, is_training=True)
		score_meter.update(train_loss.item(), ro_mse, re_mse)

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()
		if i >= args.lr_decay_start:
			scheduler.step()

		if (i+1) % args.print_interval == 0:
			print(f"iter {i+1} | {score_meter.stats_string()} | "
			      f"lr {optimizer.param_groups[0]['lr']:.2e} | time {time.time() - start:.2f}",
			      flush=True)
			score_meter.reset()
		recorder.update([train_loss.item(), ro_mse, re_mse])

		if save and (i+1) % args.save_interval == 0:
			torch.save(model.state_dict(), f'{args.checkpoint_dir}/iter{i+1}.pt')

	if save:
		with open(args.train_record_path, 'wb') as f:
			pickle.dump(recorder.record, f)


@torch.no_grad()
def test(args, model, dataloader):
	checkpoint_path = f'{args.checkpoint_dir}/iter{args.n_train_steps}.pt'
	model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
	score_meter = ScoreMeter()
	all_recon = []
	for idx in range(len(dataloader)):
		loss, recon, ro_mse, re_mse = model(dataloader[idx], is_training=False)
		score_meter.update(loss.item(), ro_mse, re_mse)
		all_recon.append(recon)
	print(f"test | {score_meter.stats_string()}", flush=True)
	with open(f'{args.checkpoint_dir}/test_v_recon.pkl', 'wb') as f:
		pickle.dump(all_recon, f)


if __name__ == '__main__':
	arg_parser = ArgParser(task='cavity')
	args = arg_parser.parse_args(verbose=True)
	train_loader, test_loader = load_cavity_data(args.partial_data, args.cavity_depth_threshold)

	selector = get_selector(args)
	reverser = get_reverser(args)
	model = CavityAutoencoder(
		selector,
		reverser,
		node_dim=5,
		edge_dim=3,
		h_dim=args.h_dim,
		z_dim=args.z_dim,
		out_dim=2,
		n_blocks=args.n_gnn_blocks,
		device=args.device).to(args.device)
	train(args, model, train_loader, save=True)
	test(args, model, test_loader)
