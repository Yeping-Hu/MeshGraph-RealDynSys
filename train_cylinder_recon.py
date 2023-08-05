import numpy as np
import pickle
import torch
import torch.optim as optim
import time
import os
from scipy.spatial import Delaunay

from core.args import ArgParser
from flow_model.sg_autoencoder import CylinderAutoencoder
from mesh_graph_modules import get_selector, get_reverser
from core.dataset import load_cfd_traj
from core.metrics import ScoreMeter, Recorder


def train(args, model, traj_data):
	recorder = Recorder(['train_loss', 'train_root_mse', 'train_re_mse'])
	score_meter = ScoreMeter()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.ExponentialLR(
		optimizer, gamma=np.exp(np.log(args.lr_decay_extent) / args.lr_decay_len))
	if args.start_step > 0:
		ckpt = torch.load(f'{args.checkpoint_dir}/step{args.start_step}.pt')
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		permutation = ckpt['permutation']
		rand_permute_t = ckpt['rand_permute_t']
	start = time.time()
	n_traj, n_frames = len(traj_data), traj_data[0]['velocity'].shape[0]
	print(f"n_traj: {n_traj}, n_frames: {n_frames}")
	all_velocities = [traj_data[i]['velocity'] for i in range(n_traj)]
	for i in range(args.start_step, args.start_step + args.n_train_steps):
		if i % n_traj == 0:
			permutation = np.random.permutation(n_traj)
		traj_id = permutation[i % n_traj]
		if i % (n_traj * n_frames) == 0:
			rand_permute_t = np.vstack([np.random.permutation(n_frames) for _ in range(n_traj)])
		t = rand_permute_t[traj_id, (i // n_traj) % n_frames]
		inputs = traj_data[traj_id]
		inputs['velocity'] = all_velocities[traj_id][t, :, :]
		train_loss, _, ro_mse, re_mse = model(inputs, is_training=True, noise=args.noise)
		score_meter.update(train_loss.item(), ro_mse, re_mse)

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()
		if i in range(args.lr_decay_start, args.lr_decay_start + args.lr_decay_len):
			scheduler.step()

		if (i+1) % args.print_interval == 0:
			print(f"iter {i+1} | {score_meter.stats_string()} | "
			      f"lr {optimizer.param_groups[0]['lr']:.2e} | time {time.time() - start:.2f}",
			      flush=True)
			score_meter.reset()
		recorder.update([train_loss.item(), ro_mse, re_mse])

		if (i+1) % args.save_interval == 0:
			torch.save({
				'step': i,
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'permutation': permutation,
				'rand_permute_t': rand_permute_t,
			}, f'{args.checkpoint_dir}/iter{i+1}.pt')

	with open(args.train_record_path, 'wb') as f:
		pickle.dump(recorder.record, f)


@torch.no_grad()
def test(args, model, traj_data):
	checkpoint_path = f'{args.checkpoint_dir}/iter{args.n_train_steps}.pt'
	model.load_state_dict(torch.load(checkpoint_path, map_location=args.device)['model'])
	score_meter = ScoreMeter()
	all_traj_recon = []
	n_traj, n_frames = len(traj_data), traj_data[0]['velocity'].shape[0]
	print(f"n_traj: {n_traj}, n_frames: {n_frames}")
	all_velocities = [traj_data[i]['velocity'] for i in range(n_traj)]
	for traj_id in range(n_traj):
		traj_recon = {
			'velocity': np.zeros_like(traj_data[traj_id]['velocity']),
			'mesh_pos': traj_data[traj_id]['mesh_pos'],
			'node_type': traj_data[traj_id]['node_type'],
			'cells': traj_data[traj_id]['cells'],
			'sample_idx': []
		}
		for t in range(n_frames):
			inputs = traj_data[traj_id]
			inputs['velocity'] = all_velocities[traj_id][t, :, :]
			loss, v_recon, ro_mse, re_mse = model(inputs, is_training=False)
			score_meter.update(loss.item(), ro_mse, re_mse)
			traj_recon['velocity'][t, :, :] = v_recon
			traj_recon['sample_idx'].append(model.selector.samples)
		all_traj_recon.append(traj_recon)

		if (traj_id+1) % 10 == 0:
			print(f"test traj {traj_id+1} | {score_meter.stats_string()}",
			      flush=True)
	with open(f'{args.checkpoint_dir}/test_v_recon.pkl', 'wb') as f:
		pickle.dump(all_traj_recon, f)


@torch.no_grad()
def save_z(args, model, dynamic_sg=True, test_only=False):
	train_traj_data, test_traj_data = load_cfd_traj(args.timestep_end, args.timestep_len, args.ood)
	checkpoint_path = f'{args.checkpoint_dir}/iter{args.n_train_steps}.pt'
	model.load_state_dict(torch.load(checkpoint_path, map_location=args.device)['model'])
	for stage, traj_data in zip(['train', 'test'], [train_traj_data, test_traj_data]):
		if test_only and stage == 'train':
			continue
		all_traj = []
		n_traj, n_frames = len(traj_data), traj_data[0]['velocity'].shape[0]
		print(f"{stage} n_traj: {n_traj}, n_frames: {n_frames}")
		all_velocities = [traj_data[i]['velocity'] for i in range(n_traj)]
		start = time.time()
		for traj_id in range(n_traj):
			try:
				n_samples = args.n_samples
			except:
				n_samples = int(args.sampler_ratio * all_velocities[traj_id].shape[1])
			traj = {
				'z': np.zeros((n_frames, n_samples, args.z_dim)),
				'z_target': np.zeros((n_frames, n_samples, args.z_dim)),
				'sample_idx': np.zeros((n_frames, n_samples), dtype=int),
				'fixed_idx': None,
				'mesh_pos': traj_data[traj_id]['mesh_pos'],
				'node_type': traj_data[traj_id]['node_type'],
				'cells': traj_data[traj_id]['cells']
			}
			for t in range(n_frames):
				inputs = traj_data[traj_id]
				inputs['velocity'] = all_velocities[traj_id][t, :, :]
				z, sample_idx = model(inputs, is_training=False, get_z=True)
				traj['z_target'][t-1, :, :] = z[traj['sample_idx'][t-1, :], :]
				traj['z'][t, :, :] = z[sample_idx, :]
				traj['sample_idx'][t, :] = sample_idx
			if dynamic_sg:
				traj['fixed_idx'] = inputs['fixed_idx']
			all_traj.append(traj)

			if (traj_id+1) % 10 == 0:
				print(f"{stage} traj {traj_id+1} | time {time.time() - start:.2f}", flush=True)
		with open(f'{args.checkpoint_dir}/{stage}_z.pkl', 'wb') as f:
			pickle.dump(all_traj, f)


def connect_subgraph(args, dynamic_sg=False, test_only=False):
	def mesh_graph_connect(mesh_pos, subgraph_idx):
		"""Connect the subgraph nodes with a delaunay triangulation
		"""
		subgraph_pos = mesh_pos[subgraph_idx, :]
		triang = Delaunay(subgraph_pos)
		subgraph_cells = triang.simplices
		return subgraph_pos, subgraph_cells

	for stage in ['train', 'test']:
		if test_only and stage == 'train':
			continue
		with open(f"{args.checkpoint_dir}/{stage}_z.pkl", 'rb') as f:
			z_all = pickle.load(f)
		for z in z_all:
			pos = z['mesh_pos']
			if not dynamic_sg:
				subgraph_idx = z['sample_idx'][0, :].astype(int)
				z['subgraph_pos'], z['subgraph_cells'] = mesh_graph_connect(pos, subgraph_idx)
			else:
				z['subgraph_pos'] = np.zeros((*(z['sample_idx'].shape), 2))
				z['subgraph_cells'] = []
				for t in range(z['sample_idx'].shape[0]):
					subgraph_idx = z['sample_idx'][t, :]
					z['subgraph_pos'][t], subgraph_cells = mesh_graph_connect(pos, subgraph_idx)
					z['subgraph_cells'].append(subgraph_cells)
		os.makedirs(f"{args.checkpoint_dir}/", exist_ok=True)
		with open(f"{args.checkpoint_dir}/{stage}_z_connected.pkl", 'wb') as f:
			pickle.dump(z_all, f)


def build_model(args):
	selector = get_selector(args)
	reverser = get_reverser(args)
	model = CylinderAutoencoder(
		selector,
		reverser,
		node_dim=6,
		edge_dim=3,
		h_dim=args.h_dim,
		z_dim=args.z_dim,
		out_dim=2,
		n_blocks=args.n_gnn_blocks,
		device=args.device).to(args.device)
	return model


if __name__ == '__main__':
	arg_parser = ArgParser(task='cylinder_recon')
	arg_parser.parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'save_z', 'connect'])
	arg_parser.parser.add_argument('--ood', action='store_true')
	arg_parser.parser.add_argument('--debug', action='store_true')
	args = arg_parser.parse_args(verbose=True)

	train_traj_data, test_traj_data = load_cfd_traj(args.timestep_end, args.timestep_len, args.ood)
	model = build_model(args)
	if args.mode == 'train':
		train(args, model, train_traj_data)
		test(args, model, test_traj_data)
	elif args.mode == 'test':
		test(args, model, test_traj_data)
	elif args.mode == 'save_z':
		save_z(args, model, dynamic_sg=args.dynamic_subgraph, test_only=args.ood)
	elif args.mode == 'connect':
		connect_subgraph(args, dynamic_sg=args.dynamic_subgraph, test_only=args.ood)
