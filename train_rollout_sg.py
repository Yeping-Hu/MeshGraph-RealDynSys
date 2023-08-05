import numpy as np
from scipy.spatial import Delaunay
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
from argparse import Namespace

from core.args import ArgParser
from core.dataset import load_cfd_traj
from core.metrics import ScoreMeter
from flow_model.node_info import CylinderNodeType
from train_cylinder_recon import build_model
from flow_model.dynamic_predictor import SubgraphPredictor


def train_single_step(args, model, dynamic_sg=False):
	with open(f"{args.autoencoder.checkpoint_dir}/train_z_connected.pkl", 'rb') as f:
		gt_z = pickle.load(f)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(args.lr_decay_extent) / args.lr_decay_len))
	n_traj, n_frames = len(gt_z), gt_z[0]['z'].shape[0]
	print(f"n_traj: {n_traj}, n_frames: {n_frames}")

	time_start = time.time()
	running_loss, running_rmse, counter = 0, 0, 0
	for i in range(0, args.n_train_steps):
		if i % n_traj == 0:
			permutation = np.random.permutation(n_traj)
		traj_id = permutation[i % n_traj]
		if i % (n_traj * n_frames) == 0:
			rand_permute_t = np.vstack([np.random.permutation(n_frames-1)+1 for _ in range(n_traj)])
		t = rand_permute_t[traj_id, (i // n_traj) % (n_frames-1)]
		z_target = gt_z[traj_id]['z_target'][t-1, :, :] if dynamic_sg else gt_z[traj_id]['z'][t, :, :]
		z_target = torch.Tensor(z_target).to(args.device)
		z_input = torch.Tensor(gt_z[traj_id]['z'][t-1:t, :, :]).to(args.device)
		if args.noise is not None:
			z_input += args.noise * torch.randn_like(z_target)

		pos = gt_z[traj_id]['subgraph_pos'][t-1] if dynamic_sg else gt_z[traj_id]['subgraph_pos']
		cells = gt_z[traj_id]['subgraph_cells'][t-1] if dynamic_sg else gt_z[traj_id]['subgraph_cells']
		_, loss, rmse = model(z_input, z_target, pos, cells, is_training=True)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		running_rmse += rmse
		counter += 1
		if args.lr_decay_start <= i < args.lr_decay_start + args.lr_decay_len:
			scheduler.step()

		if (i+1) % args.print_interval == 0:
			print(f"step {i} | loss {running_loss / counter:.5f} | rmse {running_rmse / counter:.5f} | "
			      f"lr {optimizer.param_groups[0]['lr']:.2e} | "
			      f"time {time.time() - time_start:.2f}",
			      flush=True)
			running_loss, running_rmse, counter = 0, 0, 0

		if (i+1) % args.save_interval == 0:
			torch.save(model.state_dict(), f'{args.checkpoint_dir}/iter{i+1}.pt')


@torch.no_grad()
def test(args, model, full=False, dynamic_sg=False):
	with open(f'{args.autoencoder.checkpoint_dir}/test_z_connected.pkl', 'rb') as f:
		gt_z = pickle.load(f)
	n_traj, n_frames = len(gt_z), gt_z[0]['z'].shape[0]
	print(f"n_traj: {n_traj}, n_frames: {n_frames}")
	pred_z = [{'z': gt_z[j]['z'][:1, :, :],
	           'sample_idx': gt_z[j]['sample_idx']} for j in range(n_traj)]
	model.load_state_dict(torch.load(f'{args.checkpoint_dir}/iter{args.n_train_steps}.pt'))
	model.eval()
	running_loss, running_rmse, counter = 0, 0, 0
	if full:
		save_path = f'{args.checkpoint_dir}/test_z_pred_{n_frames}steps.pkl'
	else:
		save_path = f'{args.checkpoint_dir}/test_z_pred_1step.pkl'
	for t in range(1, n_frames):
		for i in range(0, n_traj):
			z_target = gt_z[i]['z_target'][t-1, :, :] if dynamic_sg else gt_z[i]['z'][t, :, :]
			z_target = torch.Tensor(z_target).to(args.device)
			z_input = gt_z[i]['z'][:t, :, :] if dynamic_sg else pred_z[i]['z']
			z_input = torch.Tensor(z_input).to(args.device)
			pos = gt_z[i]['subgraph_pos'][t-1] if dynamic_sg else gt_z[i]['subgraph_pos']
			cells = gt_z[i]['subgraph_cells'][t-1] if dynamic_sg else gt_z[i]['subgraph_cells']
			out, loss, rmse = model(z_input, z_target, pos, cells, is_training=False)
			running_loss += loss.item()
			running_rmse += rmse
			counter += 1
			if full:
				pred_z[i]['z'] = np.vstack([pred_z[i]['z'], out.detach().cpu().numpy()[None, :, :]])
			else:
				pred_z[i]['z'] = np.vstack([pred_z[i]['z'], gt_z[i]['z'][t:t+1, :, :]])
		print(f"test time step {t} | loss {running_loss / counter:.5f} | rmse {running_rmse / counter:.5f} | ",
		      flush=True)
	with open(save_path, 'wb') as f:
		pickle.dump(pred_z, f)


@torch.no_grad()
def test_predz_decode_encode(args, model, dynamic_sg=False):
	"""Rollout is produced by z prediction followed by decode and encode."""
	ae_args = args.autoencoder
	ae_args.device = args.device
	ae_model = build_model(ae_args)
	ae_model.load_state_dict(torch.load(f'{ae_args.checkpoint_dir}/iter{ae_args.n_train_steps}.pt',
	                                    map_location=args.device)['model'])

	with open(f'{args.autoencoder.checkpoint_dir}/test_z_connected.pkl', 'rb') as f:
		gt_z = pickle.load(f)
		_, traj_data = load_cfd_traj(ae_args.timestep_end, ae_args.timestep_len, args.ood)
	criterion = nn.MSELoss().to(args.device)
	n_traj, n_frames = len(gt_z), gt_z[0]['z'].shape[0]
	print(f"n_traj: {n_traj}, n_frames: {n_frames}")
	pred_z = [{'z': gt_z[j]['z'][:1, :, :],
	           'sample_idx': gt_z[j]['sample_idx'].copy(),
	           'fixed_idx': gt_z[j]['fixed_idx'],
	           'velocity': traj_data[j]['velocity'],
	           'mesh_pos': traj_data[j]['mesh_pos'],
	           'node_type': traj_data[j]['node_type'],
	           'cells': traj_data[j]['cells']} for j in range(n_traj)]
	model.load_state_dict(torch.load(f'{args.checkpoint_dir}/iter{args.n_train_steps}.pt', map_location=args.device))
	model.eval()
	score_meter = ScoreMeter()
	v_buffer = [traj_data[i]['velocity'].copy() for i in range(n_traj)]
	mask_ids = [np.append(np.where(traj_data[i]['node_type'] == CylinderNodeType.NORMAL)[0],
	                      np.where(traj_data[i]['node_type'] == CylinderNodeType.OUTFLOW)[0]) for i in range(n_traj)]
	for t in range(1, n_frames):
		for i in range(n_traj):
			z_target = gt_z[i]['z'][t, :, :]
			z_target = torch.Tensor(z_target).to(args.device)
			z_input = torch.Tensor(pred_z[i]['z']).to(args.device)
			if dynamic_sg:
				pos = pred_z[i]['mesh_pos'][pred_z[i]['sample_idx'][t-1, :], :] # subgraph nodes position at t-1
				delaunay = Delaunay(pos)
				cells = delaunay.simplices
			else:
				pos = gt_z[i]['subgraph_pos']
				cells = gt_z[i]['subgraph_cells']
			out, _, _ = model(z_input, z_target, pos, cells, is_training=False) # predicted z_Sp{t-1} at t
			out = out.cpu().numpy()
			# decode to velocity field
			inputs = traj_data[i]
			inputs['velocity'] = v_buffer[i][t, :, :].copy()
			_, v_recon, ro_mse, re_mse = ae_model.decode(inputs, z=out, sample_idx=pred_z[i]['sample_idx'][t-1, :]) # reconstructed Vt

			# encode to z
			inputs['velocity'][mask_ids[i], :] = v_recon[mask_ids[i], :]
			inputs['fixed_idx'] = pred_z[i]['fixed_idx']
			input_sample_idx = None if dynamic_sg else gt_z[i]['sample_idx'][t, :]
			# get test sample index for next time step
			z_out, sample_idx = ae_model(inputs, is_training=False, get_z=True, input_sample_idx=input_sample_idx) # latent z and computed sample_idx Sp{t}
			pred_z[i]['sample_idx'][t, :] = sample_idx
			# get z from ground truth sample index for loss computation
			z_out_gt = torch.Tensor(z_out[gt_z[i]['sample_idx'][t, :], :]).to(args.device) # predicted z_Sgt{t}
			loss = criterion(z_out_gt, z_target)
			score_meter.update(loss.item(), ro_mse, re_mse)
			pred_z[i]['z'] = np.vstack([pred_z[i]['z'], z_out[None, sample_idx, :]]) # update z_Sp{t}
			pred_z[i]['velocity'][t] = v_recon

		print(f"test time step {t} | {score_meter.stats_string()}",
		      flush=True)

	with open(f'{args.checkpoint_dir}/test_v_pred.pkl', 'wb') as f:
		pickle.dump(pred_z, f)


if __name__ == '__main__':
	arg_parser = ArgParser(task='cylinder_sg')
	arg_parser.parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	arg_parser.parser.add_argument('--ood', action='store_true')
	args = arg_parser.parse_args(verbose=True)
	args.autoencoder = Namespace(**args.autoencoder)

	model = SubgraphPredictor(
		z_dim=args.z_dim,
		edge_dim=3,
		h_dim=args.h_dim,
		n_blocks=args.n_gnn_blocks,
		device=args.device).to(args.device)

	if args.mode == 'train':
		train_single_step(args, model, dynamic_sg=args.dynamic_subgraph)
	elif args.mode == 'test':
		test(args, model, full=False, dynamic_sg=args.dynamic_subgraph)
		test(args, model, full=True, dynamic_sg=args.dynamic_subgraph)
		test_predz_decode_encode(args, model, dynamic_sg=args.dynamic_subgraph)
