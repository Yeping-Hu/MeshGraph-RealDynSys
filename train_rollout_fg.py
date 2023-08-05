import numpy as np
import pickle
import torch
import torch.optim as optim
import time

from core.args import ArgParser
from flow_model.dynamic_predictor import FullGraphPredictor
from flow_model.node_info import CylinderNodeType
from mesh_graph_modules.core import MeshGraphNet
from core.dataset import load_cfd_traj
from core.metrics import ScoreMeter, Recorder


def train(args, model, traj_data, save=False):
	recorder = Recorder(['train_loss', 'train_root_mse', 'train_re_mse'])
	score_meter = ScoreMeter()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(args.lr_decay_extent) / args.lr_decay_len))
	if args.start_step > 0:
		ckpt = torch.load(f'{args.checkpoint_dir}/step{args.start_step}.pt')
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		permutation = ckpt['permutation']
		rand_permute_t = ckpt['rand_permute_t']
	start = time.time()
	n_traj, n_frames = len(traj_data), traj_data[0]['velocity'].shape[0]
	print(f"n_traj: {n_traj}, n_frames: {n_frames}")
	for i in range(args.start_step, args.start_step + args.n_train_steps):
		if i % n_traj == 0:
			permutation = np.random.permutation(n_traj)
		traj_id = permutation[i % n_traj]
		if i % (n_traj * n_frames) == 0:
			rand_permute_t = np.vstack([np.random.permutation(n_frames-1) for _ in range(n_traj)])
		t = rand_permute_t[traj_id, (i // n_traj) % (n_frames-1)]
		inputs = traj_data[traj_id].copy()
		inputs['velocity'] = traj_data[traj_id]['velocity'][t, :, :].copy()
		inputs['target_velocity'] = traj_data[traj_id]['velocity'][t+1, :, :]
		train_loss, _, ro_mse, re_mse = model(inputs, is_training=True, noise=args.noise)
		score_meter.update(train_loss.item(), ro_mse, re_mse)

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()
		if i in range(args.lr_decay_start, args.lr_decay_start + args.lr_decay_len):
			scheduler.step()

		if (i+1) % args.print_interval == 0:
			print(f"step {i} | {score_meter.stats_string()} | "
			      f"lr {optimizer.param_groups[0]['lr']:.2e} | time {time.time() - start:.2f}",
			      flush=True)
			score_meter.reset()
		recorder.update([train_loss.item(), ro_mse, re_mse])

		if save and (i+1) % args.save_interval == 0:
			torch.save({
				'step': i,
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'permutation': permutation,
				'rand_permute_t': rand_permute_t,
			}, f'{args.checkpoint_dir}/step{i+1}.pt')

	if save:
		with open(args.train_record_path, 'wb') as f:
			pickle.dump(recorder.record, f)


@torch.no_grad()
def test_rollout(args, model, traj_data, full=False):
	checkpoint_path = f"{args.checkpoint_dir}/step{args.n_train_steps}.pt"
	model.load_state_dict(torch.load(checkpoint_path)['model'])
	score_meter = ScoreMeter()
	all_traj_rollout = []
	n_traj, n_frames = len(traj_data), traj_data[0]['velocity'].shape[0]
	print(f"n_traj: {n_traj}, n_frames: {n_frames}")
	if full:
		save_path = f'{args.checkpoint_dir}/pred_{n_frames}steps.pkl'
	else:
		save_path = f'{args.checkpoint_dir}/pred_1step.pkl'
	for traj_id in range(n_traj):
		traj_rollout = {
			'velocity': traj_data[traj_id]['velocity'].copy(),
			'mesh_pos': traj_data[traj_id]['mesh_pos'],
			'node_type': traj_data[traj_id]['node_type'],
			'cells': traj_data[traj_id]['cells'],
		}
		mask_idx = np.append(np.where(traj_rollout['node_type'] == CylinderNodeType.NORMAL)[0],
		                     np.where(traj_rollout['node_type'] == CylinderNodeType.OUTFLOW)[0])
		for t in range(n_frames-1):
			inputs = traj_data[traj_id].copy()
			inputs['velocity'] = traj_data[traj_id]['velocity'][t, :, :]
			inputs['target_velocity'] = traj_data[traj_id]['velocity'][t+1, :, :]
			loss, v_pred, ro_mse, re_mse = model(inputs, is_training=False, noise=None)
			score_meter.update(loss.item(), ro_mse, re_mse)
			traj_rollout['velocity'][t+1, mask_idx, :] = v_pred[mask_idx, :]
			if full:
				traj_data[traj_id]['velocity'][t+1, mask_idx, :] = v_pred[mask_idx, :]
		all_traj_rollout.append(traj_rollout)

		if (traj_id+1) % 10 == 0:
			print(f"test traj {traj_id+1} | {score_meter.stats_string()}",
			      flush=True)
			with open(save_path, 'wb') as f:
				pickle.dump(all_traj_rollout, f)


if __name__ == '__main__':
	arg_parser = ArgParser(task='cylinder_rollout')
	args = arg_parser.parse_args(verbose=True)
	train_traj_data, test_traj_data = load_cfd_traj(args.timestep_end, args.timestep_len)
	learned_model = MeshGraphNet(node_dim=6, edge_dim=3, h_dim=args.h_dim, out_dim=2, n_blocks=args.n_gnn_blocks)
	model = FullGraphPredictor(learned_model, node_dim=6, device=args.device).to(args.device)
	train(args, model, train_traj_data, save=True)
	test_rollout(args, model, test_traj_data, full=False)
	test_rollout(args, model, test_traj_data, full=True)