import math
import numpy as np
import pickle
from scipy.interpolate import griddata
import scipy.spatial

from core.dataset import load_cavity_data, load_cfd_traj


class ScoreMeter:
	def __init__(self):
		self.loss_sum = 0
		self.ro_mse_sum = np.zeros(2)
		self.re_mse_sum = np.zeros(2)
		self.count = 0

	def update(self, loss, ro_mse, re_mse):
		self.loss_sum += loss
		self.ro_mse_sum += ro_mse
		self.re_mse_sum += re_mse
		self.count += 1

	def reset(self):
		self.__init__()

	def stats(self):
		loss_avg = self.loss_sum / self.count
		ro_mse_avg = self.ro_mse_sum / self.count
		re_mse_avg = self.re_mse_sum / self.count
		return loss_avg, ro_mse_avg, re_mse_avg

	def stats_string(self):
		loss_avg, ro_mse_avg, re_mse_avg = self.stats()
		return f"loss {loss_avg:.5f} | " \
			   f"root_mse_u {ro_mse_avg[0]:.5f} | root_mse_v {ro_mse_avg[1]:.5f} | " \
			   f"re_mse_u {re_mse_avg[0]:.5f} | re_mse_v {re_mse_avg[1]:.5f}"


class Recorder(object):
	def __init__(self, names):
		self.names = names
		self.record = {}
		for name in self.names:
			self.record[name] = []

	def update(self, vals):
		for name, val in zip(self.names, vals):
			self.record[name].append(val)

class VortexScoreMeter:
	def __init__(self, thres):
		self.sum_total_distance = 0
		self.sum_detected_distance = 0
		self.n_detected = 0
		self.n_total_vortex = 0
		self.thres = thres

	def update(self, gt_vortex_pos, pred_vortex_pos):
		distance = scipy.spatial.distance_matrix(gt_vortex_pos, pred_vortex_pos)
		self.n_total_vortex += distance.shape[0]
		self.sum_total_distance += distance.sum()
		detected_idx = distance < self.thres
		n_detected = np.sum(detected_idx)
		self.n_detected += n_detected
		if n_detected > 0:
			self.sum_detected_distance += np.sum(distance[detected_idx])

	def print_score(self):
		self.total_mean = self.sum_total_distance / self.n_total_vortex
		self.mddv = self.sum_detected_distance / self.n_detected
		self.vdr = self.n_detected / self.n_total_vortex
		print(f"total mean: {self.total_mean}")
		print(f"mean detected distance: {self.mddv}")
		print(f"vortex detection rate: {self.vdr}")


def find_vortex(v, pos, w, h, ngridx=1000, ngridy=1000):
	"""Find the position of vortices in a flow field."""
	xi = np.linspace(0, w, ngridx)
	yi = np.linspace(0, h, ngridy)
	x_grid, y_grid = np.meshgrid(xi, yi)
	vx_grid = griddata(pos, v[:, 0], (x_grid, y_grid), method='linear')
	vy_grid = griddata(pos, v[:, 1], (x_grid, y_grid), method='linear')
	vortex_pos = find_vortex_in_grid(x_grid, y_grid, vx_grid, vy_grid)
	return vortex_pos

def find_vortex_in_grid(x_grid, y_grid, vx_grid, vy_grid):
	"""
	Get the vortex coordinates from the velocity field in a MxN grid.
	:param x_grid: the x coordinates of the grid with shape (M, N)
	:param y_grid: the y coordinates of the grid with shape (M, N)
	:param vx_grid: the x velocity of the grid with shape (M, N)
	:param vy_grid: the y velocity of the grid with shape (M, N)
	:return: the vortex coordinates
	"""
	ngridx = vx_grid.shape[1]
	ngridy = vx_grid.shape[0]

	vortex_pos = []
	for row in range(ngridy):
		for col in range(ngridx):
			if row == 0 or row == ngridy-1 or col == 0 or col == ngridx-1:
				continue
			else:
				check_criteria_1 = np.sign(vy_grid[row][col-1]) + np.sign(vy_grid[row][col+1])
				check_criteria_2 = np.sign(vx_grid[row-1][col]) + np.sign(vx_grid[row+1][col])
				check_criteria_3 = np.sign(vy_grid[row][col-1]) + np.sign(vx_grid[row+1][col])
				if math.isnan(check_criteria_1) or math.isnan(check_criteria_2) or math.isnan(check_criteria_3):
					continue
				elif int(check_criteria_1) == 0  and int(check_criteria_2) == 0 and check_criteria_3 != 0:
					vortex_pos.append([x_grid[row][col], y_grid[row][col]])
	return np.array(vortex_pos)

def cavity_vortex_result(test_result_path):
	_, test_loader = load_cavity_data(use_partial_data=True, depth_threshold=2.)
	with open(test_result_path, 'rb') as f:
		v_recon = pickle.load(f)

	score_meter = VortexScoreMeter(thres=0.001)
	scale = 0.1 # for cavity dimension
	for inputs in test_loader:
		w = inputs['properties']['w'] * scale
		h = inputs['properties']['l'] * scale

		gt_vortex_pos = []
		for j in inputs['vortex'].keys():
			temp_vortex_pts = inputs['vortex'][j]['pts']  # list of list, contain several satisfied vortices
			gt_vortex_pos.append(np.mean(np.array(temp_vortex_pts), axis=0))
		if len(gt_vortex_pos) == 0: continue
		gt_vortex_pos = np.array(gt_vortex_pos)
		pred_vortex_pos = find_vortex(v_recon, inputs['pos'], w, h)
		score_meter.update(gt_vortex_pos, pred_vortex_pos)
	score_meter.print_score()


def get_cylinder_ROI_box(pos, node_type):
	wall_id = np.where(node_type == 3)[0]
	cylinder_id = [i for i in wall_id if 0 < pos[i, 1] < 0.40]
	l = pos[cylinder_id, 0].min() - 0.05
	r = pos[cylinder_id, 0].max() + 0.45
	b = pos[cylinder_id, 1].min() - 0.05
	t = pos[cylinder_id, 1].max() + 0.05
	return (l, r, b, t)


def cylinder_vortex_result(args, test_result_path, n_samples=10, start=0, end=300, skip=1):
	_, gt_rollout = load_cfd_traj(args.timestep_end, args.timestep_len)
	with open(test_result_path, 'rb') as f:
		pred_rollout = pickle.load(f)

	score_meter = VortexScoreMeter(thres=0.02)
	for traj_id in range(0, n_samples):
		pos = pred_rollout[traj_id]['mesh_pos']
		node_type = pred_rollout[traj_id]['node_type']
		l, r, b, t = get_cylinder_ROI_box(pos, node_type)
		ngridx = int((r-l)*1000)
		ngridy = int((t-b)*1000)
		xinterpmin = l
		xinterpmax = r
		yinterpmin = b
		yinterpmax = t
		xi = np.linspace(xinterpmin, xinterpmax, ngridx)
		yi = np.linspace(yinterpmin, yinterpmax, ngridy)
		xinterp, yinterp = np.meshgrid(xi, yi)

		for i in range(start, end, skip):
			velocity = pred_rollout[traj_id]['velocity'][i]
			velocity[np.where(pred_rollout[traj_id]['node_type'] == 3)[0]] = np.zeros(2)
			gt_velocity = gt_rollout[traj_id]['velocity'][i]

			v_interp = griddata(pos, velocity, (xinterp, yinterp), method='linear')  # (400, 1600, 2)
			gt_v_interp = griddata(pos, gt_velocity, (xinterp, yinterp), method='linear')
			pred_vortex_pos = find_vortex_in_grid(xinterp, yinterp, v_interp[:, :, 0], v_interp[:, :, 1])
			gt_vortex_pos = find_vortex_in_grid(xinterp, yinterp, gt_v_interp[:, :, 0], gt_v_interp[:, :, 1])
			score_meter.update(gt_vortex_pos, pred_vortex_pos)
	score_meter.print_score()
