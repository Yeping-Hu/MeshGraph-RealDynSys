import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata


class Selector():
	def __init__(self):
		self.samples = None

	def include_random_sample(self, n_nodes, n_random, include_idx):
		"""
		Randomly select `n_random` nodes from the graph that are not `include_idx`. Append the selected idx to
		`include_idx`
		"""
		pool = np.setdiff1d(np.arange(n_nodes), include_idx)
		random_idx = np.random.choice(pool, size=n_random, replace=False)
		return np.append(include_idx, random_idx).astype(int)

	def sample(self, node_info):
		raise NotImplementedError


class NSamplesRandomSelector(Selector):
	def __init__(self, n_samples):
		super().__init__()
		self.n_samples = n_samples

	def sample(self, node_info):
		self.samples = self.include_random_sample(node_info.n_nodes, self.n_samples, [])
		return self.samples


class RatioRandomSelector(Selector):
	def __init__(self, ratio):
		super().__init__()
		self.ratio = ratio

	def sample(self, node_info):
		self.samples = self.include_random_sample(node_info.n_nodes, int(self.ratio * node_info.n_nodes), [])
		return self.samples


class NSamplesCornerRandomSelector(NSamplesRandomSelector):
	def sample(self, node_info):
		corner_idx = node_info.corner_idx
		self.samples = self.include_random_sample(node_info.n_nodes, self.n_samples-len(corner_idx), corner_idx)
		return self.samples


class RatioCornerRandomSelector(RatioRandomSelector):
	def sample(self, node_info):
		n_samples = int(self.ratio * node_info.n_nodes)
		corner_idx = node_info.corner_idx
		self.samples = self.include_random_sample(node_info.n_nodes, n_samples-len(corner_idx), corner_idx)
		return self.samples


class NSamplesBoundRandomSelector(NSamplesRandomSelector):
	def sample(self, node_info):
		boundary_idx = node_info.boundary_idx
		self.samples = self.include_random_sample(node_info.n_nodes, self.n_samples-len(boundary_idx), boundary_idx)
		return self.samples


class RatioBoundRandomSelector(RatioRandomSelector):
	def sample(self, node_info):
		n_samples = int(self.ratio * node_info.n_nodes)
		boundary_idx = node_info.boundary_idx
		self.samples = self.include_random_sample(node_info.n_nodes, n_samples-len(boundary_idx), boundary_idx)
		return self.samples


class VortexSelector(Selector):
	def compute_vortex_gamma(self, v, pos, radius=1e-2, N=20):
		# alpha = np.random.uniform(0, 2*np.pi, size=N)
		alpha = np.linspace(0, 2*np.pi, N+1)[:N]
		m_pos = np.column_stack([radius * np.cos(alpha), radius * np.sin(alpha)])
		m_pos_norm = np.linalg.norm(m_pos, axis=1)
		n_nodes = pos.shape[0]
		gamma = np.zeros(n_nodes)

		pos_N = np.repeat(pos, N, axis=0)
		m_pos_n_nodes = np.tile(m_pos, (n_nodes, 1))
		v_m = griddata(pos, v, pos_N + m_pos_n_nodes, method='linear') # shape (N * n_nodes, 2)
		v_m = v_m.reshape((n_nodes, N, 2))

		# convert nan at boundaries to 0
		v_m_norm = np.linalg.norm(v_m, axis=2) + 1e-8
		sin = np.cross(v_m, np.tile(m_pos, (n_nodes,1,1)), axis=2) / (v_m_norm * m_pos_norm.reshape(1, -1))
		gamma = np.nan_to_num(sin.mean(axis=1), nan=0)

		return np.abs(gamma)

	def sample_vortex(self, node_info, n_vortex, selected_idx, radius=1e-3):
		exclude = np.append(node_info.boundary_idx, selected_idx)
		pool = np.setdiff1d(np.arange(node_info.n_nodes), exclude)
		gamma = self.compute_vortex_gamma(node_info.velocity, node_info.mesh_pos, radius)
		selected = pool[np.argsort(gamma[pool])[-n_vortex:]]
		self.samples = np.append(selected_idx, selected).astype(int)
		return self.samples


class VortexRandomSelector(RatioRandomSelector, VortexSelector):
	def sample(self, node_info):
		n = node_info.n_nodes
		n_samples = int(self.ratio * n)
		pool = np.arange(n)
		gamma = self.compute_vortex_gamma(node_info.velocity, node_info.mesh_pos, radius=1e-3)
		weights = gamma[pool]
		self.samples = np.random.choice(pool, size=n_samples, replace=False, p=weights/weights.sum())
		return self.samples


class NVortexRandomSelector(NSamplesRandomSelector, VortexSelector):
	def sample(self, node_info):
		n = node_info.n_nodes
		pool = np.arange(n)
		gamma = self.compute_vortex_gamma(node_info.velocity, node_info.mesh_pos, radius=1e-3)
		weights = gamma[pool]
		self.samples = np.random.choice(pool, size=self.n_samples, replace=False, p=weights/weights.sum())
		return self.samples


class HybridRandomSelector(RatioRandomSelector, VortexSelector):
	def sample(self, node_info):
		n = node_info.n_nodes
		n_random = n_vortex = int(self.ratio * n) // 2
		self.fixed_idx = self.include_random_sample(node_info.n_nodes, n_random, [])
		self.samples = self.sample_vortex(node_info, n_vortex, self.fixed_idx, radius=1e-3)
		return self.samples


class CornerHybridRandomSelector(RatioRandomSelector, VortexSelector):
	def sample(self, node_info):
		corner_idx = node_info.corner_idx
		n = node_info.mesh_pos.shape[0]
		n_random = int(self.ratio * n) // 2 - corner_idx.shape[0]
		n_vortex = int(self.ratio * n) // 2
		self.fixed_idx = self.include_random_sample(n, n_random, corner_idx)
		self.samples = self.sample_vortex(node_info, n_vortex, self.fixed_idx, radius=1e-3)
		return self.samples


class NHybridRandomSelector(NSamplesRandomSelector, VortexSelector):
	def sample(self, node_info):
		n_random = n_vortex = self.n_samples // 2
		self.fixed_idx = self.include_random_sample(node_info.n_nodes, n_random, [])
		self.samples = self.sample_vortex(node_info, n_vortex, self.fixed_idx, radius=1e-2)
		return self.samples


class NCornerHybridRandomSelector(NHybridRandomSelector):
	def sample(self, node_info):
		corner_idx = node_info.corner_idx
		n = node_info.n_nodes
		n_random = self.n_samples // 2 - corner_idx.shape[0]
		n_vortex = self.n_samples // 2
		self.fixed_idx  = self.include_random_sample(n, n_random, corner_idx)
		self.samples = self.sample_vortex(node_info, n_vortex, self.fixed_idx, radius=1e-2)
		return self.samples
