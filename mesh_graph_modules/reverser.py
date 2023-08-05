import torch
import numpy as np
from scipy.spatial import Delaunay


class IDWReverser():
	def __init__(self, n_neighbors):
		self.n_neighbors = n_neighbors

	def reverse(self, node_latents, sample_idx, node_info):
		reversed_latents = torch.zeros((node_info.n_nodes, node_latents.shape[1]),
		                                dtype=node_latents.dtype,
		                                device=node_latents.device)
		reversed_latents[sample_idx, :] = node_latents
		non_sample_idx = np.setdiff1d(np.arange(node_info.n_nodes), sample_idx)
		dmat = node_info.distance_matrix
		d2 = dmat[non_sample_idx, :][:, sample_idx] ** 2
		nearest_nodes = np.argpartition(d2, self.n_neighbors, axis=1)[:, :self.n_neighbors]
		weights = torch.Tensor(1 / d2[np.tile(np.arange(d2.shape[0]), (self.n_neighbors, 1)).T, nearest_nodes]).to(node_latents.device)
		reversed_latents[non_sample_idx] = torch.sum(weights[:, :, None] * node_latents[nearest_nodes, :], dim=1) / weights.sum(axis=1).reshape(-1,1)
		return reversed_latents


class TINReverser():
	def reverse(self, node_latents, sample_idx, node_info):
		pos = node_info.mesh_pos
		subgraph_pos = pos[sample_idx, :]
		tri = Delaunay(subgraph_pos)
		reversed_latents = torch.zeros((node_info.n_nodes, node_latents.shape[1]),
		                                dtype=node_latents.dtype,
		                                device=node_latents.device)
		reversed_latents[sample_idx, :] = node_latents
		non_sample_idx = np.setdiff1d(np.arange(node_info.n_nodes), sample_idx)
		non_sample_pos = pos[non_sample_idx, :]
		non_sample_tri_id = tri.find_simplex(non_sample_pos)
		assert np.all(non_sample_tri_id >= 0)
		b = tri.transform[non_sample_tri_id, :2] @ (non_sample_pos - tri.transform[non_sample_tri_id, 2])[:,:,None] # (n_up, 2, 2) @ (n_up, 2, 1)
		b = b.reshape(b.shape[0], b.shape[1])
		b = torch.Tensor(np.concatenate([b, 1 - b.sum(axis=1, keepdims=True)], axis=1)[:, :, None]).to(node_latents.device) # (n_up, 3, 1)
		reversed_latents[non_sample_idx] = (b * node_latents[tri.simplices[non_sample_tri_id], :]).sum(axis=1) # (n_up, 3, 1) x (n_up, 3, n_dim)
		return reversed_latents


class BoundTINReverser():
	'''
	Assign boundary nodes latents to the mean of sampled boundary nodes
	'''
	def reverse(self, node_latents, sample_idx, node_info):
		pos = node_info.mesh_pos
		sample_bound = np.intersect1d(node_info.boundary_idx, sample_idx)
		if len(sample_bound) == 0:
			raise ValueError('No boundary nodes in sample')
		rest_bound = np.setdiff1d(node_info.boundary_idx, sample_idx)
		subgraph_idx = np.append(rest_bound, sample_idx)
		subgraph_pos = pos[subgraph_idx, :]
		tri = Delaunay(subgraph_pos)
		reversed_latents = torch.zeros((node_info.n_nodes, node_latents.shape[1]),
		                                dtype=node_latents.dtype,
		                                device=node_latents.device)
		reversed_latents[sample_idx, :] = node_latents
		reversed_latents[rest_bound, :] = reversed_latents[sample_bound, :].mean(axis=0)
		non_sample_idx = np.setdiff1d(np.arange(node_info.n_nodes), subgraph_idx)
		non_sample_pos = pos[non_sample_idx, :]
		non_sample_tri_id = tri.find_simplex(non_sample_pos)
		assert np.all(non_sample_tri_id >= 0)
		b = tri.transform[non_sample_tri_id, :2] @ (non_sample_pos - tri.transform[non_sample_tri_id, 2])[:,:,None] # (n_up, 2, 2) @ (n_up, 2, 1)
		b = b.reshape(b.shape[0], b.shape[1])
		b = torch.Tensor(np.concatenate([b, 1 - b.sum(axis=1, keepdims=True)], axis=1)[:, :, None]).to(node_latents.device) # (n_up, 3, 1)
		reversed_latents[non_sample_idx] = (b * reversed_latents[subgraph_idx][tri.simplices[non_sample_tri_id], :]).sum(axis=1) # (n_up, 3, 1) x (n_up, 3, n_dim)
		return reversed_latents


class AllZeroReverser():
	def reverse(self, node_latents, sample_idx, node_info):
		reversed_latents = torch.zeros((node_info.n_nodes, node_latents.shape[1]),
		                               dtype=node_latents.dtype,
		                               device=node_latents.device)
		reversed_latents[sample_idx, :] = node_latents
		return reversed_latents