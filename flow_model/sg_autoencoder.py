import numpy as np
import torch
import torch.nn as nn

from mesh_graph_modules import EdgeSet, MultiGraph
from mesh_graph_modules.selector import NHybridRandomSelector
from mesh_graph_modules.reverser import IDWReverser
from mesh_graph_modules.core import MeshGraphNet
from .node_info import CavityNodeInfo, CylinderNodeInfo
from .base import FlowModel


class SubGraphAutoencoder(FlowModel):
	def __init__(self, selector, reverser, node_dim, edge_dim, h_dim, z_dim, out_dim, n_blocks, device):
		super().__init__(None, node_dim, device)
		self.mgn1 = MeshGraphNet(node_dim, edge_dim, h_dim, z_dim, n_blocks)
		self.selector = selector
		self.reverser = reverser
		self.mgn2 = MeshGraphNet(z_dim, edge_dim, h_dim, out_dim, n_blocks)


class CavityAutoencoder(SubGraphAutoencoder):
	def forward(self, inputs, is_training):
		mesh_graph = self.build_graph(inputs, is_training=is_training)
		h = self.mgn1(mesh_graph)

		node_info = CavityNodeInfo(inputs, use_IDW=isinstance(self.reverser, IDWReverser))
		sample_idx = self.selector.sample(node_info)
		z = h[sample_idx, :]

		h_reversed = self.reverser.reverse(z, sample_idx, node_info)
		new_graph = MultiGraph(node_features=h_reversed,
		                       edge_sets=mesh_graph.edge_sets)
		output = self.mgn2(new_graph)

		return self.loss_recon(inputs, output, is_training)


class CylinderAutoencoder(SubGraphAutoencoder):
	def forward(self, inputs, is_training, noise=None, get_z=False, input_sample_idx=None):
		velocity = inputs['velocity'].copy()
		if noise is not None:
			normal_idx = np.append(np.where(inputs['node_type'] == 0)[0], np.where(inputs['node_type'] == 2)[0])
			velocity[normal_idx, :] += np.random.normal(0, noise, size=(len(normal_idx), 2))
		node_type = np.eye(np.max(inputs['node_type'])+1)[inputs['node_type'][:,0].tolist()]
		node_features = np.hstack((velocity, node_type))
		node_features = self.node_normalizer(torch.Tensor(node_features).to(self.device), accumulate=is_training)
		mesh_graph = self.build_graph(inputs, is_training=is_training)
		mesh_graph = mesh_graph._replace(node_features=node_features)
		h = self.mgn1(mesh_graph)

		node_info = inputs['node_info'] if 'node_info' in inputs else \
			CylinderNodeInfo(inputs, use_IDW=isinstance(self.reverser, IDWReverser))
		node_info.velocity = inputs['velocity']
		inputs['node_info'] = node_info

		if input_sample_idx is None:
			sample_idx = inputs['sample_idx'] if 'sample_idx' in inputs else self.selector.sample(node_info)
			if isinstance(self.selector, NHybridRandomSelector):
				inputs['fixed_idx'] = inputs.get('fixed_idx', self.selector.fixed_idx)
				sample_idx = self.selector.sample_vortex(node_info, self.selector.n_samples // 2, inputs['fixed_idx'])
			inputs['sample_idx'] = sample_idx
		else:
			sample_idx = input_sample_idx

		if get_z: return h.detach().cpu().numpy(), sample_idx
		z = h[sample_idx, :]

		h_reversed = self.reverser.reverse(z, sample_idx, node_info)
		new_graph = MultiGraph(node_features=h_reversed,
		                       edge_sets=mesh_graph.edge_sets)
		output = self.mgn2(new_graph)

		return self.loss_recon(inputs, output, is_training)

	def decode(self, inputs, z, sample_idx):
		node_type = np.eye(np.max(inputs['node_type'])+1)[inputs['node_type'][:,0].tolist()]
		node_features = np.hstack((inputs['velocity'], node_type))
		node_features = self.node_normalizer(torch.Tensor(node_features).to(self.device), accumulate=False)
		mesh_graph = self.build_graph(inputs, is_training=False)
		mesh_graph = mesh_graph._replace(node_features=node_features)

		node_info = inputs['node_info'] if 'node_info' in inputs else \
			CylinderNodeInfo(inputs, use_IDW=isinstance(self.reverser, IDWReverser))
		node_info.velocity = inputs['velocity']
		inputs['node_info'] = node_info

		z = torch.Tensor(z).to(self.device)
		h_reversed = self.reverser.reverse(z, sample_idx, node_info)
		new_graph = MultiGraph(node_features=h_reversed,
		                       edge_sets=mesh_graph.edge_sets)
		output = self.mgn2(new_graph)
		target_normalized = self.output_normalizer(torch.Tensor(inputs['velocity']).to(self.device), accumulate=False)
		criterion = nn.MSELoss()

		v_recon = self.output_normalizer.inverse(output.detach()).cpu().numpy()
		v_true = inputs['velocity']

		mask_idx = node_info.mask_idx
		loss = criterion(target_normalized[mask_idx, :], output[mask_idx, :])
		ro_mse = self.root_mse(v_recon[mask_idx, :], v_true[mask_idx, :])
		re_mse = self.relative_mse(v_recon[mask_idx, :], v_true[mask_idx, :])
		return loss, v_recon, ro_mse, re_mse
