import numpy as np
import torch
import torch.nn as nn

from mesh_graph_modules import EdgeSet, MultiGraph
from mesh_graph_modules.gnn import GraphNetBlock
from .node_info import CylinderNodeInfo
from .normalization import Normalizer
from .base import FlowModel, triangles_to_edges


class FullGraphPredictor(FlowModel):
	def forward(self, inputs, is_training, noise=None):
		velocity = inputs['velocity'].copy()
		if noise is not None:
			normal_idx = np.append(np.where(inputs['node_type'] == 0)[0], np.where(inputs['node_type'] == 2)[0])
			velocity[normal_idx, :] += np.random.normal(0, noise, size=(len(normal_idx), 2))
		mesh_graph = inputs['mesh_graph'] if 'mesh_graph' in inputs else self.build_graph(inputs, is_training=is_training)
		node_type = np.eye(np.max(inputs['node_type'])+1)[inputs['node_type'][:,0].tolist()]
		node_features = np.hstack((velocity, node_type))
		node_features = self.node_normalizer(torch.Tensor(node_features).to(self.device), accumulate=is_training)
		mesh_graph = mesh_graph._replace(node_features=node_features)

		output = self.mgn(mesh_graph)

		v_target = inputs['target_velocity']
		v_target_update = v_target - velocity
		target_normalized = self.output_normalizer(torch.Tensor(v_target_update).to(self.device))
		criterion = nn.MSELoss()
		node_info = CylinderNodeInfo(inputs, use_IDW=False)
		mask_idx = node_info.mask_idx
		loss = criterion(target_normalized[mask_idx, :], output[mask_idx, :])

		v_update = self.output_normalizer.inverse(output.detach()).cpu().numpy()
		v_pred = velocity + v_update
		ro_mse = self.root_mse(v_pred[mask_idx, :], v_target[mask_idx, :])
		re_mse = self.relative_mse(v_pred[mask_idx, :], v_target[mask_idx, :])
		return loss, v_pred, ro_mse, re_mse


class SubgraphPredictor(nn.Module):
	def __init__(self, z_dim, edge_dim, h_dim, n_blocks, device):
		super().__init__()
		self.encoder_node_mlp = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
		)
		self.encoder_edge_mlp = nn.Sequential(
			nn.Linear(edge_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
		)
		self.decoder_mlp = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, z_dim),
		)
		self.gnn = nn.ModuleList([GraphNetBlock(z_dim=h_dim, h_dim=h_dim) for _ in range(n_blocks)])
		self.device = device
		self.message_passing_steps = n_blocks
		self.output_normalizer = Normalizer(size=z_dim)
		self.node_normalizer = Normalizer(size=z_dim)
		self.edge_normalizer = Normalizer(size=edge_dim)

	def build_graph(self, z, pos, cells, is_training=True):
		# z dim (n_nodes, z_dim)
		senders, receivers = triangles_to_edges(cells)
		relative_mesh_pos = pos[senders] - pos[receivers]
		edge_features = np.hstack((relative_mesh_pos, np.linalg.norm(relative_mesh_pos, axis=1, keepdims=True)))
		edge_features = self.edge_normalizer(torch.Tensor(edge_features).to(self.device), accumulate=is_training)
		z = self.node_normalizer(z, accumulate=is_training)

		mesh_edges = EdgeSet(
			name='mesh_edges',
			features=self.encoder_edge_mlp(edge_features),
			receivers=torch.Tensor(receivers).to(torch.int64).to(self.device),
			senders=torch.Tensor(senders).to(torch.int64).to(self.device))

		mesh_graph = MultiGraph(
			node_features=self.encoder_node_mlp(z),
			edge_sets=[mesh_edges])
		return mesh_graph

	def root_mse(self, z_pred, z_target):
		return torch.mean(torch.sqrt(torch.mean((z_pred - z_target)**2, dim=0))).item()

	def forward(self, z, z_target, pos, cells, is_training=True):
		mesh_graph = self.build_graph(z[-1, :, :], pos, cells, is_training)
		for graph_net in self.gnn:
			mesh_graph = graph_net(mesh_graph)
		output = self.decoder_mlp(mesh_graph.node_features)

		target = z_target - z[-1, :, :]
		target_normalized = self.output_normalizer(target, accumulate=is_training)
		criterion = nn.MSELoss().to(self.device)
		loss = criterion(target_normalized, output)

		z_update = self.output_normalizer.inverse(output.detach())
		rmse = self.root_mse(z[-1, :, :] + z_update, z_target)
		return z[-1, :, :] + z_update, loss, rmse
