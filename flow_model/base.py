import numpy as np
import torch
import torch.nn as nn

from mesh_graph_modules import EdgeSet, MultiGraph
from .normalization import Normalizer


class FlowModel(nn.Module):
	def __init__(self, mesh_graph_net, node_dim, device):
		super().__init__()
		self.mgn = mesh_graph_net
		self.output_normalizer = Normalizer(size=2)
		self.node_normalizer = Normalizer(size=node_dim)  # node feature(2) + N_node_type
		self.edge_normalizer = Normalizer(size=3)  # 2d coord(2) + length(1)
		self.device = device

	def build_graph(self, inputs, is_training=True):
		# construct graph nodes
		node_type = np.eye(np.max(inputs['node_type'])+1)[inputs['node_type'][:,0].tolist()]
		node_features = np.hstack((inputs['velocity'], node_type))

		# construct graph edges
		senders, receivers = triangles_to_edges(inputs['cells'])
		relative_mesh_pos = inputs['mesh_pos'][senders] - inputs['mesh_pos'][receivers]
		edge_features = np.hstack((relative_mesh_pos, np.linalg.norm(relative_mesh_pos, axis=1, keepdims=True)))

		# construct mesh graph
		mesh_edges = EdgeSet(
			name='mesh_edges',
			features=self.edge_normalizer(torch.Tensor(edge_features).to(self.device), accumulate=is_training),
			receivers=torch.Tensor(receivers).to(torch.int64).to(self.device),
			senders=torch.Tensor(senders).to(torch.int64).to(self.device))

		mesh_graph = MultiGraph(
			node_features=self.node_normalizer(torch.Tensor(node_features).to(self.device), accumulate=is_training),
			edge_sets=[mesh_edges])
		return mesh_graph

	def loss_recon(self, inputs, output, is_training):
		target_normalized = self.output_normalizer(torch.Tensor(inputs['velocity']).to(self.device), accumulate=is_training)
		criterion = nn.MSELoss()
		loss = criterion(target_normalized, output)

		v_recon = self.output_normalizer.inverse(output.detach()).cpu().numpy()
		v_true = inputs['velocity']
		ro_mse = self.root_mse(v_recon, v_true)
		re_mse = self.relative_mse(v_recon, v_true)
		return loss, v_recon, ro_mse, re_mse

	def relative_mse(self, pred, target):
		return np.sum((pred - target) ** 2, axis=0) / np.sum(pred ** 2, axis=0)

	def root_mse(self, pred, target):
		return np.sqrt(np.mean((pred - target)**2, axis=0))

	def forward(self, inputs, is_training):
		mesh_graph = self.build_graph(inputs, is_training=is_training)
		output = self.mgn(mesh_graph)
		return self.loss_recon(inputs, output, is_training)


def triangles_to_edges(cells):
	'''
		Compute mesh edges from triangle cells
	'''
	edges = np.vstack((cells[:, [0,1]], cells[:, [1,2]], cells[:, [2,0]]))

	receivers = np.amin(edges, axis =1)
	senders = np.amax(edges, axis = 1)

	# remove duplicated edges
	unique_edges = np.unique(np.stack((senders, receivers), axis=1), axis = 0)
	senders, receivers = unique_edges[:,0].astype(int), unique_edges[:,1].astype(int)

	return (np.hstack((senders, receivers)), np.hstack((receivers, senders)))