import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import MultiGraph


class GraphNetBlock(nn.Module):
	'''
	A message passing graph network block.
	'''
	def __init__(self, z_dim = 128, h_dim = 128):
		super(GraphNetBlock, self).__init__()
		self.z_dim = z_dim
		self.h_dim = h_dim
		node_dim = self.z_dim * 2
		edge_dim = self.z_dim * 3

		self.node_fn = nn.Sequential(
			nn.Linear(node_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.z_dim)
		)
		self.edge_fn = nn.Sequential(
			nn.Linear(edge_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.h_dim),
			nn.ReLU(),
			nn.Linear(self.h_dim, self.z_dim)
		)

	def update_edge_features(self, node_features, edge_set):
		'''
		Aggregates node features, and applies edge function
		'''
		sender_features = node_features[edge_set.senders]
		receiver_features = node_features[edge_set.receivers]
		features = [sender_features, receiver_features, edge_set.features]
		return self.edge_fn(torch.cat(features, dim = 1))



	def update_node_features(self, node_features, edge_sets):
		'''
		Aggregates edge features, and applies node function
		'''
		N_nodes = node_features.shape[0]
		features = [node_features]

		for edge_set in edge_sets:
			N_edge_features = edge_set.features.shape[1]
			sum_edge_feature_to_node = torch.zeros(N_nodes, N_edge_features, device=edge_set.features.device)
			sum_edge_feature_to_node = sum_edge_feature_to_node.scatter_add(
				dim=0,
				index=edge_set.receivers.expand(N_edge_features, -1).t(),
				src=edge_set.features)
			features.append(sum_edge_feature_to_node)

		return self.node_fn(torch.cat(features, dim = 1))


	def forward(self, graph):
		'''
		Applies GraphNetBlock and returns updated MultiGraph
		'''
		# apply edge function
		new_edge_sets = []
		for edge_set in graph.edge_sets:
			updated_edge_features = self.update_edge_features(graph.node_features, edge_set)
			new_edge_sets.append(edge_set._replace(features=updated_edge_features))

		# apply node function
		new_node_features = self.update_node_features(graph.node_features, new_edge_sets)

		# add residual connections
		new_node_features += graph.node_features
		new_edge_sets = [es._replace(features=es.features + old_es.features)
		                 for es, old_es in zip(new_edge_sets, graph.edge_sets)]

		return MultiGraph(node_features=new_node_features, edge_sets=new_edge_sets)
