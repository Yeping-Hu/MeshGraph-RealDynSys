import torch
import torch.nn as nn

from .gnn import GraphNetBlock
from . import EdgeSet, MultiGraph
from .selector import *
from .reverser import *


class MeshGraphNet(nn.Module):
	def __init__(self, node_dim, edge_dim, h_dim, out_dim, n_blocks):
		super(MeshGraphNet, self).__init__()
		self.graph_net = nn.ModuleList([GraphNetBlock(z_dim=h_dim, h_dim=h_dim) for _ in range(n_blocks)])
		self.node_dim = node_dim
		self.edge_dim = edge_dim

		self.message_passing_steps = n_blocks

		self.encoder_node_mlp = nn.Sequential(
			nn.Linear(self.node_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim)
		)
		self.encoder_edge_mlp = nn.Sequential(
			nn.Linear(self.edge_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim)
		)
		self.decoder_mlp = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, out_dim)
		)

	def encoder(self, graph):
		node_latents = self.encoder_node_mlp(graph.node_features)
		new_edge_sets = []
		for edge_set in graph.edge_sets:
			edge_latents = self.encoder_edge_mlp(edge_set.features)
			new_edge_sets.append(edge_set._replace(features=edge_latents))
		return MultiGraph(node_features=node_latents, edge_sets=new_edge_sets)

	def decoder(self, graph):
		return self.decoder_mlp(graph.node_features)

	def forward(self, graph):
		graph = self.encoder(graph)
		for graph_net in self.graph_net:
			graph = graph_net(graph)
		return self.decoder(graph)


def get_selector(args):
	selectorClass = globals()[args.selector]
	if issubclass(selectorClass, RatioRandomSelector):
		return selectorClass(args.select_ratio)
	else:
		return selectorClass(args.n_samples)


def get_reverser(args):
	if args.reverser == 'IDWReverser':
		upsampler = IDWReverser(args.IDW_n_neighbors)
	elif args.reverser == 'AllZeroReverser':
		upsampler = AllZeroReverser()
	elif args.reverser == 'TINReverser':
		upsampler = TINReverser()
	elif args.reverser == 'BoundTINReverser':
		upsampler = BoundTINReverser()
	else:
		raise ValueError(args.upsampler)
	return upsampler


if __name__ == '__main__':
	from argparse import Namespace
	mgn1 = MeshGraphNet(node_dim=5, edge_dim=3, h_dim=8, out_dim=4, n_blocks=3)
	mgn2 = MeshGraphNet(node_dim=4, edge_dim=3, h_dim=8, out_dim=2, n_blocks=3)
	selector = NSamplesRandomSelector(n_samples=3)
	reverser = AllZeroReverser()
	graph = MultiGraph(node_features=torch.randn(10, 5),
					   edge_sets=[EdgeSet(name='edges', features=torch.randn(10, 3),
										  senders=torch.randint(0, 10, (10,)).long(),
										  receivers=torch.randint(0, 10, (10,)).long())])
	h = mgn1(graph)
	print(h.shape)
	node_info = Namespace()
	node_info.n_nodes = 10
	idx = selector.sample(node_info)
	z = h[idx, :]
	print(z.shape)
	h_reversed = reverser.reverse(z, idx, node_info)
	print(h_reversed.shape)
	new_graph = MultiGraph(node_features=h_reversed, edge_sets=graph.edge_sets)
	h2 = mgn2(new_graph)
	print(h2.shape)
