import enum
import numpy as np
import scipy.spatial


class CavityNodeType(enum.IntEnum):
	NORMAL = 0
	WALL = 1
	TOP = 2
	VORTEX = 3

class CylinderNodeType(enum.IntEnum):
	NORMAL = 0
	INFLOW = 1
	OUTFLOW = 2
	WALL = 3
	VORTEX = 4


class NodeInfo:
	def __init__(self, inputs, use_IDW):
		self.mesh_pos = np.array(inputs['mesh_pos'])
		self.distance_matrix = self.compute_distances(self.mesh_pos) if use_IDW else None
		self.n_nodes = self.mesh_pos.shape[0]
		self.velocity = np.array(inputs['velocity'])
		self.cells = inputs['cells']

	def compute_distances(self, mesh_pos):
		return scipy.spatial.distance_matrix(mesh_pos, mesh_pos)

	def get_corner_idx(self):
		raise NotImplementedError


class CavityNodeInfo(NodeInfo):
	def __init__(self, inputs, use_IDW):
		super().__init__(inputs, use_IDW)
		self.normal_idx = np.where(inputs['node_type'] == CavityNodeType.NORMAL)[0]
		self.top_idx = np.where(inputs['node_type'] == CavityNodeType.TOP)[0]
		self.wall_idx = np.where(inputs['node_type'] == CavityNodeType.WALL)[0]
		self.boundary_idx = np.hstack((self.top_idx, self.wall_idx))
		self.vortex_idx = np.where(inputs['node_type'] == CavityNodeType.VORTEX)[0]
		self.cavity = inputs['properties']
		self.corner_idx = self.get_corner_idx()

	def get_corner_idx(self):
		w = self.cavity['w'] / 10
		l = self.cavity['l'] / 10
		x_dent = self.cavity['x_dent'] * w
		y_dent = self.cavity['y_dent'] * l
		if x_dent == 0 and y_dent == 0:
			corner_pos = np.array([[0, 0], [0, l], [w, 0], [w, l]], dtype=np.float32)
		else:
			corner_pos = np.array([[0, 0], [0, l], [w, y_dent], [w, l], [w - x_dent, 0], [w - x_dent, y_dent]], dtype=np.float32)
		corner_idx = np.array([np.argwhere((self.mesh_pos == corner_pos[i]).all(axis=1))[0, 0] for i in range(corner_pos.shape[0])])
		assert len(corner_idx) == corner_pos.shape[0]
		return corner_idx


class CylinderNodeInfo(NodeInfo):
	def __init__(self, inputs, use_IDW):
		super().__init__(inputs, use_IDW)
		self.normal_idx = np.where(inputs['node_type'] == CylinderNodeType.NORMAL)[0]
		self.inflow_idx = np.where(inputs['node_type'] == CylinderNodeType.INFLOW)[0]
		self.outflow_idx = np.where(inputs['node_type'] == CylinderNodeType.OUTFLOW)[0]
		self.wall_idx = np.where(inputs['node_type'] == CylinderNodeType.WALL)[0]
		self.boundary_idx = np.hstack((self.inflow_idx, self.outflow_idx, self.wall_idx))
		self.vortex_idx = np.where(inputs['node_type'] == CylinderNodeType.VORTEX)[0]
		self.mask_idx = np.append(self.normal_idx, self.outflow_idx)
		self.w = inputs['mesh_pos'][:, 0].max() - inputs['mesh_pos'][:, 0].min()
		self.h = inputs['mesh_pos'][:, 1].max() - inputs['mesh_pos'][:, 1].min()
		self.corner_pos = np.array([[0, 0], [0, self.h], [self.w, 0], [self.w, self.h]], dtype=np.float32)
		self.corner_idx = self.get_corner_idx()

	def get_corner_idx(self):
		corner_idx = np.array([np.argwhere((self.mesh_pos == self.corner_pos[i]).all(axis=1))[0, 0] for i in range(self.corner_pos.shape[0])])
		return corner_idx
