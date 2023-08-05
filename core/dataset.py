import os
import json
import pickle
import tensorflow as tf
import functools
import numpy as np
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


def load_cavity_data(use_partial_data, depth_threshold):
	"""
	Load lid-driven cavity flow data.
	:param use_partial_data: if True, only use data with cavity depth less than `depth_threshold`
	:param depth_threshold: cavity depth threshold
	:return: train_loader, test_loader
	"""
	with open('./data/cavity_flow_all.pkl', 'rb') as f:
		all_data_dict = pickle.load(f)

	# ====== Select part of the data that has small depth ====== #
	if use_partial_data:
		satisfied_list = []
		for i in all_data_dict.keys():
			if all_data_dict[i]['properties']['l'] <= depth_threshold:
				satisfied_list.append(i)

	if use_partial_data:
		train_idx, test_idx = train_test_split(satisfied_list, train_size=0.8, random_state=42)
	else:
		train_idx, test_idx = train_test_split(list(all_data_dict.keys()), train_size=0.8, random_state=42)
	print('Total number of training data = ', len(train_idx))
	print('Total number of test data = ', len(test_idx))

	train_loader = [all_data_dict[i] for i in train_idx]
	test_loader = [all_data_dict[i] for i in test_idx]
	return train_loader, test_loader


def load_cfd_traj(end=300, interval=1, ood=False):
	data_dir = f'./data/cylinder_flow_e{end}_i{interval}'
	with open(f'{data_dir}/train.pickle', 'rb') as f:
		train_data = pickle.load(f)
	test_path = f'./data/cfd_two_cylinder.pkl' if ood else f'{data_dir}/valid.pickle'
	with open(test_path, 'rb') as f:
		test_data = pickle.load(f)
	return train_data, test_data


def load_cfd_tfrecord2pickle(end=300, interval=1):
	"""
	Load DeepMind CylinderFlow data from tfrecord and save as pickle. The data is saved in the directory 
	`data/cylinder_flow_e{end}_i{interval}`.
	:param end: end time step
	:param interval: interval between time steps
	"""
	def _parse(proto, meta):
		"""Parses a trajectory from tf.Example."""
		feature_lists = {k: tf.io.VarLenFeature(tf.string)
		                 for k in meta['field_names']}
		features = tf.io.parse_single_example(proto, feature_lists)
		out = {}
		for key, field in meta['features'].items():
			data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
			data = tf.reshape(data, field['shape'])
			if field['type'] == 'static':
				data = tf.tile(data, [meta['trajectory_length'], 1, 1])
			elif field['type'] == 'dynamic_varlen':
				length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
				length = tf.reshape(length, [-1])
				data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
			elif field['type'] != 'dynamic':
				raise ValueError('invalid data format')
			out[key] = data
		return out

	data_dir = './data/cylinder_flow'
	out_dir = f'./data/cylinder_flow_e{end}_i{interval}'
	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join('./data/cylinder_flow/meta.json'), 'r') as fp:
		meta = json.loads(fp.read())
	for split in ['train', 'valid', 'test']:
		ds = tf.data.TFRecordDataset(f'{data_dir}/{split}.tfrecord')
		ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
		loader = iter(ds)
		all_data = []
		for raw_data in loader:
			data = {}
			for field in ['velocity', 'pressure']:
				data[field] = raw_data[field].numpy()[:end:interval, :, :]
				assert data[field].shape[0] == end // interval
			for field in ['cells', 'mesh_pos', 'node_type']:
				data[field] = raw_data[field].numpy()[0]
			# Convert node type in the original dataset to 0, 1, 2, 3
			data['node_type'][np.where(data['node_type']==4)[0]] = 1
			data['node_type'][np.where(data['node_type']==5)[0]] = 2
			data['node_type'][np.where(data['node_type']==6)[0]] = 3
			all_data.append(data)
		with open(f'{out_dir}/{split}.pickle', 'wb') as f:
			pickle.dump(all_data, f)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--end', type=int, default=300, help='desired end time step')
	parser.add_argument('--interval', type=int, default=1, help='interval of time step')
	args = parser.parse_args()
	load_cfd_tfrecord2pickle(end=args.end, interval=args.interval)
