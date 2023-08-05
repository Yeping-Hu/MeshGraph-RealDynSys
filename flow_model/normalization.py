import torch
import torch.nn as nn


class Normalizer(nn.Module):
	'''
	'''

	def __init__(self, size, max_accumulations = 10**6, std_epsilon = 1e-8):
		super(Normalizer, self).__init__()

		self.max_accumulations = max_accumulations
		self.std_epsilon = std_epsilon

		self.acc_count = nn.Parameter(torch.zeros(1).type(torch.FloatTensor), requires_grad=False)
		self.num_accumulations = nn.Parameter(torch.zeros(1).type(torch.FloatTensor), requires_grad=False)

		self.acc_sum = nn.Parameter(torch.zeros(size).type(torch.FloatTensor), requires_grad=False)

		self.acc_sum_squared = nn.Parameter(torch.zeros(size).type(torch.FloatTensor), requires_grad=False)



	def accumulate(self, batched_data):

		count = batched_data.shape[0]
		data_sum = torch.sum(batched_data, dim=0)
		squared_data_sum = torch.sum(batched_data**2, dim=0)

		self.acc_sum += data_sum.detach()
		self.acc_sum_squared += squared_data_sum.detach()
		self.acc_count += count
		self.num_accumulations += 1


	def mean(self):
		safe_count = torch.maximum(self.acc_count, torch.ones(1, device=self.acc_count.device))
		return self.acc_sum / safe_count


	def std_with_epsilon(self):
		safe_count = torch.maximum(self.acc_count, torch.ones(1, device=self.acc_count.device))
		std = torch.sqrt(self.acc_sum_squared / safe_count - self.mean()**2)
		return torch.maximum(std, torch.ones(1, device=std.device) * self.std_epsilon)

	def inverse(self, normalized_batch_data):
		return normalized_batch_data * self.std_with_epsilon() + self.mean()

	def forward(self, batched_data, accumulate = True):
		if accumulate and self.num_accumulations < self.max_accumulations:
			self.accumulate(batched_data)

		return (batched_data - self.mean()) / self.std_with_epsilon()















