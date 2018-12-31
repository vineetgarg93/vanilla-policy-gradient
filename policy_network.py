import torch.nn as nn
import torch

class Policy(nn.module):
	"""docstring for Policy"""
	def __init__(self, nS, nA, network_size = (64, 64), continuous = False):
		super(Policy, self).__init__()
		self.nA = nA
		self.nS = nS
		self.continuous = continuous

		if not self.continuous:
			self.model = nn.Sequential(nn.Linear(self.nS, network_size[0]),
									nn.Linear(network_size[0], network_size[1]),
									nn.Linear(network_size[1], self.nA),
									nn.Softmax())
		else:
			self.action_mean = nn.Sequential(nn.Linear(self.nS, network_size[0]),
									nn.Linear(network_size[0], network_size[1]),
									nn.Linear(network_size[1], self.nA))

			self.action_log_std = nn.Parameter(torch.zeros(1, self.nA))

	
	def forward(self, state)

		if not self.continuous:

			action_prob = self.model(state)
			return action_prob
			
		else:

			action_mean = self.action_mean(state)
			action_log_std = self.action_log_std.expand_as(action_mean)
			action_std = torch.exp(action_log_std)
			return action_mean, action_log_std, action_std