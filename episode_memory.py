import numpy as np


class Memory(object):
	"""docstring for Memory"""
	def __init__(self):
		super(Memory, self).__init__()

		self.episode_states = []
		self.episode_actions = []
		self.episode_rewards = []
		self.discounted_rewards = []
		
	
	def reset(self):

		self.episode_states = []
		self.episode_actions = []
		self.episode_rewards = []
		self.discounted_rewards = []

	def store_transition(state, action, reward):




		self.episode_states.append(state)
		self.episode_actions.append(action)
		self.episode_rewards.append(reward)


	def get_cumulative_discounted_rewards(self, discount = 0.99):
		
		for i in range(len(self.episode_rewards)):
			if len(self.discounted_rewards) == 0:
				self.discounted_rewards.append(self.episode_rewards[-1-i])
			else:
				self.discounted_rewards.append(discount*self.discounted_rewards[-1])

		
		self.discounted_rewards = self.discounted_rewards[::-1]
		return self.discounted_rewards