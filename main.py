from policy_network import Policy
import gym
from episode_memory import Memory
from config import Config
import torch
import torch.nn.functional as F

env = gym.make()

policy = Policy(env.nS, env.NA)
train_config = Config()
memory = Memory()

optimizer = optim.Adam(policy.parameters(), lr = train_config.learning_rate)

for episode in range(train_config.num_epsiodes):

	# Training
	policy.train()
	optimizer.zero_grad()

	done = False
	running_length = 0
	memory.reset()
	temp = []
	state = env.reset()

	while(not done and running_length < train_config.max_length)
		
		action_prob = policy.get_action(state)
		action = torch.multinomial(action_prob, 1, replacement=True)

		next_state, reward, done, _ = env.step(action.data.numpy().item())
		memory.store_transition(next_state, action.data.numpy().item(), reward)

		running_length += 1
		state = next_state

		temp.append(F.cross_entropy(action_prob, action))

	# Episode Complete
	loss = -sum(map(lambda x,y: x*y, memory.get_cumulative_discounted_rewards(train_config.discount), temp))
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	# Evaluation
	policy.eval()

	done = False
	running_length = 0
	eval_reward = 0
	state = env.reset()

	while(not done and running_length < train_config.max_length)
		
		action_prob = policy.get_action(state)
		action = torch.multinomial(action_prob, 1, replacement=True)

		next_state, reward, done, _ = env.step(action.data.numpy().item())

		eval_reward += reward*train_config.discount**running_length
		running_length += 1

		state = next_state

	print("Total Reward at the end of: {0} episode is {1}".format(episode+1, eval_reward))