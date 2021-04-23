import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
import time
import itertools
import argparse
import datetime
import torch
from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

### Import Custom Dubin's Gym Environment File
# from dubin_gymenv import DubinGym
from dubins_randomized_AtoB import DubinGym

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=50000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=300, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()

def main():

	### Declare variables for environment
	# start_point = [0., 0., 1.57]
	# target_point = [4., 8., 1.57]
	# waypoints = [[0., 1., 1.57], [0., 2., 1.57],[1., 3., 1.57], [2., 4., 1.57], [3., 5., 1.57], [4., 6., 1.57], [4., 7., 1.57]]
	# n_waypoints = 1 #look ahead waypoints
	# env =  DubinGym(start_point, waypoints, target_point, n_waypoints)
	env =  DubinGym()

	### Load your trained model
	actor_path = "models/sac_actor_random_initial_1"
	critic_path = "models/sac_critic_random_initial_1"
	agent = SAC(env.observation_space.shape[0], env.action_space, args)
	agent.load_model(actor_path, critic_path)

	### Evaluation Parameters	
	num_goal_reached = 0
	max_steps = 200
	num_episodes = 10

	### Reset Environment and Render
	state = env.reset()
	env.render()

	### Evaluation Loop	
	for ep in range(num_episodes):
		ep_reward = 0.		
		done = False
		state = env.reset()
		for _ in range(max_steps):
			action = agent.select_action(state)
			next_state, reward, done, _ = env.step(action)

			env.render()
			ep_reward += reward	

			print("\r Car is at : {}, reward : {:.4f}".format(next_state, reward), end = '\r')		
			if done:

				num_goal_reached += 1
				break

			state = next_state

		print("Episode : {}, \tEpisode Total Reward : {:.4f}, \tNumber of Times Goal Reached : {}".format(ep, ep_reward, num_goal_reached))

if __name__ == '__main__':
	main()