import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
import time
import itertools
import argparse
import datetime
import random
import torch
from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
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
parser.add_argument('--num_steps', type=int, default=300000, metavar='N',
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
#parser.add_argument('--cuda', action="store_false",
#                    help='run on CUDA (default: False)')
parser.add_argument('--cuda', type = int, default = 0, metavar = 'N',
                    help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=400, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()


MAX_STEER = 2.84
MAX_SPEED = 0.22
MIN_SPEED = 0.
THRESHOLD_DISTANCE_2_GOAL = 0.02
MAX_X = 10.
MAX_Y = 10.
THETA0 = np.pi/4

# Vehicle parameters
LENGTH = 0.25  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.05  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.15  # [m]

show_animation = True

class DubinGym(gym.Env):

	def __init__(self, start_point):
		super(DubinGym,self).__init__()
		metadata = {'render.modes': ['console']}
		self.action_space = spaces.Box(np.array([-0.22, -2.84]), np.array([0.22, 2.84]), dtype = np.float32) # max rotational velocity of burger is 2.84 rad/s
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low, high, dtype=np.float32)
		self.target = [0./MAX_X, 0./MAX_Y, 1.57]
		self.pose = [start_point[0]/MAX_X, start_point[1]/MAX_Y, start_point[2]]
		self.action = [0., 0.]
		self.traj_x = [self.pose[0]*MAX_X]
		self.traj_y = [self.pose[1]*MAX_Y]
		self.traj_yaw = [self.pose[2]]

	def reset(self): 
		x = random.uniform(-1., 1.)
		y = random.choice([-1., 1.])*math.sqrt(1. - x**2)
		theta = self.get_heading([x, y], self.target)
		yaw = random.uniform(theta - THETA0, theta + THETA0)
		self.pose = np.array([x/MAX_X, y/MAX_Y, yaw])
		self.traj_x = [x]
		self.traj_y = [y]
		self.traj_yaw = [yaw]
		return np.array(self.pose)

	def get_reward(self):
		x_target = self.target[0]
		y_target = self.target[1]
		yaw_target = self.target[2]
		x = self.pose[0]
		y = self.pose[1]
		yaw_car = self.pose[2]
		head_to_target = self.get_heading(self.pose, self.target)
		alpha = head_to_target - self.pose[2]
		ld = self.get_distance(self.pose, self.target)
		crossTrackError = math.sin(alpha) * ld

		return -1*(abs(crossTrackError)**2 + abs(x - x_target) + abs(y - y_target) + 3*abs (head_to_target - yaw_car)/1.57)/6

	def get_distance(self,x1, x2):
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1, x2):
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))		

	def step(self,action):
		reward = 0
		done = False
		info = {}
		self.action = action
		self.pose = self.update_state(self.pose, action, 0.005) # 0.005 Modify time discretization


		if ((abs(self.pose[0]) < 1.) and (abs(self.pose[1]) < 1.)):

			if(abs(self.pose[0]-self.target[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(self.pose[1]-self.target[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')
				#print("Distance : {:.3f} {:.3f}".format(abs(self.pose[0]-self.target[0])*MAX_X, abs(self.pose[1]-self.target[1])*MAX_Y))
			else:
				reward = self.get_reward()	
		else :
			done = True
			reward = -1.
			print("Outside range")
			#print("Distance : {:.3f} {:.3f}".format(abs(self.pose[0]-self.target[0])*MAX_X, abs(self.pose[1]-self.target[1])*MAX_Y))

		return np.array(self.pose), reward, done, info     

	def render(self):
		self.traj_x.append(self.pose[0]*MAX_X)
		self.traj_y.append(self.pose[1]*MAX_Y)
		self.traj_yaw.append(self.pose[2])
	  
		plt.cla()
		# for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect('key_release_event',
				lambda event: [exit(0) if event.key == 'escape' else None])
		plt.plot(self.traj_x, self.traj_y, "ob", markersize = 2, label="trajectory")
		plt.plot(self.target[0]*MAX_X, self.target[1]*MAX_Y, "xg", label="target")
		self.plot_car()
		plt.axis("equal")
		plt.grid(True)
		plt.title("Simulation")
		plt.pause(0.0001)
		

	def close(self):
		pass

	def update_state(self, state, a, DT):

		lin_velocity = a[0]
		rot_velocity = a[1]

		if rot_velocity >= MAX_STEER:
			rot_velocity = MAX_STEER
		elif rot_velocity <= -MAX_STEER:
			rot_velocity = -MAX_STEER

		if lin_velocity > MAX_SPEED:
			lin_velocity = MAX_SPEED
		elif lin_velocity < MIN_SPEED:
			lin_velocity = MIN_SPEED


		state[0] = state[0] + lin_velocity * math.cos(state[2]) * DT
		state[1] = state[1] + lin_velocity * math.sin(state[2]) * DT
		state[2] = state[2] + rot_velocity * DT

		return state

	def plot_car(self, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
		# print("Plotting Car")
		x = self.pose[0]*MAX_X #self.pose[0]
		y = self.pose[1]*MAX_Y #self.pose[1]
		yaw = self.pose[2] #self.pose[2]
		steer = self.action[1]*MAX_STEER #self.action[1]

		outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
							[WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

		fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
							 [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

		rr_wheel = np.copy(fr_wheel)

		fl_wheel = np.copy(fr_wheel)
		fl_wheel[1, :] *= -1
		rl_wheel = np.copy(rr_wheel)
		rl_wheel[1, :] *= -1

		Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
						 [-math.sin(yaw), math.cos(yaw)]])
		Rot2 = np.array([[math.cos(steer), math.sin(steer)],
						 [-math.sin(steer), math.cos(steer)]])

		fr_wheel = (fr_wheel.T.dot(Rot2)).T
		fl_wheel = (fl_wheel.T.dot(Rot2)).T
		fr_wheel[0, :] += WB
		fl_wheel[0, :] += WB

		fr_wheel = (fr_wheel.T.dot(Rot1)).T
		fl_wheel = (fl_wheel.T.dot(Rot1)).T

		outline = (outline.T.dot(Rot1)).T
		rr_wheel = (rr_wheel.T.dot(Rot1)).T
		rl_wheel = (rl_wheel.T.dot(Rot1)).T

		outline[0, :] += x
		outline[1, :] += y
		fr_wheel[0, :] += x
		fr_wheel[1, :] += y
		rr_wheel[0, :] += x
		rr_wheel[1, :] += y
		fl_wheel[0, :] += x
		fl_wheel[1, :] += y
		rl_wheel[0, :] += x
		rl_wheel[1, :] += y

		plt.plot(np.array(outline[0, :]).flatten(),
				 np.array(outline[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fr_wheel[0, :]).flatten(),
				 np.array(fr_wheel[1, :]).flatten(), truckcolor)
		plt.plot(np.array(rr_wheel[0, :]).flatten(),
				 np.array(rr_wheel[1, :]).flatten(), truckcolor)
		plt.plot(np.array(fl_wheel[0, :]).flatten(),
				 np.array(fl_wheel[1, :]).flatten(), truckcolor)
		plt.plot(np.array(rl_wheel[0, :]).flatten(),
				 np.array(rl_wheel[1, :]).flatten(), truckcolor)
		plt.plot(x, y, "*") 

def main():

	env =  DubinGym([1., 0., -1.57])
	## Model Training
	agent = SAC(env.observation_space.shape[0], env.action_space, args)
	# Memory
	memory = ReplayMemory(args.replay_size, args.seed)
	#Tesnorboard
	writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'DeepracerGym',
														 args.policy, "autotune" if args.automatic_entropy_tuning else ""))

	total_numsteps = 0
	updates = 0
	num_goal_reached = 0

	for i_episode in itertools.count(1):
		# print("New episode")
		episode_reward = 0
		episode_steps = 0
		done = False
		state = env.reset()
		
		while not done:
			# env.render() # Rendering toggle
			start_time = time.time()
			if args.start_steps > total_numsteps:
				action = env.action_space.sample()  # Sample random action
			else:
				action = agent.select_action(state)  # Sample action from policy

			next_state, reward, done, _ = env.step(action) # Step
			if (reward > 9) and (episode_steps > 1): #Count the number of times the goal is reached
				num_goal_reached += 1 

			episode_steps += 1
			total_numsteps += 1
			episode_reward += reward
			if episode_steps > args.max_episode_length:
				done = True

			# Ignore the "done" signal if it comes from hitting the time horizon.
			# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
			mask = 1 if episode_steps == args.max_episode_length else float(not done)
			# mask = float(not done)
			memory.push(state, action, reward, next_state, mask) # Append transition to memory

			state = next_state
			# print(done)

		# if i_episode % UPDATE_EVERY == 0: 
		if len(memory) > args.batch_size:
			# Number of updates per step in environment
			for i in range(args.updates_per_step*args.max_episode_length):
				# Update parameters of all the networks
				critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

				writer.add_scalar('loss/critic_1', critic_1_loss, updates)
				writer.add_scalar('loss/critic_2', critic_2_loss, updates)
				writer.add_scalar('loss/policy', policy_loss, updates)
				writer.add_scalar('loss/entropy_loss', ent_loss, updates)
				writer.add_scalar('entropy_temprature/alpha', alpha, updates)
				updates += 1

		if total_numsteps > args.num_steps:
			break

		if (episode_steps > 1):
			writer.add_scalar('reward/train', episode_reward, i_episode)
			writer.add_scalar('reward/episode_length',episode_steps, i_episode)
			writer.add_scalar('reward/num_goal_reached',num_goal_reached, i_episode)

		print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
		print("Number of Goals Reached: ",num_goal_reached)

	print('----------------------Training Ending----------------------')
	# env.stop_car()

	agent.save_model("burger", suffix = "1")
	return True

	# # Environment Test
	# max_steps = int(1e6)
	# state = env.reset()
	# env.render()
	# for ep in range(5):
	# 	state = env.reset()
	# 	env.render()
	# 	for i in range(max_steps):
	# 		action = [1.0, 0.]
	# 		n_state,reward,done,info = env.step(action)
	# 		env.render()
	# 		if done:
	# 			state = env.reset()
	# 			done = False                   
	# 			break

if __name__ == '__main__':
	main()
