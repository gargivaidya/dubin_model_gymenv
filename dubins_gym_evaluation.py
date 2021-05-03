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


# Training constants
MAX_STEER = np.pi/3
MAX_SPEED = 10.0
MIN_SPEED = 0.
THRESHOLD_DISTANCE_2_GOAL = 0.02
MAX_X = 5.
MAX_Y = 5.
THETA0 = np.pi/4

# Vehicle parameters
LENGTH = 0.25  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.05  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.15  # [m]


def plot_car(pose, action, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
		# print("Plotting Car")
		# Scale up the car pose to MAX_X, MAX_Y grid
		x = pose[0] 
		y = pose[1] 
		yaw = pose[2]
		steer = action[1]*MAX_STEER 

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


def render(pose, action, traj, waypoints, reset=False):

	if reset:
		traj[0].clear()
		traj[1].clear()
	
	# Storing tracked trajectory 
	traj[0].append(pose[0])
	traj[1].append(pose[1])
	
	plt.cla()
	# for stopping simulation with the esc key.
	plt.gcf().canvas.mpl_connect('key_release_event',
			lambda event: [exit(0) if event.key == 'escape' else None])
	plt.plot(traj[0], traj[1], "ob", markersize = 2, label="trajectory")
	# Rendering waypoint sequence
	for i in range(len(waypoints)-1):
		plt.plot(waypoints[i][0], waypoints[i][1], "^r", label="waypoint")
	plt.plot(waypoints[-1][0], waypoints[-1][1], "xg", label="target")
	# Rendering the car and action taken
	plot_car(pose, action)
	plt.axis("equal")
	plt.grid(True)
	plt.title("Simulation")
	plt.pause(0.0001)


def pose_transform(pose, target, subtract):
	if not subtract:
		return (pose[0]*MAX_X)+target[0], (pose[1]*MAX_Y)+target[1]
	else:
		return (pose[0]-target[0])/MAX_X, (pose[1]-target[1])/MAX_X

def get_distance(x1,x2):
	return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

def get_closest_idx(waypoints, pose):
	# Get closest waypoint index from current position of car
	d_to_waypoints = np.zeros(len(waypoints)) 

	for i in range(len(waypoints)):
		d_to_waypoints[i] = get_distance(waypoints[i], pose) # Calculate distance from each of the waypoints 

	prev_ind, next_ind = np.argpartition(d_to_waypoints, 2)[:2] # Find the index to two least distance waypoints
	return max(prev_ind, next_ind)  # Next waypoint to track is higher of the two indices in the sequence of waypoints


def main():

	### Declare variables for environment
	start_point = [0., 0., 1.57]
	waypoints = [[0., 1., 1.57], [0., 2., 1.57],[0.5, 3., 1.57], [1., 4., 1.57], [2., 5., 1.57], [3., 6., 1.57], [3., 7., 1.57], [4., 8., 1.57]]
	#waypoints = [[0., 1., 1.57], [0.5, 2., 1.57],[1., 3., 1.57], [2., 3.5, 1.57], [3., 3.5, 1.57], [4., 3., 1.57], [4.5, 2., 1.57], [5., 1., 1.57], [5., 0., 1.57]]
	#waypoints = [[0., 1., 1.57], [1., 2., 1.57],[2., 3., 1.57], [3., 3., 1.57], [4., 2., 1.57], [5., 1., 1.57], [5., 0., 1.57]]
	#waypoints = [[0., 1., 1.57], [0.5, 2., 1.57],[1., 2.5, 1.57], [2., 3.5, 1.57], [3., 3.5, 1.57], [4., 2.5, 1.57], [4.5, 2., 1.57], [5., 1., 1.57], [5., 0., 1.57]]
	#waypoints = [[1., 2., 1.57], [3., 2., 1.57],[4., 0., 1.57]]
	#waypoints = [[2., 0., 1.57], [1., -1., 1.57], [0., 0., 1.57]]

	traj = [[],[]]
	env =  DubinGym(start_point)

	### Load your trained model
	actor_path = "models/sac_actor_random_initial_4"
	critic_path = "models/sac_critic_random_initial_4"
	agent = SAC(env.observation_space.shape[0], env.action_space, args)
	agent.load_model(actor_path, critic_path)

	### Evaluation Parameters
	num_goal_reached = 0
	max_steps = 200
	num_iterations = 3

	### Reset and Render
	state = env.reset()
	#Fix the target for the trained model
	env.target = [0, 0, 1.57]

	### Evaluation Loop	
	for ep in range(num_iterations):

		print('########### Iteration: {} ###########'.format(ep+1))
		goals_in_iter = 0
		current_pose = [x for x in start_point]

		#Render environment
		render(current_pose, [0., 0.], traj, waypoints, True)

		for i_waypoint in range(len(waypoints)):

			ep_reward = 0.
			done = False

			# Transform pose to real world
			if i_waypoint > 0:
				current_pose[0], current_pose[1] = pose_transform([state[0], state[1]], target, False)
				current_pose[2] = state[2]

			#Update target point
			target = waypoints[i_waypoint]

			#Transform pose to model architecture
			x, y = pose_transform(current_pose, target, True)
			state = env.pose = [x, y, current_pose[2]]

			for episode_steps in range(max_steps):
				action = agent.select_action(state, True)
				next_state, reward, done, _ = env.step(action)
				ep_reward += reward

				#print("\r Car is at : {}, reward : {:.4f}".format(next_state, reward), end = '\r')

				state = next_state

				# Check if target needs to be updated
				# if i_waypoint < len(waypoints)-1:
				# 	x, y = pose_transform(state, target, False)
				# 	if get_distance([x,y], waypoints[i_waypoint+1]) < (get_distance([x,y], target) + get_distance(target, waypoints[i_waypoint+1])):
				# 		done = True

				# Transform pose to real world
				current_pose[0], current_pose[1] = pose_transform(state, target, False)
				current_pose[2] = state[2]
				
				#Render environment
				render(current_pose, action, traj, waypoints)

				# Check if target needs to be updated
				if i_waypoint > 0:
					if get_closest_idx(waypoints, current_pose) > i_waypoint:
						done = True

				# if done with current goal - reached or passed
				if done:
					# if goal_reached:
					if (reward > 9) and (episode_steps > 1):
						num_goal_reached += 1
						goals_in_iter += 1
					break


			print("Episode : {}, \tEpisode Total Reward : {:.4f}, \tNumber of Times Goal Reached : {}".format(i_waypoint+1, ep_reward, goals_in_iter))

		# Print current iteration statistics
		print("Total Number of Times Goal Reached : {}/{}".format(num_goal_reached, len(waypoints)*(ep+1)))
		print('####################################')

if __name__ == '__main__':
	main()	
