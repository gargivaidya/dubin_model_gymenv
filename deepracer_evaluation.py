#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, Pose2D
from ctrl_pkg.msg import ServoCtrlMsg
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
from std_srvs.srv import Empty
import argparse
import datetime
import itertools
import torch, gc
import message_filters
gc.collect()
from sac import SAC
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
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
parser.add_argument('--num_steps', type=int, default=500000, metavar='N',
					help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
					help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
					help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
					help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
					help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
					help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda',type=int, default=0, metavar='N',
					help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=400, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()
rospy.init_node('deepracer_gym', anonymous=True)
x_pub = rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=1)

pos = [0,0]
yaw_car = 0
MAX_VEL = 1.0 #1
steer_precision = 0 # 1e-3
MAX_STEER = (np.pi*0.8 - steer_precision)
MAX_YAW = 2*np.pi
MAX_X = 5
MAX_Y = 5
THRESHOLD_DISTANCE_2_GOAL =  0.05#0.6/max(MAX_X,MAX_Y)
UPDATE_EVERY = 5
count = 0
total_numsteps = 0
updates = 0
num_goal_reached = 0
done = False
i_episode = 1
episode_reward = 0
max_ep_reward = 0
episode_steps = 0
done = False
# memory = ReplayMemory(args.replay_size, args.seed)

class DeepracerGym(gym.Env):

	def __init__(self, start_point):
		super(DeepracerGym,self).__init__()
		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32) # speed and steering
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.target_point = [0./MAX_X, 0./MAX_Y, 1.57]
		self.pose = [start_point[0]/MAX_X, start_point[1]/MAX_Y, start_point[2]]
		self.action = [0., 0.]
		self.traj_x = [self.pose[0]*MAX_X]
		self.traj_y = [self.pose[1]*MAX_Y]
		self.traj_yaw = [self.pose[2]]

	def reset(self):        
		global yaw_car
		self.stop_car() 
		pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose 
		return_state = pose_deepracer
		
		return return_state

	def step(self,action):
		global yaw_car
		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = action[0]*MAX_VEL
		msg.angle = action[1]*MAX_STEER
		x_pub.publish(msg)
		time.sleep(0.03)
		reward = 0
		done = False

		if((abs(pos[0]) < 1.) and (abs(pos[1]) < 1.) ):
			if(abs(pos[0]-self.target_point[0])<THRESHOLD_DISTANCE_2_GOAL and abs(pos[1]-self.target_point[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')

			# else:
			# 	reward = self.get_reward(pos[0],pos[1])

			pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose

		else: 
			done = True
			print('Outside Range')
			reward = -1
			temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
			temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]

			head = math.atan((self.target_point[1]-pos[1])/(self.target_point[0]-pos[0]+0.01)) #calculate pose to target dierction
			pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose 

		info = {}

		return_state = pose_deepracer
		return return_state,reward,done,info     

	def stop_car(self):
		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = 0.
		msg.angle = 0.
		x_pub.publish(msg)
		time.sleep(0.03)
	
	def render(self):
		pass

	def close(self):
		pass


def axis_transform(data):
	return [data.y, -1*data.x, data.yaw+1.57]

def pose_transform(pose, target, subtract):
	if not subtract:
		return (pose[0]*MAX_X)+target[0], (pose[1]*MAX_Y)+target[1]
	else:
		return (pose[0]-target[0])/MAX_X, (pose[1]-target[1])/MAX_Y

def get_distance(x1,x2):
	return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

def get_closest_idx(waypoints, pose):
	# Get closest waypoint index from current position of car
	d_to_waypoints = np.zeros(len(waypoints)) 

	for i in range(len(waypoints)):
		d_to_waypoints[i] = get_distance(waypoints[i], pose) # Calculate distance from each of the waypoints 

	prev_ind, next_ind = np.argpartition(d_to_waypoints, 2)[:2] # Find the index to two least distance waypoints
	return max(prev_ind, next_ind)  # Next waypoint to track is higher of the two indices in the sequence of waypoints


### Set Waypoints
start_point = [0., 0., 1.57]
waypoints = [[0., 1., 1.57], [0., 2., 1.57],[0.5, 3., 1.57], [1., 4., 1.57], [2., 5., 1.57], [3., 6., 1.57], [3., 7., 1.57], [4., 8., 1.57]]


### Initialize model and complete setup
env =  DeepracerGym(start_point)
actor_path = "models/sac_actor_random_initial_2"
critic_path = "models/sac_critic_random_initial_2"
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model(actor_path, critic_path)
agent = SAC(env.observation_space.shape[0], env.action_space, args)
state = np.zeros(env.observation_space.shape[0])

torch.manual_seed(args.seed)
np.random.seed(args.seed)


#Fix the target for the trained model
env.target_point = [0, 0, 1.57]
#Initialize variables
i_waypoint = 0


def pose_callback(data):
	global i_waypoint, waypoints
	# global pos
	# print("Actual State: ", data.x, data.y)
	# pos[0] = (data.x + 1.0)/MAX_X  # Add as per offset 
	# pos[1] = (data.y - 0.5)/MAX_Y # Subtract as per offset
	# yaw_car = data.theta
	# state = np.array([pos[0], pos[1], yaw_car])
	current_pose = axis_transform(data)
	print('Actual State: ', current_pose)

	# state = env.reset()
	updates = 0		
	episode_reward = 0
	episode_steps = 0
	total_numsteps = 1000000
	num_goal_reached = 0
	
	if get_closest_idx(waypoints, current_pose) > i_waypoint:
        #Update target point
        i_waypoint += 1
	
	#Transform pose to model architecture
	x, y = pose_transform(current_pose, waypoints[i_waypoint], True)
	state = [x, y, current_pose[2]]

	yaw_car = current_pose[2]

	action = agent.select_action(state)  # Sample action from policy
	time.sleep(0.01) # Added delay to make up fo network delay during training
	next_state, reward, done, _ = env.step(action) # Step

	if (reward > 9) and (episode_steps > 1): #Count the number of times the goal is reached
		num_goal_reached += 1 

	mask = 1 if episode_steps == args.max_episode_length else float(not done)
	state = next_state

	if done and (i_waypoint == len(waypoints)-1):
		print('----------------------Evaluation Ending----------------------')
		env.stop_car()

def start():
	global ts
	torch.cuda.empty_cache()		
	x = rospy.Subscriber("/pose2D", Pose2D, pose_callback)
	state = env.reset()
	rospy.spin()

if __name__ == '__main__':
	try:
		Flag = False
		Flag = start()
		if Flag:
			print('----------_All Done-------------')
	except rospy.ROSInterruptException:
		pass