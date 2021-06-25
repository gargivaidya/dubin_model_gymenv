#!/usr/bin/env python
"""

This script uses Gazebo simulation environment via ROS interface for an 'RL Evaluation' Task in PyTorch.
Task - Evaluate RL model from Dubins Gym to navigate from any random point to global origin

"""
import rospy
import time
from std_msgs.msg import Bool, Float32, Float64
from ackermann_msgs.msg import AckermannDriveStamped
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
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
import pandas as pd

from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

# Gym Environment Constants
MAX_VEL = 1.0 
MAX_STEER = ((1/3)*np.pi)
MAX_YAW = 2*np.pi
MAX_X = 5
MAX_Y = 5
THRESHOLD_DISTANCE_2_GOAL =  0.2/max(MAX_X,MAX_Y)#0.6/max(MAX_X,MAX_Y)

# Global Initialisation
pos = [0,0]
yaw_car = 0
done = False
episode_steps = 0
rospy.init_node('deepracer_gym', anonymous=True)
x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)

class DeepracerGym(gym.Env):
	'''
	Open AI Gym API for Reinforcement Learning Task for Ground Vehicle in Gazebo
	Defines the state, action, reward and reset functionalities for the Gazebo simulation via ROS interface
	'''
	def __init__(self):
		super(DeepracerGym,self).__init__()		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32) # speed and steering
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.target_point = [0./MAX_X, 0./MAX_Y, 1.57]
		self.pose = [pos[0]/MAX_X, pos[1]/MAX_Y, yaw_car]
		self.action = [0., 0.]
		self.traj_x = [self.pose[0]*MAX_X]
		self.traj_y = [self.pose[1]*MAX_Y]
		self.traj_yaw = [self.pose[2]]

	def reset(self):
		'''
		Reset simulation and pose of the vehicle 
		'''
		global yaw_car
		self.stop_car()        
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			# pause physics
			# reset simulation
			# un-pause physics
			self.pause()
			self.reset_simulation_proxy()
			self.unpause()
			print('Simulation reset')
		except rospy.ServiceException as exc:
			print("Reset Service did not process request: " + str(exc))

		# Update agent pose
		pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose 
		return pose_deepracer

	def step(self,action):
		'''
		Executes the action in Gazebo simulation and returns updated pose
		'''
		global yaw_car, x_pub
		msg = AckermannDriveStamped()
		msg.drive.speed = action[0]*MAX_VEL
		msg.drive.steering_angle = action[1]*MAX_STEER
		x_pub.publish(msg)
		time.sleep(0.02)

		reward = 0
		info = {}

		# Check if Goal reached				
		if(abs(pos[0]/MAX_X-self.target_point[0])<THRESHOLD_DISTANCE_2_GOAL and abs(pos[1]/MAX_Y-self.target_point[1])<THRESHOLD_DISTANCE_2_GOAL):
			reward = 10            
			done = True
			print('Goal Reached')
			print('Counter:',episode_steps)
			self.stop_car()

		pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose

		return pose_deepracer, reward, done, info
		  

	def stop_car(self):
		'''
		Stop the vehicle
		'''
		global x_pub		
		msg = AckermannDriveStamped()
		msg.drive.speed = 0.
		msg.drive.steering_angle = 0.
		x_pub.publish(msg)
		time.sleep(1)

# RL Model paths
actor_path = "models/sac_actor_random_initial_5"
critic_path = "models/sac_critic_random_initial_5"

# Instantiate RL Environment and load saved model
env =  DeepracerGym()
agent = SAC(env.observation_space.shape[0], env.action_space, args)
memory = ReplayMemory(args.replay_size, args.seed)
agent.load_model(actor_path, critic_path)
state = np.zeros(env.observation_space.shape[0])

def euler_from_quaternion(x, y, z, w):
	'''
	Convert a quaternion into euler angles (roll, pitch, yaw)
	roll is rotation around x in radians (counterclockwise)
	pitch is rotation around y in radians (counterclockwise)
	yaw is rotation around z in radians (counterclockwise)
	'''
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)

	return roll_x, pitch_y, yaw_z # in radians

def pose_callback(pose_data):
	'''
	Callback function to run the trained RL policy
	'''
	global episode_steps, action
	racecar_pose = pose_data.pose[1]
	pos[0] = racecar_pose.position.x
	pos[1] = racecar_pose.position.y
	q = (
			pose_data.pose[1].orientation.x,
			pose_data.pose[1].orientation.y,
			pose_data.pose[1].orientation.z,
			pose_data.pose[1].orientation.w)
	euler =  euler_from_quaternion(q[0], q[1], q[2], q[3])
	head = math.atan2(pos[1], pos[0]) # Heading to the origin
	yaw = euler[2] 
	state = np.array([(pos[0]/MAX_X), (pos[1]/MAX_Y), yaw]) # golden1 model
	done = False # Ends episode
	
	# Sample action from policy
	action = agent.select_action(state, True)	

	if done: 
		# Stop the car and reset episode
		env.stop_car()
		env.reset()
		print('Counter:',episode_steps)
	else:
		# Execute action
		next_state, reward, done, _ = env.step(action)
		episode_steps += 1
			 
def start():
	'''
	Subscribe to robot pose topic and initiate callback thread
	'''
	global ts, episode_steps, action1, action2
	torch.cuda.empty_cache()
	rospy.init_node('deepracer_gym', anonymous=True)		
	pose_subscriber = rospy.Subscriber("/gazebo/model_states_drop", ModelStates, pose_callback)
	rospy.spin()

if __name__ == '__main__':
	try:
		start()
	except rospy.ROSInterruptException:
		pass
