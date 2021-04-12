import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
import time

MAX_STEER = np.pi/3
MAX_SPEED = 10.0
MIN_SPEED = 0.
THRESHOLD_DISTANCE_2_GOAL = 0.02
MAX_X = 10.
MAX_Y = 10.

# Vehicle parameters
LENGTH = 0.45  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.1  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.25  # [m]

show_animation = True

class DeepracerGym(gym.Env):

	def __init__(self,start_point, target_point):
		super(DeepracerGym,self).__init__()
		metadata = {'render.modes': ['console']}

		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32)
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.target = [target_point[0]/MAX_X, target_point[1]/MAX_Y]
		self.pose = [start_point[0]/MAX_X, start_point[1]/MAX_Y, start_point[2]]
		self.action = [0., 0.]

	def reset(self): 
		self.observation_space = np.array([0., 0., 1.57])
		return np.array([0., 0., 1.57])

	def get_reward(self):
		x_target = self.target[0]
		y_target = self.target[1]
		x = self.pose[0]
		y = self.pose[1]
		yaw_car = self.pose[2]
		head = math.atan((y_target-y)/(x_target-x+0.01))
		return -1*(abs(x - x_target) + abs(y - y_target) + abs (head - yaw_car))

	def step(self,action):
		reward = 0
		done = False
		info = {}
		self.action = action
		self.pose = self.update_state(self.pose, action, 0.005)
		print(self.pose)

		if ((abs(self.pose[0]) < 1.) and (abs(self.pose[1]) < 1.)):

			if(abs(self.pose[0]-self.target[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(self.pose[1]-self.target[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')
			else:
				reward = self.get_reward()	
		else :
			done = True
			reward = -1.
			print("Outside range")

		return np.array(self.pose), reward, done, info     

	def render(self):
		print("Rendering")
		show_animation = True
		if show_animation:  # pragma: no cover
			plt.cla()
			# for stopping simulation with the esc key.
			plt.gcf().canvas.mpl_connect('key_release_event',
					lambda event: [exit(0) if event.key == 'escape' else None])
			plt.plot(self.pose[0]*MAX_X, self.pose[1]*MAX_Y, "ob", label="trajectory")
			plt.plot(self.target[0]*MAX_X, self.target[1]*MAX_Y, "xg", label="target")
			# self.plot_car()
			plt.axis("equal")
			plt.grid(True)
			# plt.title("Time[s]:" + str(np.round(time, 2)) + ", speed[km/h]:" + str(np.round(self.action[0] * 3.6, 2)))
			plt.pause(0.0001)
		pass

	def close(self):
		pass

	def update_state(self, state, a, DT):
		# print("Updating state")
		throttle = a[0]
		steer = a[1]
		# input check
		if steer >= MAX_STEER:
			steer = MAX_STEER
		elif steer <= -MAX_STEER:
			steer = -MAX_STEER

		if throttle > MAX_SPEED:
			throttle = MAX_SPEED
		elif throttle < MIN_SPEED:
			throttle = MIN_SPEED

		state[0] = state[0] + throttle * math.cos(state[2]) * DT
		state[1] = state[1] + throttle * math.sin(state[2]) * DT
		state[2] = state[2] + throttle / WB * math.tan(steer) * DT

		# state.v = state.v + a * DT	

		return state

	# def plot_car(self, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
	# 	print("Plotting Car")
	# 	x = self.pose[0]
	# 	y = self.pose[1]
	# 	yaw = self.pose[2]
	# 	steer = self.action[1]


	# 	outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
	# 						[WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

	# 	fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
	# 						 [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

	# 	rr_wheel = np.copy(fr_wheel)

	# 	fl_wheel = np.copy(fr_wheel)
	# 	fl_wheel[1, :] *= -1
	# 	rl_wheel = np.copy(rr_wheel)
	# 	rl_wheel[1, :] *= -1

	# 	Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
	# 					 [-math.sin(yaw), math.cos(yaw)]])
	# 	Rot2 = np.array([[math.cos(steer), math.sin(steer)],
	# 					 [-math.sin(steer), math.cos(steer)]])

	# 	fr_wheel = (fr_wheel.T.dot(Rot2)).T
	# 	fl_wheel = (fl_wheel.T.dot(Rot2)).T
	# 	fr_wheel[0, :] += WB
	# 	fl_wheel[0, :] += WB

	# 	fr_wheel = (fr_wheel.T.dot(Rot1)).T
	# 	fl_wheel = (fl_wheel.T.dot(Rot1)).T

	# 	outline = (outline.T.dot(Rot1)).T
	# 	rr_wheel = (rr_wheel.T.dot(Rot1)).T
	# 	rl_wheel = (rl_wheel.T.dot(Rot1)).T

	# 	outline[0, :] += x
	# 	outline[1, :] += y
	# 	fr_wheel[0, :] += x
	# 	fr_wheel[1, :] += y
	# 	rr_wheel[0, :] += x
	# 	rr_wheel[1, :] += y
	# 	fl_wheel[0, :] += x
	# 	fl_wheel[1, :] += y
	# 	rl_wheel[0, :] += x
	# 	rl_wheel[1, :] += y

	# 	plt.plot(np.array(outline[0, :]).flatten(),
	# 			 np.array(outline[1, :]).flatten(), truckcolor)
	# 	plt.plot(np.array(fr_wheel[0, :]).flatten(),
	# 			 np.array(fr_wheel[1, :]).flatten(), truckcolor)
	# 	plt.plot(np.array(rr_wheel[0, :]).flatten(),
	# 			 np.array(rr_wheel[1, :]).flatten(), truckcolor)
	# 	plt.plot(np.array(fl_wheel[0, :]).flatten(),
	# 			 np.array(fl_wheel[1, :]).flatten(), truckcolor)
	# 	plt.plot(np.array(rl_wheel[0, :]).flatten(),
	# 			 np.array(rl_wheel[1, :]).flatten(), truckcolor)
	# 	plt.plot(x, y, "*") 

def plot_car(n_state, action, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
	print("Plotting Car")

	x = n_state[0] #self.pose[0]
	y = n_state[1] #self.pose[1]
	yaw = n_state[2] #self.pose[2]
	steer = action[1] #self.action[1]

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

	start_point = [0., 0., 1.57]
	target_point = [0., 5., 1.57]
	env =  DeepracerGym(start_point, target_point)
	max_steps = int(1e6)

	state = env.reset()
	env.render()

	x = [state[0]]
	y = [state[1]]
	yaw = [state[2]]

	for i in range(max_steps):
		action = [1.0, -0.8]
		n_state,reward,done,info = env.step(action)
		x.append(n_state[0]*MAX_X)
		y.append(n_state[1]*MAX_Y)
		yaw.append(n_state[2])
		# env.render()
		# plot_car(n_state, action)
  		# pragma: no cover
		# plt.cla()
		# for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect('key_release_event',
				lambda event: [exit(0) if event.key == 'escape' else None])
		# plt.plot(n_state[0]*MAX_X, n_state[1]*MAX_Y, "ob", markersize = 2, label="trajectory")
		plt.plot(x*10, y*10, "ob", markersize = 2, label="trajectory")
		plt.plot(target_point[0], target_point[1], "xg", label="target")
		# plot_car(n_state, action)
		plt.axis("equal")
		plt.grid(True)
		# plt.title("Time[s]:" + str(np.round(time, 2)) + ", speed[km/h]:" + str(np.round(self.action[0] * 3.6, 2)))
		plt.pause(0.0001)
		if done:
			state = env.reset()                   
			break

if __name__ == '__main__':
	main()