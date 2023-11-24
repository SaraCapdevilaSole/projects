#!/usr/bin/env python
# coding: utf-8
# %%
import random
from collections import deque

import torch
import torch.nn.functional as F 
import torch.optim as optim 
from torch import nn
from gym.core import Env
import gym

import math
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time
import optuna ### hyperparameter optimisation
import matplotlib 


# %%
class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)


# %%
class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int], activation=None):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
            activation: single activation function or list of activation functions (ReLU by default)
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
        self.activation = activation
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer, activation in zip(self.layers[:-1], self.activation): #self.layers[:-1]
            x = activation(layer(x))
            
        x = self.layers[-1](x)
        return x


# %%
class DQN_Experiments(DQN, ReplayBuffer):
    def __init__(self, 
                 EPSILON, 
                 EPSILON_FUNC, 
                 OPTIMISER_TYPE, 
                 ACTIVATION_TYPE,
                 N_BUFFER=1, 
                 LR=1.,
                 N_BATCH=1,
                 HIDDEN_LAYERS=[4,2],
                 EP_TG_UPDATE=1,
                 #TAU=1e-2,
                 NUM_RUNS=10, ## fixed
                 EPISODES=300):
        
        self.N_BUFFER = N_BUFFER
        self.N_BATCH = N_BATCH
        self.LR = LR
        self.NET_LAYERS =  [4] + HIDDEN_LAYERS + [2]
        self.EPSILON = EPSILON
        self.OPTIMISER_TYPE = OPTIMISER_TYPE
        self.ACTIVATION_TYPE = ACTIVATION_TYPE
        self.EPSILON_FUNC = EPSILON_FUNC
        self.activation = self._set_activation()
        self.epsilon_func = self._set_epsilon_func()
        self.EP_TG_UPDATE = EP_TG_UPDATE
        self.NUM_RUNS = NUM_RUNS
        self.EPISODES = EPISODES
        self.run_results = []
        self.policy_net = []
        
        DQN.__init__(self, layer_sizes=self.NET_LAYERS, activation=self.activation)
        ReplayBuffer.__init__(self, size=N_BUFFER)
        
    def DQN_train(self):
        env = gym.make('CartPole-v1')
        for run in tqdm(range(self.NUM_RUNS)):
            policy_net = DQN(self.NET_LAYERS, self.activation)
            target_net = DQN(self.NET_LAYERS, self.activation)
            self._update_target(target_net, policy_net)
            target_net.eval()

            #optimiser = optim.SGD(policy_net.parameters(), lr=1.)
            optimiser = self._set_optimiser(policy_params=policy_net.parameters())
            memory = ReplayBuffer(self.N_BUFFER)

            episode_durations = []

            for i_episode in range(self.EPISODES):
                epsilon = self.epsilon_func(self.EPSILON, i_episode)
                observation, info = env.reset()
                state = torch.tensor(observation).float()

                done = False
                terminated = False
                t = 0
                while not (done or terminated):

                    # Select and perform an action
                    action = self._epsilon_greedy(epsilon, policy_net, state)

                    observation, reward, done, terminated, info = env.step(action)
                    reward = torch.tensor([reward])
                    action = torch.tensor([action])
                    next_state = torch.tensor(observation).reshape(-1).float()

                    memory.push([state, action, next_state, reward, torch.tensor([done])])

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    if not len(memory.buffer) < self.N_BATCH:
                        transitions = memory.sample(self.N_BATCH)
                        state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                        # Compute loss
                        mse_loss = self._loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                        # Optimize the model
                        optimiser.zero_grad()
                        mse_loss.backward()
                        optimiser.step()

                    if done or terminated:
                        episode_durations.append(t + 1)
                    t += 1
                # Update the target network in episode EP_TG_UPDATE, copying all weights and biases in DQN
                #soft_update_target(target_net,policy_net,tau=TAU)
                if i_episode % self.EP_TG_UPDATE == 0: 
                    self._update_target(target_net, policy_net)
            self.run_results.append(episode_durations)
            self.policy_net.append(policy_net)
    
    def get_run_results(self):
        return self.run_results
    
    def get_policy_net(self):
        return self.policy_net
        
    def _set_optimiser(self, policy_params): 
        optim_types = ["SGD", "Adam", "NAdam", "RAdam", "SGD momentum"] #optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','SGD']
        assert self.OPTIMISER_TYPE in optim_types, "Unknown Optimiser Type {}".format(self.OPTIMISER_TYPE)
        if self.OPTIMISER_TYPE == optim_types[0]:
            optimiser = optim.SGD(policy_params, lr=self.LR) ### weight decay??? and other params
        elif self.OPTIMISER_TYPE == optim_types[1]:
            optimiser = optim.Adam(policy_params, lr=self.LR)
        elif self.OPTIMISER_TYPE == optim_types[2]:
            optimiser = optim.NAdam(policy_params, lr=self.LR)
        elif self.OPTIMISER_TYPE == optim_types[3]:
            optimiser = optim.RAdam(policy_params, lr=self.LR)
        elif self.OPTIMISER_TYPE == optim_types[4]:
            optimiser = optim.SGD(policy_params, lr=self.LR, momentum=0.9)
        return optimiser

    def _set_activation(self):        
        #If a single activation function is provided, apply same activation function to all layers
        if not isinstance(self.ACTIVATION_TYPE, list):
            self.ACTIVATION_TYPE = [self.ACTIVATION_TYPE] * (len(self.NET_LAYERS) - 2)
            
        assert len(self.ACTIVATION_TYPE) == len(self.NET_LAYERS) - 2, "size of activation list should be {}".format(len(self.NET_LAYERS) - 2)
            
        act_types = ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "Softmax"] 
        activation_list = []
        for activation_name in self.ACTIVATION_TYPE:
            assert activation_name in act_types, "Unknown Activation Type {}".format(activation_name)
            if activation_name == act_types[0]:
                 activation = F.relu
            elif activation_name == act_types[1]:
                activation = F.sigmoid
            elif activation_name == act_types[2]:
                activation = F.tanh
            elif activation_name == act_types[3]:
                activation = F.leaky_relu
            elif activation_name == act_types[4]:
                activation = F.softmax
            activation_list.append(activation)
        return activation_list

    def _set_epsilon_func(self):
        epsilon_funcs = ["exponential" ,"linear", "cosh", "const."]
        assert self.EPSILON_FUNC in epsilon_funcs, "Unknown Epsilon function {}".format(self.EPSILON_FUNC)
        if self.EPSILON_FUNC == epsilon_funcs[0]: # exponential annealing
            epsilon = lambda x, k: x**(k/300) 
        elif self.EPSILON_FUNC == epsilon_funcs[1]: # linear annealing
            epsilon = lambda x, k: x + (0.01 - x) * (k/300) 
        elif self.EPSILON_FUNC == epsilon_funcs[2]: # cosh decay
            epsilon = lambda x, k: 1.1-(1/np.cosh(math.exp(-(k-abs(x-0.35)*300)/(0.1*300)))+(k*0.1/300))
        elif self.EPSILON_FUNC == epsilon_funcs[3]:
            epsilon = lambda x, k: x # constant
        return epsilon
    
    @staticmethod
    def _loss(policy_dqn:DQN, target_dqn:DQN,
              states:torch.Tensor, actions:torch.Tensor,
              rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
        """Calculate Bellman error loss

        Args:
            policy_dqn: policy DQN
            target_dqn: target DQN
            states: batched state tensor
            actions: batched action tensor
            rewards: batched rewards tensor
            next_states: batched next states tensor
            dones: batched Boolean tensor, True when episode terminates

        Returns:
            Float scalar tensor with loss value
        """

        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
        q_values = policy_dqn(states).gather(1, actions).reshape(-1)
        return ((q_values - bellman_targets)**2).mean()
    
    @staticmethod
    def _update_target(target_dqn:DQN, policy_dqn:DQN):
        """Update target network parameters using policy network.
        Does not return anything but modifies the target network passed as parameter

        Args:
            target_dqn: target network to be modified in-place
            policy_dqn: the DQN that selects the action
        """
        target_dqn.load_state_dict(policy_dqn.state_dict())
    
    @staticmethod
    def  _greedy_action(dqn:DQN, state:torch.Tensor)->int:
        """Select action according to a given DQN

        Args:
            dqn: the DQN that selects the action
            state: state at which the action is chosen

        Returns:
            Greedy action according to DQN
        """
        return int(torch.argmax(dqn(state)))
    
    @staticmethod
    def _epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
        """Sample an epsilon-greedy action according to a given DQN

        Args:
            epsilon: parameter for epsilon-greedy action selection
            dqn: the DQN that selects the action
            state: state at which the action is chosen

        Returns:
            Sampled epsilon-greedy action
        """
        q_values = dqn(state)
        num_actions = q_values.shape[0]
        greedy_act = int(torch.argmax(q_values))
        p = float(torch.rand(1))
        if p>epsilon:
            return greedy_act
        else:
            return random.randint(0,num_actions-1)


# %%
class Plotting_DQN:
    @staticmethod
    def plotQ1_both_DQN_results(DQN_result, DQN_unopt=None):
        results = torch.tensor(DQN_result)
        means = results.float().mean(0)
        stds = results.float().std(0)

        plt.figure(figsize=(10,6), dpi=600)
        plt.plot(torch.arange(300), means, color='g', label="Optimised DQN")
        plt.fill_between(np.arange(300), means, means+stds, alpha=0.3, color='g')
        plt.fill_between(np.arange(300), means, means-stds, alpha=0.3, color='g')

        if DQN_unopt: 
            results_unopt = torch.tensor(DQN_unopt)
            means_unopt = results_unopt.float().mean(0)
            stds_unopt = results_unopt.float().std(0)
            plt.plot(torch.arange(300), means_unopt, color='#1A73CD', label="Unoptimised DQN")
            plt.fill_between(np.arange(300), means_unopt, means_unopt+stds_unopt, alpha=0.3, color='b')
            plt.fill_between(np.arange(300), means_unopt, means_unopt-stds_unopt, alpha=0.3, color='b')

        plt.plot([0,300], [100,100], '--', color='r', linewidth=0.8, label="Return threshold")
        plt.ylabel("Return [au]", fontsize=15)
        plt.xlabel("Episodes [au]", fontsize=15)
        plt.legend(loc='upper left')
        plt.show()
        
    @staticmethod
    def plot_DQN_policy(policy_net, x=0, v=0, q=True, angle_range=.2095, omega_range=1,angle_samples=100, omega_samples=100, ax=None):
        """Plots the policy of the DQN in 2d plot displaying pole 
        angular velocity on the y-axis against pole angle on the x-axis
        Args:
            policy_net: list of DQN policy for all the runs
            x: the cart position
            v: the cart velocity
            q: Boolean -  whether q values or greedy policy is visualised
            angle_range, omega_range: the range of the pole angle and its angular velocity 
                                    # environment stops once this angle=.2095 is reached
            angle_samples, omega_samples: number of samples to plot
            ax: Matplotlib axis to plot on
        """
        angles = torch.linspace(angle_range, -angle_range, angle_samples)
        omegas = torch.linspace(-omega_range, omega_range, omega_samples)

        greedy_q_array = torch.zeros((angle_samples, omega_samples), dtype=torch.int32)
        policy_array = torch.zeros((angle_samples, omega_samples), dtype=torch.int32)
        for i, angle in enumerate(angles):
            for j, omega in enumerate(omegas):
                state = torch.tensor([x, v, angle, omega]) # x, dx/dy, theta, omega
                with torch.no_grad():
                    # find mean of q values
                    q_vals_list = []
                    for pn in policy_net:
                        q_vals_list.append(pn(state))

                    q_vals_tensor = torch.stack(q_vals_list)

                    first_column = torch.mean(q_vals_tensor[:, 0], dim=0, keepdim=True)
                    second_column = torch.mean(q_vals_tensor[:, 1], dim=0, keepdim=True)

                    q_vals = torch.cat([first_column, second_column], dim=0)

                    greedy_action = q_vals.argmax()
                    greedy_q_array[i, j] = q_vals[greedy_action]
                    policy_array[i, j] = greedy_action

        if ax is None:
            fig, ax = plt.subplots(dpi=600)
        ax.tick_params(axis='both', labelsize=15)
        if q:
            cf = ax.contourf(angles, omegas, greedy_q_array.T, cmap='cividis', levels=100)
            cbar = plt.colorbar(cf, ax=ax, orientation='vertical')
            cbar.set_label('Q-values', fontsize=20) 
            cbar.ax.tick_params(labelsize=20)
        else:
            cf = ax.contourf(angles, omegas, policy_array.T, cmap='cividis')
            color1_patch = mpatches.Patch(color=cf.cmap(0.0), label='L')
            color2_patch = mpatches.Patch(color=cf.cmap(1.0), label='R')
            ax.legend(handles=[color1_patch, color2_patch], loc='upper right')

        if ax is None:
            plt.show()

        ax.set_xlabel(r"$\theta$ (rad)", fontsize=20)
        ax.set_ylabel(r"$\omega$ (rad/s)", fontsize=20)

        # action 0: left
        # action 1: right
    
    @staticmethod
    def plot_q2_policy(policy_net, v_values, x=0, q=True, angle_range=.2095, omega_range=1, angle_samples=100, omega_samples=100):
        """
        Plot the greedy policy according to your DQN in 4 separate two dimensional plots displaying pole 
        angular velocity on the y-axis against pole angle on the x-axis
        """
        # Create a figure with 1x4 subplots
        fig, axs = plt.subplots(1, 4, figsize=(5*5, 5), dpi=600)

        for i, v_val in enumerate(v_values):
            ax = axs[i] #[i // 2, i % 2]  # Row, Column

            Plotting_DQN.plot_DQN_policy(policy_net, v=v_val, ax=ax, x=x, q=q, angle_range=angle_range, omega_range=omega_range, angle_samples=angle_samples, omega_samples=omega_samples)
            ax.set_title(f'v = {v_val} m/s', fontsize=18)

        plt.tight_layout()
        plt.show()


# %%
## Hyperparameter exploration functions
class Hyperparameter_exploration:
    ############## HIDDEN LAYERS ##############
    @staticmethod
    def h_layer_variation(layers): 
        """
        Args: 
            layers: list of hidden layers to experiment

        Returns:
            Results of runs
        """
        results = []
        for h_layer in layers:
            DQN_exp = DQN_Experiments(EPSILON=0.5, 
                                      EPSILON_FUNC="linear",
                                      OPTIMISER_TYPE="Adam", 
                                      ACTIVATION_TYPE="LeakyRelu",
                                      N_BUFFER=2000, 
                                      LR=0.001,
                                      N_BATCH=20,
                                      HIDDEN_LAYERS=h_layer,
                                      EP_TG_UPDATE=8,
                                      NUM_RUNS=10)
            DQN_exp.DQN_train()
            DQN_result = DQN_exp.get_run_results()
            results.append(DQN_result)
        return results
    
    @staticmethod
    def plot_h_layer_variation(layers):
        assert len(layers)<=4, "Please input MAX 4 types of hidden layers"
        print("Running experiments with layers: {}".format(layers))
        results_h_layer = Hyperparameter_exploration.h_layer_variation(layers)
        print("Plotting results:")
        colours = ["#659B00", "#F49031", "#009999", "#A423DC"]
        plt.figure(dpi=600, figsize=(10,6))
        for i, result_i in enumerate(results_h_layer):
            layer = [4] + layers[i] + [2]
            means = torch.tensor(result_i).float().mean(0)
            plt.plot(means, label=layer, linewidth=0.8, color=colours[i])
        plt.plot([0,300], [100,100], '--', color='r', linewidth=0.8, label="Return threshold")
        plt.ylabel("Return [au]", fontsize=15)
        plt.xlabel("Episodes [au]", fontsize=15)
        plt.legend(loc='upper left')
        
    ############## EPISLON_FUNC ##############
    @staticmethod
    def epsilon_variation(epsilon_func, epsilon): 
        """
        Args: 
            epsilon_func: epsilon function
            epislon: epsilon value

        Returns:
            Results of runs
        """
        results = []
        for epsilon_func_i in epsilon_func:
            results_epsilon = []
            for epsilon_i in epsilon:
                DQN_exp = DQN_Experiments(EPSILON=epsilon_i, 
                                          EPSILON_FUNC=epsilon_func_i,
                                          OPTIMISER_TYPE="Adam", 
                                          ACTIVATION_TYPE="LeakyRelu",
                                          N_BUFFER=2000, 
                                          LR=0.001,
                                          N_BATCH=20,
                                          HIDDEN_LAYERS=[32, 16],
                                          EP_TG_UPDATE=8)
                DQN_exp.DQN_train()
                DQN_result = DQN_exp.get_run_results()
                results_epsilon.append(DQN_result)
            results.append(results_epsilon)
        return results
    
    @staticmethod
    def plot_epsilon_variation(epsilon_func, epsilon, eps_results=None):
        assert len(epsilon)<=4, "Please input MAX 4 different epsilons"
        if eps_results==None:
            print("Running {} experiments with epsilon values: {}".format(int(len(epsilon)*len(epsilon_func)), epsilon))
            eps_results = Hyperparameter_exploration.epsilon_variation(epsilon_func, epsilon)
        print("Plotting results:")
        colours = ["#659B00", "#F49031", "#A423DC", "#009999"]
        fig, axes = plt.subplots(nrows=len(epsilon_func), ncols=1, dpi=600, figsize=(10, 6 * len(epsilon_func)))
        
        for i, epsilon_func_i in enumerate(epsilon_func):
            if len(epsilon_func) == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.set_title(f'$\epsilon$ function: {epsilon_func[i]}', fontsize=18)
            for j, epsilon_i in enumerate(epsilon):
                epsilon_func_results = eps_results[i][j]
                results = torch.tensor(epsilon_func_results)
                means = results.float().mean(0)
                stds = results.float().std(0)
                ax.plot(torch.arange(300), means, label=epsilon_i, linewidth=0.8, color=colours[j])
                ax.set_ylabel("Return [au]", fontsize=18)
                ax.set_xlabel("Episodes [au]", fontsize=18)
            ax.plot([0, 300], [100, 100], '--', color='r', linewidth=0.8, label="Return threshold") 
            ax.legend(loc='upper left')    
        plt.tight_layout()
        plt.show()
    
    ############## OPTIMISER ############## 
    @staticmethod
    def optimiser_variation(optimiser_types): 
        """
        Args: 
            optimiser_types: names of optimisers

        Returns:
            Results of runs
        """
        results = []
        for optimiser in optimiser_types:
            DQN_exp = DQN_Experiments(EPSILON=0.5, 
                                      EPSILON_FUNC="linear",
                                      OPTIMISER_TYPE=optimiser, 
                                      ACTIVATION_TYPE="Relu",
                                      N_BUFFER=2000, 
                                      LR=0.001,
                                      N_BATCH=20,
                                      HIDDEN_LAYERS=[32, 16],
                                      EP_TG_UPDATE=8)
            DQN_exp.DQN_train()
            DQN_result = DQN_exp.get_run_results()
            results.append(DQN_result)
        return results
    
    @staticmethod
    def plot_optimiser_variation(optimiser_types, optimiser_results=None):
        assert len(optimiser_types)<=5, "Please input MAX 5 different optimiser types"
        if optimiser_results==None:
            print("Running experiments with optimiser_types: {}".format(optimiser_types))
            optimiser_results = Hyperparameter_exploration.optimiser_variation(optimiser_types)
        print("Plotting results:")
        colours = ["#659B00", "#F49031", "#009999", "#A423DC", "#E65024"]
        plt.figure(dpi=600, figsize=(10,6))
        for i, result_i in enumerate(optimiser_results):
            means = torch.tensor(result_i).float().mean(0)
            plt.plot(means, label=optimiser_types[i], linewidth=0.8, color=colours[i])
            plt.plot([0,300], [100,100], '--', color='r', linewidth=0.8, label="Return threshold")
        plt.ylabel("Return [au]", fontsize=15)
        plt.xlabel("Episodes [au]", fontsize=15)
        plt.legend(loc='upper left')
        
    ############## ACTIVATION FUNCTION ############## 
    @staticmethod
    def activation_variation(activation_types): 
        """
        Args: 
            optimiser_types: names of optimisers

        Returns:
            Results of runs
        """
        results = []
        for activation in activation_types:
            DQN_exp = DQN_Experiments(EPSILON=0.5, 
                                      EPSILON_FUNC="linear",
                                      OPTIMISER_TYPE="Adam", 
                                      ACTIVATION_TYPE=activation,
                                      N_BUFFER=2000, 
                                      LR=0.001,
                                      N_BATCH=20,
                                      HIDDEN_LAYERS=[32, 16],
                                      EP_TG_UPDATE=8)
            DQN_exp.DQN_train()
            DQN_result = DQN_exp.get_run_results()
            results.append(DQN_result)
        return results
    
    @staticmethod
    def plot_activation_variation(activation_types, activation_results=None):
        assert len(activation_types)<=5, "Please input MAX 5 different activation types"
        if activation_results==None:
            print("Running experiments with activation_types: {}".format(activation_types))
            activation_results = Hyperparameter_exploration.activation_variation(activation_types)
        print("Plotting results:")
        colours = ["#659B00", "#F49031", "#009999", "#A423DC", "#E65024"]
        plt.figure(dpi=600, figsize=(10,6))
        for i, result_i in enumerate(activation_results):
            means = torch.tensor(result_i).float().mean(0)
            plt.plot(means, label=activation_types[i], linewidth=0.8, color=colours[i])
            plt.plot([0,300], [100,100], '--', color='r', linewidth=0.8, label="Return threshold")
        plt.ylabel("Return [au]", fontsize=15)
        plt.xlabel("Episodes [au]", fontsize=15)
        plt.legend(loc='upper left')
        
    ############## BUFFER SIZE ############## - + lit
    @staticmethod
    def buffer_variation(buffer_types): 
        """
        Args: 
            optimiser_types: names of optimisers

        Returns:
            Results of runs
        """
        results = []
        times = []
        for buffer in buffer_types:
            DQN_exp = DQN_Experiments(EPSILON=0.5, 
                                      EPSILON_FUNC="linear",
                                      OPTIMISER_TYPE="Adam", 
                                      ACTIVATION_TYPE="LeakyRelu",
                                      N_BUFFER=buffer, 
                                      LR=0.001,
                                      N_BATCH=20,
                                      HIDDEN_LAYERS=[32, 16],
                                      EP_TG_UPDATE=8)
            start_time = time.time()
            DQN_exp.DQN_train()
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            DQN_result = DQN_exp.get_run_results()
            results.append(DQN_result)
            times.append(elapsed_time)
        return results, times
    
    @staticmethod
    def plot_buffer_variation(buffer_types, buffer_results=None, elapsed_times=None, plot_results=True, plot_complexity=True):
        assert len(buffer_types)<=5, "Please input MAX 5 different buffer types"
        if buffer_results==None:
            print("Running experiments with buffer_types: {}".format(buffer_types))
            buffer_results, elapsed_times = Hyperparameter_exploration.buffer_variation(buffer_types)
        
        colours = ["#659B00", "#F49031", "#009999", "#A423DC", "#E65024"]
        if plot_results:
            print("Plotting results:")
            
            plt.figure(dpi=600, figsize=(10,6))
            for i, result_i in enumerate(buffer_results):
                means = torch.tensor(result_i).float().mean(0)
                plt.plot(means, label=buffer_types[i], linewidth=0.8, color=colours[i])
                plt.plot([0,300], [100,100], '--', color='r', linewidth=0.8, label="Return threshold")
            plt.ylabel("Return [au]", fontsize=15)
            plt.xlabel("Episodes [au]", fontsize=15)
            plt.legend(loc='upper left')
        if plot_complexity:
            print("plotting complexity:")
            scatter_color = 'red'
            text_offset = (25, -4)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=600, sharex=False)

            # Plot the first scatter plot
            ax1.set_ylabel("Mean Score [au]")
            ax1.grid()
            for i, br in enumerate(buffer_results):
                counts = Optuna_Study.returns_above_100(br)
                ax1.scatter(elapsed_times[i] / 10, counts, color=scatter_color, s=20)
                ax1.annotate(
                    str(buffer_types[i]),
                    (elapsed_times[i] / 10, counts),
                    textcoords="offset points",
                    xytext=text_offset,
                    ha='center'
                )

            # Plot the second scatter plot
            ax2.set_xlabel("Mean Time per run (s)")
            ax1.set_xlabel("Mean Time per run (s)")
            ax2.set_ylabel("Standard Deviation of Score [au]")
            ax2.grid()
            for i, br in enumerate(buffer_results):
                std = Optuna_Study.returns_std_above_100(br)
                ax2.scatter(elapsed_times[i] / 10, std, color=scatter_color, s=20)
                ax2.annotate(
                    str(buffer_types[i]),
                    (elapsed_times[i] / 10, std),
                    textcoords="offset points",
                    xytext=text_offset,
                    ha='center'
                )

            plt.show()
    
    ############## BATCH SIZE ############## - lit
    ############## LEARNING RATE ############## - lit
    ############## EPISODES TO UPDATE ############## - lit


# %%
class Optuna_Study:
    def train_evaluate(params_nn):
        """Runs DQN and returns corresponding value, with given set of parameters

        Args:
            params_nn: parameters for DQN

        Returns:
             count of returns above 100, std of returns above 100
        """
        DQN_exp = DQN_Experiments(EPSILON=params_nn['EPSILON'], 
                                  EPSILON_FUNC=params_nn['EPSILON_FUNC'], 
                                  OPTIMISER_TYPE=params_nn['OPTIMISER_TYPE'], 
                                  ACTIVATION_TYPE=params_nn['ACTIVATION_TYPE'], 
                                  N_BUFFER=params_nn['N_BUFFER'], 
                                  LR=params_nn['LR'], 
                                  N_BATCH=params_nn['N_BATCH'], 
                                  HIDDEN_LAYERS=params_nn['HIDDEN_LAYERS'],
                                  EP_TG_UPDATE=params_nn['EP_TG_UPDATE'])
        
        DQN_exp.DQN_train()
        DQN_result = DQN_exp.get_run_results()
        count_returns = Optuna_Study.returns_above_100(DQN_result)
        std_returns = Optuna_Study.returns_std_above_100(DQN_result)
        return count_returns, std_returns
    
    def returns_above_100(DQN_result):
        """Counts number of returns have value greater than 100:  
        To find agent with reward of 100 over 10 runs of training for over 50 episodes

        Args:
            DQN_result: Output of DQN

        Returns:
             number of episodes for which agent has reward 100 over 10 runs of training 
        """
        results = torch.tensor(DQN_result)
        means = results.float().mean(0)
        return np.count_nonzero(means>100)
    
    def returns_std_above_100(DQN_result):
        """Counts number of returns - standard deviation have value of rewards greater than 100:  
        To find agent with most stable reward over 100 for 10 runs of training

        Args:
            DQN_result: Output of DQN
        Returns:
             count of standard deviation of the 10 runs of training above 100
        """
        results = torch.tensor(DQN_result)
        means = results.float().mean(0)
        stds = results.float().std(0)
        lower_lim = means-stds
        return np.count_nonzero(lower_lim>100)
    
    def objective(trial):
        h_layers = 2 #trial.suggest_int('h_layers', 1, 2)
        params_nn ={
            'EPSILON': trial.suggest_categorical('EPSILON', [0.2 + 0.1*i for i in range(8)]),
            'EPSILON_FUNC': "linear", #trial.suggest_categorical('EPSILON_FUNC', ["exponential" ,"linear", "cosh"]),  
            'OPTIMISER_TYPE': "NAdam", #trial.suggest_categorical('OPTIMISER_TYPE', ["SGD", "Adam", "RAdam"]),
            'ACTIVATION_TYPE': "LeakyRelu",
            'N_BUFFER': trial.suggest_categorical('N_BUFFER', [int(10000*i) for i in range(1,7)]), 
            'LR': trial.suggest_categorical('LR', [1e-3*10**i for i in range(3)]),
            'N_BATCH': trial.suggest_categorical('N_BATCH', [10*i for i in range(1,7)]), #10
            'HIDDEN_LAYERS': [trial.suggest_categorical(f'layer_{i}_size', [40 + i*20 for i in range(9)]) for i in range(h_layers)], #trial.suggest_categorical('NET_LAYERS', suggest_nn_structure())
            'EP_TG_UPDATE': trial.suggest_categorical('EP_TG_UPDATE', [2*i for i in range(1, 4)])
            }
        return Optuna_Study.train_evaluate(params_nn)
    
    def start_study(n_trials=20):
        study = optuna.create_study(study_name="Hyperparameter Optimisation", directions=['maximize', 'maximize'])
        study.optimize(Optuna_Study.objective, n_trials=n_trials)

        print("Best study:")
        best_trials = study.best_trials

        best_count =  ({}, [0, 0], 0)  # trial with best count: params, values and trial number
        best_std = ({}, [0, 0], 0)  # trial with best std: params, values and trial number

        for bt in best_trials:
            if bt.values[0] > best_count[1][0]: # compare count
                best_count = (bt.params, bt.values, bt.number)
            if bt.values[1] > best_std[1][1]: # compare std
                best_std = (bt.params, bt.values, bt.number)   

        assert best_count[0] and best_std != {}, "Run trial again; no optimal parameters found"
        return best_count, best_std

# %%
