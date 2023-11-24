#!/usr/bin/env python
# coding: utf-8

# In[582]:


import numpy as np 
import random
import matplotlib.pyplot as plt # Graphical library
#from sklearn.metrics import mean_squared_error # Mean-squared error function


# # Coursework 1 :
# See pdf for instructions. 

# In[2]:


# WARNING: fill in these two functions that will be used by the auto-marking script
# [Action required]

def get_CID():
  return "01727810" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

def get_login():
  return "sc3719" # Return your short imperial login

def mean_squared_error(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  MSE = np.mean((y_true - y_pred) ** 2)
  return MSE # returns MSE between two arrays

# ## Helper class

# In[4]:


# This class is used ONLY for graphics
# YOU DO NOT NEED to understand it to work on this coursework

class GraphicsMaze(object):

  def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    # Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
      self.walls[ob] = 20

    # Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
      self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

    # Print the map to show it
    self.paint_maps()

  def paint_maps(self):
    """
    Print the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders)
    plt.show()

  def paint_state(self, state):
    """
    Print one state on the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    states = np.zeros(self.shape)
    states[state] = 30
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders + states)
    plt.show()

  def draw_deterministic_policy(self, Policy):
    """
    Draw a deterministic policy
    input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, action in enumerate(Policy):
      if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
        continue
      arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
      action_arrow = arrows[action] # Take the corresponding action
      location = self.locations[state] # Compute its location on graph
      plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    plt.show()

  def draw_policy(self, Policy):
    """
    Draw a policy (draw an arrow in the most probable direction)
    input: Policy {np.array} -- policy to draw as probability
    output: /
    """
    deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
    self.draw_deterministic_policy(deterministic_policy)

  def draw_value(self, Value):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, value in enumerate(Value):
      if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
        continue
      location = self.locations[state] # Compute the value location on graph
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    plt.show()

  def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple deterministic policies
    input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Policies)): # Go through all policies
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, action in enumerate(Policies[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
          continue
        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
        action_arrow = arrows[action] # Take the corresponding action
        location = self.locations[state] # Compute its location on graph
        plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
    plt.show()

  def draw_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policies (draw an arrow in the most probable direction)
    input: Policy {np.array} -- array of policies to draw as probability
    output: /
    """
    deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
    self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

  def draw_value_grid(self, Values, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policy values
    input: Values {np.array of np.array} -- array of policy values to draw
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Values)): # Go through all values
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, value in enumerate(Values[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
          continue
        location = self.locations[state] # Compute the value location on graph
        plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
    plt.show()


# ## Maze class

# In[177]:


# This class define the Maze environment

class Maze(object):

  # [Action required]
  def __init__(self, prob = None):
    """
    Maze initialisation.
    input: /
    output: /
    """
    
    # [Action required]
    # Properties set from the CID
    #self._CID = get_CID() #not sure if we can do this without input
    if prob is None:
        self._prob_success = 0.8 + 0.02 * (9 - 1) #float(self._CID[-2])) # float
    else:
        self._prob_success = prob
    self._gamma = 0.8 + 0.02 * 1 #float(self._CID[-2]) # float
    self._goal = 0 # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

    # Build the maze
    self._build_maze()
                              

  # Functions used to build the Maze environment 
  # You DO NOT NEED to modify them
  def _build_maze(self):
    """
    Maze initialisation.
    input: /
    output: /
    """

    # Properties of the maze
    self._shape = (13, 10)
    self._obstacle_locs = [
                          (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
                          (2,1), (2,2), (2,3), (2,7), \
                          (3,1), (3,2), (3,3), (3,7), \
                          (4,1), (4,7), \
                          (5,1), (5,7), \
                          (6,5), (6,6), (6,7), \
                          (8,0), \
                          (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
                          (10,0)
                         ] # Location of obstacles
    self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
    self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ]
    self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] #Reward of absorbing states
    self._default_reward = -1 # Reward for each action performs in the environment
    self._max_t = 500 # Max number of steps in the environment

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on
        
    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j) 
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4)) 
    
    for state in range(self._state_size):
      loc = self._get_loc_from_state(state)

      # North
      neighbour = (loc[0]-1, loc[1]) # North neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('N')] = state

      # East
      neighbour = (loc[0], loc[1]+1) # East neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('E')] = state

      # South
      neighbour = (loc[0]+1, loc[1]) # South neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('S')] = state

      # West
      neighbour = (loc[0], loc[1]-1) # West neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing
    self._absorbing = np.zeros((1, self._state_size))
    for a in self._absorbing_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = 1

    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
    for action in range(self._action_size):
      for outcome in range(4): # For each direction (N, E, S, W)
        # The agent has prob_success probability to go in the correct direction
        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
        # Equal probability to go into one of the other directions
        else:
          prob = (1.0 - self._prob_success) / 3.0
          
        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          # If absorbing state, probability of 0 to go to any other states
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome] # Post state number
            post_state = int(post_state) # Transform in integer to avoid error
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
    self._R = self._default_reward * self._R # Set default_reward everywhere
    for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
      post_state = self._get_state_from_loc(self._absorbing_locs[i])
      self._R[:,post_state,:] = self._absorbing_rewards[i]

    # Creating the graphical Maze world
    self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)
    
    # Reset the environment
    self.reset()


  def _is_location(self, loc):
    """
    Is the location a valid state (not out of Maze and not an obstacle)
    input: loc {tuple} -- location of the state
    output: _ {bool} -- is the location a valid state
    """
    if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
      return False
    elif (loc in self._obstacle_locs):
      return False
    else:
      return True


  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def _get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  # Getter functions used only for DP agents
  # You DO NOT NEED to modify them
  def get_T(self):
    return self._T

  def get_R(self):
    return self._R

  def get_absorbing(self):
    return self._absorbing

  # Getter functions used for DP, MC and TD agents
  # You DO NOT NEED to modify them
  def get_graphics(self):
    return self._graphics

  def get_action_size(self):
    return self._action_size

  def get_state_size(self):
    return self._state_size

  def get_gamma(self):
    return self._gamma

  # Functions used to perform episodes in the Maze environment
  def reset(self):
    """
    Reset the environment state to one of the possible starting states
    input: /
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
    self._reward = 0
    self._done = False
    return self._t, self._state, self._reward, self._done

  def step(self, action):
    """
    Perform an action in the environment
    input: action {int} -- action to perform
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Drawing a random number used for probaility of next state
    probability_success = random.uniform(0,1)

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0: 
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < probability_success) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."
    
    # Setting new t, state, reward and done
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done


# ## DP Agent

# In[1125]:


# This class define the Dynamic Programing agent 

class DP_agent(object):
    
    def __init__(self, gamma=None, threshold = 0.001):
        """
        Initialize the Monte Carlo agent.
        input:
            gamma {float, optional} -- Discount factor (default: None)
        """
        self.gamma = gamma
        self.threshold = threshold
        self.delta_values = []
        self.c = 0

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env, record_deltas=False):
        """
        Solve a given Maze environment using Dynamic Programming
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - V {np.array} -- Corresponding value function 
        """
        # Extract environment properties 
        T = env.get_T()
        R = env.get_R()
        absorbing = env.get_absorbing()
        state_size = env.get_state_size()
        action_size = env.get_action_size()

        # Initialisation
        if self.gamma is None:
            self.gamma = env.get_gamma()
                
        gamma = self.gamma
        threshold = self.threshold
        policy = np.zeros((state_size, action_size))
        V = np.zeros(state_size)
        self.c = 0 # count number of epochs

        while True:
            # Policy evaluation - predicting the optimal policy 
            while True:
                self.c += 1
                delta = 0 # initialise delta
                for s in range(state_size):
                    if absorbing[0, s] == 0: # if not in an absorbing state
                        v = np.copy(V[s])
                        V[s] = np.sum([policy[s, a] * np.sum([T[s, s_prime, a]*(R[s, s_prime, a] + gamma * V[s_prime]) 
                                       for s_prime in range(state_size)]) for a in range(action_size)])
                        delta = max(delta, abs(v - V[s]))
                
                # Append delta value to the list
                if record_deltas:
                    self.delta_values.append(delta)
                
                if delta < threshold:
                    break

            # Policy Improvement
            policy_stability = True
            for s in range(state_size):
                if absorbing[0, s] == 0: # if not in an absorbing state
                    old_policy = np.copy(policy[s])
                    a_array = [np.sum([T[s, s_prime, a]*(R[s, s_prime, a] + gamma * V[s_prime]) 
                                       for s_prime in range(state_size)]) for a in range(action_size)]
                    a_max = np.argmax(a_array) #np.random.choice(np.flatnonzero(a_array == np.max(a_array))) # with ties broken arbitrarily
                    policy[s] = np.zeros(action_size)
                    policy[s, a_max] = 1
                    if not np.array_equal(policy[s], old_policy):
                        policy_stability = False

            if policy_stability == True:
                break

        return policy, V
    
    def get_deltas(self):
        return self.delta_values
    
    def get_count(self):
        return self.c


# ## MC agent

# In[1164]:


# This class define the Monte-Carlo agent

class MC_agent(object):
    
    def __init__(self, gamma=None, num_episodes=3000, x=0.99, seed = 23, epsilon=None):
        """
        Initialize the Monte Carlo agent.
        input:
            gamma {float, optional} -- Discount factor (default: None)
        """
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.T = np.zeros(num_episodes) #record length of episodes
        self.total_rewards = np.zeros(num_episodes)
        self.seed = seed
        self.x = x # increase value of x for more exploration e.g. 0.999 (better for higher gamma)
        self.epsilon = epsilon
        
    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env):
        """
        Solve a given Maze environment using Monte Carlo learning (first visit MC)
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        note: this agent should not use: env.get_T(), env.get_R() or env.get_absorbing()
        """

        # Initialisation 
        if self.gamma is None:
                self.gamma = env.get_gamma()

        num_episodes = self.num_episodes
        gamma = self.gamma
        #np.random.seed(self.seed)
        epsilon_func = lambda k: self.epsilon if self.epsilon is not None else self.x ** k  # Lambda function for epsilon

        action_size = env.get_action_size()
        state_size = env.get_state_size()

        Q = np.zeros((state_size, action_size)) # arbitrary initialisation of Q
        Returns = np.zeros((state_size, action_size, num_episodes))
        V = np.zeros(state_size)
        policy = np.ones((state_size, action_size))/action_size # Default is a random probability matrix
        values = []

        for k in range(num_episodes):
            returns_count = np.zeros(state_size)
            returns_sum = np.zeros(state_size)
            epsilon = epsilon_func(k)
            episode_history = []
            done = True

            #Generate one episode following the policy
            while True:
                if done:  
                    t, state, reward, done = env.reset()
                else:     
                    t, state, reward, done = env.step(action) # action will update after 1st step
                    
                action = np.random.choice(action_size, p = policy[state, :])
                episode_history.append((t, state, reward, done, action))

                self.total_rewards[k] += reward # Total non-discounted reward for the episode (k)

                if done:
                    T = t
                    self.T[k] += T # record length of episode k
                    break

            G = 0
            for t in range(T-1, -1, -1): # array from T-1 to 0 at -1 timesteps
                t_, state, _, _, action = episode_history[t]
                _, _, reward_1, _, _ = episode_history[t + 1]
                assert t_ == t, "Time values do not match"

                G = gamma * G + reward_1

                if (state, action) not in [(episode_history[x][1], episode_history[x][-1]) for x in range(t)]:
                    Returns[state, action, k] += G 

                    Q[state, action] = sum(Returns[state, action, :])/(k+1) # mean of returns over all episodes
                    V[state] = np.max(Q[state, :]) # sum([Q[state, a]*policy[state, a] for a in range(action_size)])
                    
                    a_star = np.random.choice(np.flatnonzero(Q[state, :] == np.max(Q[state, :]))) # with ties broken arbitrarily
                    policy[state, :] = epsilon / action_size
                    policy[state, a_star] = 1 - epsilon + epsilon / action_size
            
            values.append(V.copy())

        return policy, values, self.total_rewards
    
    def get_ep_lengths(self):
        return self.T
        


# ## TD agent

# In[1010]:


# This class define the Temporal-Difference agent

class TD_agent(object):
    
    def __init__(self, gamma=None, num_episodes=3000, epsilon=None, alpha=None, x=0.999):
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.T = np.zeros(num_episodes) #record length of episodes
        self.total_rewards = np.zeros(num_episodes)
        self.x = x

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        note: this agent shouldn't use env.get_T(), env.get_R() or env.get_absorbing() 
        """
        # Initialisation
        if self.gamma is None:
                self.gamma = env.get_gamma()
                
        epsilon_func = lambda k: self.epsilon if self.epsilon is not None else self.x ** k  # Lambda function for epsilon
        alpha_func = lambda k: self.epsilon if self.epsilon is not None else self.x ** k  # Lambda function for alpha
       
        num_episodes = self.num_episodes
        gamma = self.gamma
        np.random.seed(22)

        action_size = env.get_action_size()
        state_size = env.get_state_size()

        Q = np.zeros((state_size, action_size))
        V = np.zeros(state_size)
        policy = np.zeros((env.get_state_size(), env.get_action_size())) 
        values = []
   

        for k in range(num_episodes):
            epsilon = epsilon_func(k)
            alpha = alpha_func(k)
            
            _, state, reward, _ = env.reset()

            while True:
                # epsilon gredy policy w.r.t to our Behavioural Policy
                a_star = np.random.choice(np.flatnonzero(Q[state, :] == np.max(Q[state, :]))) # with ties broken arbitrarily
                policy[state, :] = epsilon / action_size
                policy[state, a_star] = 1 - epsilon + epsilon / action_size
                action = np.random.choice(action_size, p = policy[state, :])

                V[state] = sum([Q[state, a]*policy[state, a] for a in range(action_size)]) #Q[state, action]

                t, next_state, reward, done = env.step(action)

                # Update Q-value based on action (and act greedily with input from optimal policy)
                Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

                self.total_rewards[k] += reward

                state = next_state  
                if done:
                    self.T[k] += t
                    break

            values.append(V.copy())

        return policy, values, self.total_rewards

    def get_ep_lengths(self):
        return self.T


# ## Example main

# In[188]:


if __name__ == "__main__":
    # Example main (can be edited)

    ### Question 0: Defining the environment

    print("Creating the Maze:\n")
    maze = Maze()


    ### Question 1: Dynamic programming

    dp_agent = DP_agent()
    dp_policy, dp_value = dp_agent.solve(maze)

    print("Results of the DP agent:\n")
    maze.get_graphics().draw_policy(dp_policy)
    maze.get_graphics().draw_value(dp_value)


    ### Question 2: Monte-Carlo learning

    mc_agent = MC_agent()
    mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)

    print("Results of the MC agent:\n")
    maze.get_graphics().draw_policy(mc_policy)
    maze.get_graphics().draw_value(mc_values[-1])


    ### Question 3: Temporal-Difference learning

    td_agent = TD_agent()
    td_policy, td_values, td_total_rewards = td_agent.solve(maze)

    print("Results of the TD agent:\n")
    maze.get_graphics().draw_policy(td_policy)
    maze.get_graphics().draw_value(td_values[-1])


# ## Q1: Dynamic programming

# ## Method, parametes, assumptions

# In[1127]:


# Method chosen and justification: 
# parameters to dynamic programming: gamma (CID specific), threshold (precision to cut)
# Assume will converge
# my_script.py

def threshold():
    mse_array = []
    counts_array = []
    dp_value_array = []
    dp_agent = DP_agent(threshold = 1e-20)
    dp_policy, dp_value_ref = dp_agent.solve(maze)
    
    threshold_array = [10**(-k) for k in range(-1, 10)]
    for th in threshold_array:
        dp_agent = DP_agent(threshold = th)
        dp_policy, dp_value = dp_agent.solve(maze)     
        dp_value_array.append(dp_value)
        mse_array.append(mean_squared_error(dp_value, dp_value_ref))
        counts_array.append(dp_agent.get_count())
    return threshold_array, dp_value_array, mse_array, counts_array

if __name__ == "__main__":
    threshold_array, dp_value_array, mse_array, counts_array = threshold()


# In[1141]:


if __name__ == "__main__":
    colours = ["#F0C076", "#82DE73", "#FD7777"]

    ax = plt.axes() 
    #plt.legend()
    #plt.minorticks_on()
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("log(MSE)", color = colours[2])

    ax1 = ax.twinx()

    ax1.plot(range(0,11), counts_array, color = colours[1], linewidth = 0.9)
    ax.plot(range(0,11), mse_array, color = colours[2], linewidth = 0.9)

    ax.set_xticks(range(0,11)) 
    ax.set_xticklabels(threshold_array)

    ax.set_yscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("epochs", color=colours[1])

    plt.show()

    print(mse_array, counts_array)


# ## Optimal Value Function

# In[1258]:


if __name__ == "__main__":    
    print("Creating the Maze:\n")
    maze = Maze()

    dp_agent = DP_agent()
    dp_policy, dp_value = dp_agent.solve(maze)

    print("Results of the DP agent:\n")
    maze.get_graphics().draw_policy(dp_policy)
    maze.get_graphics().draw_value(dp_value)


# ## The effect of prob_success on the Policy

# In[184]:


if __name__ == "__main__":   
    p = [0.15,0.25,0.35]
    for p_i in p:
        plt.figure(dpi=1200) 
        print('probability of success is', p_i)
        maze = Maze(prob = p_i) # initialise maze with probability of success
        dp_agent = DP_agent()
        dp_policy, dp_value = dp_agent.solve(maze)
        maze.get_graphics().draw_policy(dp_policy)
        maze.get_graphics().draw_value(dp_value)


# ## The effect of gamma on the Policy

# In[186]:


if __name__ == "__main__":
    g = [0.1, 0.9]
    maze = Maze() 
    for g_i in g:
        plt.figure(dpi=1200) 
        print('gamma is', g_i)
        dp_agent = DP_agent(gamma = g_i)
        dp_policy, dp_value = dp_agent.solve(maze)
        maze.get_graphics().draw_policy(dp_policy)
        maze.get_graphics().draw_value(dp_value)


# ## Q2: Monte Carlo

# ## Choice of parameters

# In[921]:


if __name__ == "__main__":  
    maze = Maze()
    epsilon_array = np.linspace(0.01,0.5,10)
    mse_array = np.zeros(len(epsilon_array))
    cv_array = np.zeros(len(epsilon_array))
    for i, eps_i in enumerate(epsilon_array):
        print(i)
        mc_agent = MC_agent(epsilon=eps_i)
        mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)
        mse_array[i] = mean_squared_error(mc_values[-1], dp_value)
        mean = np.mean(mc_total_rewards)
        std = np.std(mc_total_rewards)
        cv_array[i] = mean/std


# In[979]:


if __name__ == "__main__":
    mc_agent = MC_agent()
    mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)
    mse_value = mean_squared_error(mc_values[-1], dp_value)
    mean = np.mean(mc_total_rewards)
    std = np.std(mc_total_rewards)
    cv_value = mean/std


# In[939]:


if __name__ == "__main__": 
    print(mse_value, cv_value)
    mse_min = np.argmin(mse_array) # value of epsilon that minimises MSE
    cv_max = np.argmax(cv_array) # value of epsilon that maximises CV
    print(min(mse_array), cv_array[mse_min])
    print(mse_array[cv_max], max(cv_array))


# In[969]:


if __name__ == "__main__":  
    mc_agent = MC_agent(epsilon=epsilon_array[mse_min], gamma=1)
    _, _, mc_total_rewards_mse = mc_agent.solve(maze)

    mc_agent = MC_agent(epsilon=epsilon_array[cv_max], gamma=1)
    _, _, mc_total_rewards_cv = mc_agent.solve(maze)

    mc_agent = MC_agent(x=0.999, gamma=1)
    mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)


# In[972]:


if __name__ == "__main__": 
    plt.figure(figsize=(10,6), dpi=1200)
    colours = ["#F0C076", "#82DE73", "#FD7777"]
    plt.plot(mc_total_rewards_mse, color = colours[0], linewidth = 0.5, label = r"$\epsilon$ = {:.1f}, MSE = {:.0f}".format(epsilon_array[mse_min], min(mse_array)))
    plt.plot(mc_total_rewards_cv, color = colours[1], linewidth = 0.5, label = r"$\epsilon$ = {:.1f}, MSE = {:.0f}".format(epsilon_array[cv_max], mse_array[cv_max]))
    plt.plot(mc_total_rewards, color = colours[2], linewidth = 0.5, label = r"$\epsilon = {:.3f}^k$, MSE = {:.0f}".format(0.999, mse_value))
    plt.legend()
    plt.minorticks_on()
    plt.ylim([-650, 600])
    plt.xlabel("Episodes")
    plt.ylabel("Sum of undiscounted rewards")
    plt.show()


# ## Optimal Policy

# In[1167]:


if __name__ == "__main__":  
    maze = Maze()
    mc_agent = MC_agent()
    mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)

    print("Results of the MC agent:\n")
    maze.get_graphics().draw_policy(mc_policy)
    maze.get_graphics().draw_value(mc_values[-1])


# In[1036]:


if __name__ == "__main__":
    diff_td = [np.mean(mc_values[i+1] - mc_values[i]) for i in range(len(mc_values)-1)]
    plt.figure(figsize=(10,6), dpi=1200)
    colours = ["#2253FB", "#EC7D69", "#E8EE3F", "#82DE73"]
    plt.plot(diff_td, linewidth = 0.5, color = colours[1])
    plt.minorticks_on()
    plt.xlabel("Episodes")
    plt.ylabel(r"change in $V_*$", labelpad=15)
    plt.show()


# ## Learning Curve of Agent

# In[982]:


if __name__ == "__main__":
    plt.figure(figsize=(10,6))
    plt.plot(range(len(mc_total_rewards)), mc_total_rewards, linewidth = 0.7, label = "MC")
    plt.plot()
    #plt.plot(range(len(td_total_rewards)), td_total_rewards, linewidth = 0.7, label = "TD")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Sum of undiscounted rewards")


# In[693]:


if __name__ == "__main__":    
    maze = Maze()
    n = 25
    mc_policy_array = []
    mc_values_array = []
    mc_total_rewards_array = []
    for i in range(n):
        mc_agent = MC_agent(gamma=1, x=0.999) #undiscounted rewards
        mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)
        mc_policy_array.append(mc_policy)
        mc_values_array.append(mc_values[-1])
        mc_total_rewards_array.append(mc_total_rewards)


# In[695]:


if __name__ == "__main__":    
    mc_total_rewards_mean = np.mean(mc_total_rewards_array, axis = 0)
    mc_total_rewards_std = np.std(mc_total_rewards_array, axis = 0)


# In[696]:


if __name__ == "__main__":
    fig, ax1 = plt.subplots(figsize=(10,6), dpi=1200)

    ax2 = ax1.twinx()

    lns1 = ax2.plot(range(len(mc_total_rewards_mean)), [0.999**k for k in range(len(mc_total_rewards_mean))], '-.',
             label = r"$\epsilon$", color =  "red", linewidth = 1)

    lns2 = ax1.plot(range(len(mc_total_rewards_mean)), mc_total_rewards_mean, linewidth = 0.9, label = r"$\mu$", color = "#FF9600")

    lns3 = ax1.fill_between(range(len(mc_total_rewards_mean)), mc_total_rewards_mean + mc_total_rewards_std, 
                     mc_total_rewards_mean - mc_total_rewards_std, alpha = 0.5, label = r'$\pm$ 1$\sigma$', color = "#FFB54A", linewidth = 0.1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    plt.minorticks_on()
    ax1.set_ylim([-650, 700])
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Sum of undiscounted rewards")
    ax2.set_ylabel(r"$\epsilon$")
    plt.show()


# In[ ]:


if __name__ == "__main__":
    maze = Maze()
    n = 25
    mc_lengths = []
    for i in range(n):
        mc_agent = MC_agent() 
        mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)
        T_mc = mc_agent.get_ep_lengths()
        mc_lengths.append(T_mc)


# In[677]:


if __name__ == "__main__":
    #T_td = td_agent.get_ep_lengths()
    plt.figure(figsize=(10,6))
    plt.plot(range(len(T_mc)), T_mc, linewidth = 0.7, label = "MC")
    #plt.plot(range(len(T_td)), T_td, linewidth = 0.7, label = "MC")
    #plt.plot(range(len(td_total_rewards)), td_total_rewards, linewidth = 0.7, label = "TD")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Epochs")


# ## Q3: Temporal Difference 

# In[896]:


if __name__ == "__main__":
    alpha_1, epsilon_1 = (0.95, 0.74)
    td_agent_1 = TD_agent(alpha=alpha_1, epsilon=epsilon_1, gamma = 1) # if not input, uses ditrbutions 0.999**k
    _, _, td_total_rewards_1 = td_agent_1.solve(maze)
    td_agent_1 = TD_agent(alpha=alpha_1, epsilon=epsilon_1)
    _, td_values_1, _ = td_agent_1.solve(maze)
    mse_1 = mean_squared_error(td_values_1[0], dp_value)

    alpha_2, epsilon_2 = (0.17, 0.01)
    td_agent_2 = TD_agent(alpha=alpha_2, epsilon=epsilon_2, gamma = 1)
    _, _, td_total_rewards_2 = td_agent_2.solve(maze)
    td_agent_2 = TD_agent(alpha=alpha_2, epsilon=epsilon_2)
    _, td_values_2, _ = td_agent_2.solve(maze)
    mse_2 = mean_squared_error(td_values_2[0], dp_value)

    td_agent_3 = TD_agent(gamma = 1)
    _, _, td_total_rewards_3 = td_agent_3.solve(maze)
    td_agent_3 = TD_agent()
    _, td_values_3, _ = td_agent_3.solve(maze)
    mse_3 = mean_squared_error(td_values_3[0], dp_value)


# In[1238]:


if __name__ == "__main__":
    plt.figure(figsize=(10,6), dpi=1200)
    colours = ["#F0C076", "#82DE73", "#FD7777"]
    plt.plot(td_total_rewards_1, color = colours[0], linewidth = 0.5, label = r"$\epsilon$ = {}, $\alpha$ = {}, MSE = {:.0f}".format(epsilon_1, alpha_1, mse_1))
    plt.plot(td_total_rewards_2, color = colours[1], linewidth = 0.5, label = r"$\epsilon$ = {}, $\alpha$ = {}, MSE = {:.0f}".format(epsilon_2, alpha_2, mse_2))
    plt.plot(td_total_rewards_3, color = colours[2], linewidth = 0.5, label = r"$\epsilon$, $\alpha = {}^k$, MSE = {:.0f}".format(0.999, mse_3))
    plt.legend()
    plt.minorticks_on()
    plt.ylim([-650, 600])
    plt.xlabel("Episodes")
    plt.ylabel("Sum of undiscounted rewards")
    plt.show()


# In[1251]:


if __name__ == "__main__":
    epsilon_values = [None,0.01,0.2,0.6,0.8]
    td_policy_array = []
    td_values_array = []
    td_total_rewards_array = []
    cv_eps = []

    for eps_i in epsilon_values:
        #for alpha_i in epsilon_values:
        td_agent = TD_agent(epsilon=eps_i, alpha=0.1, gamma=1)#alpha_i)
        td_policy, td_values, td_total_rewards = td_agent.solve(maze)
        td_policy_array.append(td_policy)
        td_values_array.append(td_values[-1])
        td_total_rewards_array.append(td_total_rewards)
        x = td_total_rewards[1500:]
        cv_eps.append(np.mean(x)/np.std(x))


# In[1254]:


if __name__ == "__main__":
    plt.figure(figsize=(10,6), dpi=1200)
    epsilon_values = ["$0.999^k$",0.01,0.2,0.6,0.8]
    colours = ["#2253FB", "k","#EC7D69", "#E8EE3F", "#82DE73"]

    a = [plt.plot(range(len(x)), x, label = r"$\epsilon$ = {}".format(epsilon_values[i]), linewidth = 0.5, zorder=10-i*2, color = colours[i]) for i,x in enumerate(td_total_rewards_array)]
    plt.legend()

    plt.minorticks_on()
    plt.ylim([-650, 600])
    plt.xlabel("Episodes")
    plt.ylabel("Sum of undiscounted rewards")
    plt.show()


# In[1255]:


if __name__ == "__main__":
    alpha_values = [None,0.15,0.5,0.75,0.95] #[None,0.01,0.2,0.6,0.8]
    td_policy_array = []
    td_values_array = []
    td_total_rewards_array = []
    cv_alpha = []

    for alpha_i in alpha_values:
        #for alpha_i in epsilon_values:
        td_agent = TD_agent(epsilon=0.17, alpha=alpha_i, gamma=1)
        td_policy, td_values, td_total_rewards = td_agent.solve(maze)
        td_policy_array.append(td_policy)
        td_values_array.append(td_values[-1])
        td_total_rewards_array.append(td_total_rewards)
        x = td_total_rewards[1000:]
        cv_alpha.append(np.mean(x)/np.std(x))


# In[1257]:


if __name__ == "__main__":    
    alpha_values = ["$0.999^k$", 0.15,0.5,0.75,0.95]
    plt.figure(figsize=(10,6), dpi=1200)
    colours = ["#2253FB", "#EC7D69", "#E8EE3F", "#82DE73", "k"]
    a = [plt.plot(range(len(x)), x, label = r"$\alpha$ = {}".format(alpha_values[i]), linewidth = 0.5, zorder=10-i*2, color = colours[i]) for i,x in enumerate(td_total_rewards_array)]
    plt.legend()

    plt.minorticks_on()
    plt.ylim([-650, 600])
    plt.xlabel("Episodes")
    plt.ylabel("Sum of undiscounted rewards")
    plt.show()


# In[1011]:


if __name__ == "__main__":
    maze = Maze()
    dp_agent = DP_agent()
    dp_policy, dp_value = dp_agent.solve(maze)


# In[815]:


if __name__ == "__main__":
    epsilon_values = np.linspace(0.01,1,20)
    alpha_values =  np.linspace(0.01,1,20)
    n = len(epsilon_values)
    td_std = np.zeros((n,n))
    td_mean = np.zeros((n,n))
    td_CV = np.zeros((n,n))
    mse = np.zeros((n,n))

    dp_agent = DP_agent()
    dp_policy, dp_value = dp_agent.solve(maze)

    for e, eps_i in enumerate(epsilon_values):
        for a, alpha_i in enumerate(alpha_values):
            print(eps_i, alpha_i)
            td_agent = TD_agent(epsilon=eps_i, alpha=alpha_i)
            td_policy, td_values, td_total_rewards = td_agent.solve(maze)
            mse[e,a] = mean_squared_error(td_values[-1], dp_value)

            # remove first 100 episodes and calculate std 
            td_agent = TD_agent(epsilon=eps_i, alpha=alpha_i, gamma = 1)
            td_policy, td_values, td_total_rewards = td_agent.solve(maze)
            td_rewards_ = td_total_rewards[100:]
            td_std[e, a] = np.std(td_rewards_)
            td_mean[e, a] = np.mean(td_rewards_)
            td_CV[e, a] = td_mean[e, a]/td_std[e, a]


# In[863]:


if __name__ == "__main__":
    min_index = np.unravel_index(np.argmin(mse), mse.shape)
    mse[min_index]


# In[816]:


if __name__ == "__main__":
    # MINIMISE MSE (DP vs TD)

    plt.rcParams.update(          {'axes.labelsize': 10,
          'font.size': 7,
           'xtick.labelsize': 7,
           'ytick.labelsize': 7,
           #'xtick.major.bottom': True,
           'xtick.major.pad': 0.001,
           'xtick.major.size': 6,
          'figure.dpi': 1200
            })

    x = alpha_values
    y = epsilon_values

    X, Y = np.meshgrid(x, y)
    Z = mse

    # Find the indices of the minimum point
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x, min_y = x[min_index[1]], y[min_index[0]]

    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # Plot the maximum point as a red dot on the x-y plane
    #ax.scatter(max_x, max_y, 0,color='red', s=10, zorder=10)
    ax.plot([min_x, min_x], [min_y, min_y], [0, Z[min_index] + 1], color='red', linestyle='dotted', zorder=5)
    ax.plot([min_x, 1], [min_y, min_y], [0, 0], color='red', linestyle='dotted', zorder=5)
    ax.plot([min_x, min_x], [0, min_y], [0, 0], color='red', linestyle='dotted', zorder=5)

    ax.scatter(min_x, min_y, Z[min_index] + 2.5, color='red', s=50, zorder=10)


    ax.text(min_x - 0.2, min_y, Z[min_index], r'$\epsilon$: {:.2f}, $\alpha$: {:.2f}'.format(min_y, min_x), color='red',
            fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='white'), zorder = 100)

    # Set axes labels
    ax.set_xlabel(r"$\alpha$", labelpad=5)
    ax.set_ylabel(r"$\epsilon$", labelpad=5)
    ax.set_zlabel('MSE', labelpad=5)

    cax = fig.add_axes([0.9, 0.1, 0.03, 0.75]) #[x, y, width, height]
    cbar = fig.colorbar(surf, cax=cax)

    ax.view_init(elev=30, azim=-40) 
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    plt.show()


# In[817]:


if __name__ == "__main__":
    ## Maxmimise CV index (mean/std)
    x = alpha_values
    y = epsilon_values

    X, Y = np.meshgrid(x, y)
    Z = td_CV

    # Find the indices of the minimum point
    max_index = np.unravel_index(np.argmax(Z), Z.shape)
    max_x, max_y = x[max_index[1]], y[max_index[0]]

    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # Plot the maximum point as a red dot on the x-y plane
    #ax.scatter(max_x, max_y, 0,color='red', s=10, zorder=10)
    ax.plot([max_x, max_x], [max_y, max_y], [0, Z[max_index] + 1], color='red', linestyle='dotted', zorder=5)
    ax.plot([max_x, 1], [max_y, max_y], [0, 0], color='red', linestyle='dotted', zorder=5)
    ax.plot([max_x, max_x], [0, max_y], [0, 0], color='red', linestyle='dotted', zorder=5)

    ax.scatter(max_x, max_y, Z[max_index] + 2.5, color='red', s=50, zorder=10)


    ax.text(max_x - 0.2, max_y, Z[max_index], r'$\epsilon$: {:.2f}, $\alpha$: {:.2f}'.format(max_y, max_x), color='red',
            fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='white'), zorder = 100)

    # Set axes labels
    ax.set_xlabel(r"$\alpha$", labelpad=5)
    ax.set_ylabel(r"$\epsilon$", labelpad=5)
    ax.set_zlabel('CV score', labelpad=5)

    cax = fig.add_axes([0.9, 0.1, 0.03, 0.75]) #[x, y, width, height]
    cbar = fig.colorbar(surf, cax=cax)

    ax.view_init(elev=30, azim=-40) 
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    plt.show()


# In[864]:


if __name__ == "__main__":    
    max_index = np.unravel_index(np.argmax(td_CV), td_CV.shape)
    td_CV[max_index]


# In[998]:


if __name__ == "__main__":
    td_agent = TD_agent(epsilon=min_y, alpha=min_x)
    td_policy, td_values, td_total_rewards = td_agent.solve(maze)
    print("Results of the TD agent:\n")
    maze.get_graphics().draw_policy(td_policy)
    maze.get_graphics().draw_value(td_values[-1])


# In[1034]:


if __name__ == "__main__":
    td_agent = TD_agent(epsilon=max_y, alpha=max_x)
    td_policy, td_values, td_total_rewards = td_agent.solve(maze)
    print("Results of the TD agent:\n")
    maze.get_graphics().draw_policy(td_policy)
    maze.get_graphics().draw_value(td_values[-1])

