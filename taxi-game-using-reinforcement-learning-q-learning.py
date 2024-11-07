#!/usr/bin/env python
# coding: utf-8

# # Overview
# Available Environments
# 1-Taxi-v3
# 2-FrozenLake-v1
# 3-CliffWalking-v0
# you can choose from them and the defualt one is Taxi Proplem definition : There are 4 locations (labeled by different letters), and our job is to pick up the passenger at one location and drop him off at another. We receive +20 points for a successful drop-off and lose 1 point for every time-step it takes. There is also a 10 points penalty for illegal pick-up and drop-off actions. Introduction In this project we will implement the Q-leaning algorithm and will see how the decay of the hyperparameter such as learning rate and discount factor and eplison will effect the results and we will implement a grid search to select the best parameters.
# 

# # Setup the requirement libraries

# In[1]:


get_ipython().system('pip install gym')
get_ipython().system('pip install numpy')


# # Setup an environment

# In[2]:


def setup_environment(env_name):
    import gym
    env = gym.make(env_name).env
    env.reset()  # reset environment to a new, random state
    env.render()
    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))
    return env


# ## Choose your environment from the available environments below

# In[3]:


get_ipython().system('pip install gym[toy_text]')


# In[4]:


import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


# In[5]:


environment_names=["Taxi-v3","FrozenLake-v1","CliffWalking-v0"]
env=setup_environment(environment_names[0])
env.render()


# # Try to take random actions to achieve the goal

# In[6]:


def random_action_to_end(env):
    epochs = 0
    penalties, reward = 0, 0
    frames = []  # for animation
    done = False
    while not done:
      # automatically selects one random action
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))
    return frames


# In[7]:


def print_frames(frames):
    from IPython.display import clear_output
    from time import sleep
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        #print(frame['frame'].getvalue())
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


# In[8]:


frames=random_action_to_end(env)


# In[9]:


print_frames(frames)


# # Train the agent using Q-learning algorithm

# In[10]:


import random
from IPython.display import clear_output
import numpy as np
def train_the_agent(env,alpha,gamma,epsilon,training_steps,decay_steps,decay=False):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    alpha_decay=1e-4
    gamma_decay=1e-4
    epsilon_decay=1e-4
    learning_epochs=[]
    learning_penalties=[]
    for i in range(1,training_steps ):
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done = False
        if decay and not i % decay_steps :
            alpha-=alpha_decay
            gamma-=gamma_decay
            epsilon-=epsilon_decay
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
        learning_epochs.append(epochs)
        learning_penalties.append(penalties)

    print("Training finished.\n")
    return q_table,learning_epochs,learning_penalties


# In[11]:


q_table,learning_epochs,learning_penalties=train_the_agent(env,alpha=0.1,gamma=0.6,epsilon=0.1,decay_steps=10000,training_steps=100001,decay=True)


# # Evaluation
# Evaluate agent's performance after Q-learning

# In[12]:


def evaluate(q_table2,episodes):
    total_epochs, total_penalties = 0, 0
    for _ in range(episodes):
        # Choose random initial state
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        frames=[]
        while not done:
            action = np.argmax(q_table2[state])
            # print(action)
            state, reward, done, info = env.step(action)
            # Put each rendered frame into dict for animation
            frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
            )

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs
    print_frames(frames)
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    return total_epochs,total_penalties


# In[13]:


total_epochs,total_penalties=evaluate(q_table2=q_table,episodes=100)


# # Grid search to get the best hyperparameter

# In[14]:


import pandas as pd
alpha=[0.1,0.3,0.5,0.9]
gamma=[0.3,0.6,0.8,0.9]
epsilon=[0.3,0.6,0.8,0.9]
best_alpha,best_gamma,best_epsilon=0,0,0
mini_penalties=999999999999999999
mini_epochs=9999999999999999
parameters={
    "Alpha":[],
    "Gamma":[],
    "Epsilon":[],
    "Evaluation Total Penalties":[],
    "Evaluation Total Epochs":[],
}
for ep in epsilon :
    for al in alpha :
        for gm in gamma :
            returned_q_table,returened_learning_epochs,returened_learning_penalties=train_the_agent(env=env,alpha=al,gamma=gm,epsilon=ep,decay_steps=10000,training_steps=10000,decay=False)
#             total_epochs,total_penalties=evaluate(q_table2=returned_q_table,episodes=100)
            parameters['Alpha'].append(al)
            parameters['Gamma'].append(gm)
            parameters['Epsilon'].append(ep)
            parameters['Evaluation Total Penalties'].append(total_penalties)
            parameters['Evaluation Total Epochs'].append(total_epochs)
            if total_penalties<=mini_penalties:
                mini_penalties=total_penalties
                best_alpha=al
                best_gamma=gm
                best_epsilon=ep
            if  total_epochs<=mini_epochs:
                total_epochs=mini_epochs
                best_alpha=al
                best_gamma=gm
                best_epsilon=ep

parameters=pd.DataFrame(parameters)
print(parameters)
print(best_alpha,best_gamma,best_epsilon)


# In[15]:


parameters


# # Train with the best hyper parameters

# In[16]:


best_q_table,learning_epochs,learning_penalties=train_the_agent(env,0.9,0.9,0.9,training_steps=100000,decay_steps=10000,decay=True)


# ### Plot the Penalties and number of epochs in each iteration

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
bins=list(range(0,100000,2000))
np_learning_epochs=np.array(learning_epochs)
np_learning_penalties=np.array(learning_penalties)
sns.lineplot(x=bins, y=np_learning_epochs[bins],label='Learning epochs')
sns.lineplot(x=bins, y=np_learning_penalties[bins],label='Learning penalties')
plt.show()


# # Visualization of trained model over number of episodes

# In[18]:


total_epochs, total_penalties = 0, 0
episodes = 500
for episode in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    frames=[]
    while not done:
        action = np.argmax(best_q_table[state])
        state, reward, done, info = env.step(action)
        # Put each rendered frame into dict for animation
        frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
        )
        if reward == -10:
            penalties += 1

        epochs += 1
    total_penalties += penalties
    total_epochs += epochs
    clear_output(wait=False)
    print_frames(frames=frames)
    print(f"Episode {episode}")

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


# In[ ]:





# In[ ]:




