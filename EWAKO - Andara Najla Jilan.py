#!/usr/bin/env python
# coding: utf-8

# # Perbandingan Rute Terpendek Graph Tak Berarah Menggunakan Q-Learning dan SARSA
# 
# Andara Najla Jilan - EWAKO

# In[1]:


import numpy as np
import pylab as plt 
import networkx as nx
import pandas as pd 


# Buatlah graf tak berarah dengan 11 node (0-10)

# In[2]:



edge_list = [(0, 2), (0, 1), (0, 3), (2, 4), (5, 6), (7, 4), (0, 6), (5, 3), (3, 7), (0, 8), (5, 7), (7, 9), (9, 4), (10, 1)]
graph = nx.Graph()
graph.add_edges_from(edge_list)


# In[3]:


# tentukan node akhir
goal = 9
# initial state yang digunakan pada testing
initial_state = 0


# In[4]:


#gambarkan graphnya
position = nx.spring_layout(graph)

nx.draw_networkx_nodes(graph, position)
nx.draw_networkx_edges(graph, position)
nx.draw_networkx_labels(graph, position)
plt.show()


# In[5]:


# ukuran size matrix dan reward table
SIZE_MATRIX = 11


# In[6]:


# buat reward matrix dengan initial value -1
R = np.matrix(np.ones(shape=(SIZE_MATRIX, SIZE_MATRIX)))
R *= -1


# In[7]:



for edge in edge_list:
    #print(edge)
    if edge[1] == goal:
        R[edge] = 100
    else:
        R[edge] = 0
    if edge[0] == goal:
        R[edge[::-1]] = 100
    else:
        R[edge[::-1]] = 0

R[goal, goal] = 100

pd.DataFrame(R)


# In[8]:


# discount factor
gamma = 0.8
# learning rate
alpha = 0.3


# In[9]:


# buat Q table
Q = np.matrix(np.zeros([SIZE_MATRIX, SIZE_MATRIX]))


# In[10]:


def get_available_actions(state):
    """
    Get all actions where the rewards are 0 or greater 
    are available from the current state. Action is an edge that exists.
    """
    current_state_row = R[state,]
    available_actions = np.where(current_state_row >= 0)[1]
    return available_actions


# In[11]:


def sample_next_action(available_actions):
    """
    Choose the next state at random
    """
    next_action = int(np.random.choice(available_actions, size=1))
    return next_action


# In[12]:


def shortest_path(start_state, end_state):
    """
    Test the trained Q-table to find the shortest path
    """
    current_state = start_state
    steps = [current_state]

    while current_state != end_state:

        next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size = 1))
        else:
            next_step_index = int(next_step_index)

        steps.append(next_step_index)
        current_state = next_step_index

    print("Most efficient path:")
    print(steps)


# # Q-learning
# 
# ![image.png](attachment:39b9ea43-6e52-46a9-a546-c171986c9a4a.png)

# In[13]:


def update_Q(current_state, action, gamma):
    """
    Update Q-values in the current Q-table
    """
    
    # np.where returns a tuple of ndarrays where the output array contains elements 
    # of x where condition is True. We pick the second element of the tuple which 
    # is a list of states where the reward is highest. If there are multiple states 
    # where the reward is highest, we pick one state at random.
    
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1] 

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    Q[current_state, action] += alpha * (R[current_state, action] + gamma * max_value - Q[current_state, action])  


# ## Train menggunakan Q-learning

# In[14]:


for i in range(1000):
    current_state = np.random.randint(0, int(Q.shape[0])) 
    available_action = get_available_actions(current_state)
    action = sample_next_action(available_action)
    update_Q(current_state, action, gamma)

print("Trained Q matrix:")
pd.DataFrame(Q)


# ## Test shortest path, trained using Q-learning method

# In[15]:


shortest_path(initial_state, goal)


# # SARSA
# ![image.png](attachment:b2fdd28b-8947-4540-8f64-b51b0ec87429.png)

# In[16]:


def update_Q_SARSA(current_state, action, gamma):
    """
    Update Q-values in the current Q-table using SARSA
    """
    available_action = get_available_actions(action)
    next_action = sample_next_action(available_action)
    next_state = action

    Q[current_state, action] += alpha * (R[current_state, action] + gamma * Q[next_state, next_action] - Q[current_state, action])  


# ## Train menggunakan SARSA

# In[17]:


#inisialisasi Q-matrix
Q = np.matrix(np.zeros([SIZE_MATRIX, SIZE_MATRIX]))


# In[18]:


for i in range(1000):
    current_state = np.random.randint(0, int(Q.shape[0])) 
    available_action = get_available_actions(current_state)
    action = sample_next_action(available_action)
    update_Q_SARSA(current_state, action, gamma)

print("Trained Q matrix:")
pd.DataFrame(Q)


# In[19]:


shortest_path(initial_state, goal)

