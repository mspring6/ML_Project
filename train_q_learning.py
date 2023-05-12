"""
Import relevant modules
"""
import logging

import matplotlib.pyplot as plt # Used to plot our final reward and (optional) visualization during training
import numpy as np

from QTable import QTable
from QMap import QMap, Visualize

from helper_functions import *

import sys

import time

"""
=========================
| VISUALIZATION OPTIONS |
=========================

0: NOTHING
1: TRAINING
2: MOVES

"""

"""
==========================
DEFAULT SCRIPT CALL
==========================

Call the python script followed by the following parameters:

1. Number of training episodes - Default is 700 episodes
2. Training Visualization mode (reference the VISUALIZATION OPTIONS) - Default is NOTHING (0)
3. Navigation Visualization mode (reference the VISUALIZATION OPTIONS) - Default is MOVES (2)

python train_q_learning.py 700 0 2

"""

# Receive user input for this particular q learning run
if len(sys.argv) > 0:
    try:
        num_episodes = int(sys.argv[1])
        training_visualization = Visualize(int(sys.argv[2]))
        navigation_visualization = Visualize(int(sys.argv[3]))
    except:
        print(f'Please pass in 3 valid parameters to this Python script... passed in {len(sys.argv)}')
        raise Exception
else: 
    # Default values
    num_episodes = 500
    training_visualization = Visualize.NOTHING
    navigation_visualization = Visualize.MOVES

print(f'Will train for {num_episodes} episodes')

# Setup logging configuration
logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# Create np array which corresponds to our jetbot map
jetbot_map = np.array([
    [0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0]
]) # 0 = OPEN, 1 = BLOCKED

# Create our map object which will be used for Q learning (using our jetbotmap)
JETBOT_MAP_SIM = QMap(jetbot_map)

# Set the visualization mode
JETBOT_MAP_SIM.visualize(training_visualization)

# Create the QTable object which is based off of our jetbot map
jetbot_QTable = QTable(JETBOT_MAP_SIM)

# Kickoff Q Learning
reward_history, goal_history, total_number_of_episodes, total_training_time = jetbot_QTable.start_q_learning(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=num_episodes)

# Plot our graphs depicting the reward history and the rate at which the jetbot reached the goal square
try:
    reward_history  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.manager.set_window_title('Jetbot QTable')
    ax1.plot(*zip(*goal_history))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(reward_history)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except Exception:
    print(f'Error attempting to plot the history graphs...')
    pass
plt.show()

# Set our visualization mode
JETBOT_MAP_SIM.visualize(navigation_visualization)

print(f'Launching the bot using our learned Q model...')
JETBOT_MAP_SIM.navigate(jetbot_QTable, start_square=(0,0))
time.sleep(5)

# Sort our QTable according to the position
# --> Dictionary will start with Position (0,0) and increment to (6,5)
sorted_jetbot_QTable = sort_dict(jetbot_QTable.Q)

# Export our QTable dictionary to a txt file
Q_table_filename = 'Q_table.txt'
export_dict_to_txt(sorted_jetbot_QTable, filename=Q_table_filename)

# Reload our dictionary from the txt file
# --> Not necessary, may use this function later (need to make sure it works)
Q_dict = pickle_load_dict_from_txt(filename=Q_table_filename)

# Export our QTable dictionary to a csv file
csv_header = ['Position', 'Action', 'Reward']
export_dict_to_csv(Q_dict, filename='Q_table.csv', csv_header=csv_header)
