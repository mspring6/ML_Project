"""
Import relevant modules
"""
import logging
import random
from datetime import datetime
import numpy as np
import QMap

class QTable():

    def __init__(self, map, **kwargs):
        self.map = map  # Map corresponding to our jetbot map (should be passed in as np array)
        self.Q = dict() # Our QTable will be stored in this dictionary

    def start_q_learning(self, **kwargs):
        """
        Initiate our Q Learning

        Training Hyperparameters:

        discount (gamma): Likelihood to incorporate future rewards (range 0-1)
        exploration_rate (epsilon): Likelihood to lean into exploration (range 0-1)
        exploration_decay: Reduction in the exploration_rate after subsequent steps (range 0-1)
        learning_rate (eta): Likelihood to incorporate new information
        episodes: Number of training episodes
        check_reward_interval: How often to save off our reward history
        """

        discount = kwargs.get('discount', 0.9)
        exploration_rate = kwargs.get('exploration_rate', 0.1)
        exploration_decay = kwargs.get('exploration_decay', 0.995)
        learning_rate = kwargs.get('learning_rate', 0.1)
        episodes = kwargs.get('episodes', 1000)
        check_reward_interval = kwargs.get('check_reward_interval', 5)

        self.aggregate_reward = 0   # Running total of our reward
        self.reward_history = []    # Entries correspond to the reward for a given episode
        self.goal_history = []      # Entries correspond to the number of times our jetbot made it to the goal square per episode

        """
        'start_list' houses which squares we can start in. Once a square has been chosen, then it will be removed from 
        the list. Once all possible starting squares have been exercised, then we will repopulate with all possible
        squares. This is done to ensure that our random starting squares reach every possible square.
        """
        start_list = list() 

        # Variable to help track total training time
        start_time = datetime.now()

        # Begin Q Learning
        for episode in range(1, episodes + 1):

            print(f'============================')
            print(f'Episode: {episode}')

            # Choose random starting square from available options
            if not start_list:
                start_list = self.map.open.copy()
            start_square = random.choice(start_list)
            start_list.remove(start_square)
            print(f'start_square: {start_square}')

            # Establish our 'state' or position in the map
            state = self.map.reset(start_square)
            state = tuple(state.flatten())

            while True:

                # Randomly choose an action to take in our current state
                # --> Lean into choosing the action based off a greedy policy
                if np.random.random() < exploration_rate:
                    action = random.choice(self.map.actions)
                else:
                    action = self.predict(state)
                next_state, reward, status = self.map.step(action)
                next_state = tuple(next_state.flatten())

                # Keep track of the total reward
                self.aggregate_reward += reward

                # Create new action for a given state/position if it doesn't exist
                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0

                # Greedy Policy
                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.map.actions])
                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                # Once the jetbot has reached the goal square, has hit a wall, or has reached the minimum reward - stop training
                if status in (QMap.Status.GOAL, QMap.Status.LOSE):
                    break

                # Increment state/position based off the action taken
                state = next_state

                # Visualize the new state/position (if visualization_option != Visualize.NOTHING)
                self.map.visualize_q(self)

            self.reward_history.append(self.aggregate_reward)

            # Log relevant information
            logging.info(f'Episode: {episode}/{episodes} | Status: {status.name} | Exploration Rate: {exploration_rate}')

            # Check if the 
            if episode % check_reward_interval == 0:
                goal_all, goal_rate = self.map.reach_goal_from_all_squares(self)
                self.goal_history.append((episode, goal_rate))

            exploration_rate *= exploration_decay

        logging.info(f'Episodes: {episode} | Time Spent: {datetime.now() - start_time}')

        return self.reward_history, self.goal_history, episode, datetime.now() - start_time

    def q(self, state):
        """
        Get the Q values or reward values for all possible actions for a given state
        """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.map.actions])

    def predict(self, state):
        """
        Function to choose the action with the highest reward value from the QTable
        """
        q = self.q(state)

        logging.debug(f'q[] = {q}')

        actions = np.nonzero(q == np.max(q))[0]
        return random.choice(actions)
