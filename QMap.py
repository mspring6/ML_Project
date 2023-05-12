import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np


"""
Global Variables
"""
UL_SQUARE = (0,0)

class Square(IntEnum):
    OPEN = 0     # Reserved for Squares which the JetBot can navigate through
    BLOCKED = 1  # Reserved for Squares which is walled off from the JetBot
    CURRENT = 2 # Reserved for Squares which currently houses the JetBot

class Possible_Actions(IntEnum):
    LEFT = 0        # Move one square to the LEFT
    RIGHT = 1       # Move one square to the RIGHT
    UP = 2          # Move one square UP
    DOWN = 3        # Move one square DOWN

class Visualize(Enum):
    NOTHING = 0     # No visualization
    TRAINING = 1    # See the visualization through the training stage
    MOVES = 2       # See the moves taken during a one-shot

class Status(Enum):
    GOAL = 0        # Reached the goal
    LOSE = 1        # Hit a wall
    NAVIGATING = 2  # Currently training

class QMap:

    # Create a list of possible actions
    actions = [Possible_Actions.LEFT, Possible_Actions.RIGHT, Possible_Actions.UP, Possible_Actions.DOWN]

    def __init__(self, map, start_square=UL_SQUARE, goal_square=None, **kwargs):

        # Initialize our reward/penalty values
        self.goal_reward = kwargs.get('goal_reward', 10.0)
        self.move_penalty = kwargs.get('move_penalty', -0.05)
        self.revisit_penalty = kwargs.get('revisit_penalty', -0.25)
        self.BLOCKED_penalty = kwargs.get('BLOCKED', -0.75)

        self.map = map

        self.__minimum_reward = -0.5 * self.map.size # Stop the current episode if the aggregate reward get too low

        # Create our map
        map_rows, map_cols = self.map.shape
        self.squares = [(col, row) for col in range(map_cols) for row in range(map_rows)]
        self.open = [(col, row) for col in range(map_cols) for row in range(map_rows) if self.map[row, col] == Square.OPEN]
        self.goal_square = (map_cols - 1, map_rows - 1) if goal_square is None else goal_square
        self.open.remove(self.goal_square)

        # Check for impossible map layout
        if self.goal_square not in self.squares:
            raise Exception(f'Error: goal square at {self.goal_square} is not inside map')
        if self.map[self.goal_square[::-1]] == Square.BLOCKED:
            raise Exception(f'Error: goal square at {self.goal_square} is BLOCKED')

        # Variables for visualizing in Matplotlib
        self.__visualize_option = Visualize.NOTHING
        self.__action_ax = None # axes to visualize the actions
        self.__best_ax = None # axes to visualize the best action for each square

        self.reset(start_square)

    def reset(self, start_square=UL_SQUARE):
        """
        Method to initialize an episode (start our Jetbot at some random location and establish our goal square)
        """
        if start_square not in self.squares:
            raise Exception(f'Error: start square at {start_square} is not inside the map')
        if self.map[start_square[::-1]] == Square.BLOCKED:
            raise Exception(f'Error: start square at {start_square} is BLOCKED')
        if start_square == self.goal_square:
            raise Exception(f'Error: start and goal square are the same --> {start_square}')

        self.previous_square = self.jetbot_square = start_square
        self.__aggregate_reward = 0.0
        self.__visited = set()

        # Initialize the visualization (if it applies)
        if self.__visualize_option in (Visualize.TRAINING, Visualize.MOVES):

            map_rows, map_cols = self.map.shape
            self.__action_ax.clear()
            self.__action_ax.set_xticks(np.arange(0.5, map_rows, step=1))
            self.__action_ax.set_xticklabels([])
            self.__action_ax.set_yticks(np.arange(0.5, map_cols, step=1))
            self.__action_ax.set_yticklabels([])
            self.__action_ax.plot(*self.jetbot_square, "gs", markersize=30) # Start square is a BIG GREEN SQUARE
            self.__action_ax.text(*self.jetbot_square, 'Start', ha="center", va="center", color="white")
            self.__action_ax.plot(*self.goal_square, "rs", markersize=30) # Goal square is a BIG RED SQUARE
            self.__action_ax.text(*self.goal_square, 'Goal', ha="center", va="center", color="white")
            self.__action_ax.imshow(self.map, cmap='binary')
            self.__action_ax.get_figure().canvas.draw()
            self.__action_ax.get_figure().canvas.flush_events()

        return self.observe_state()

    def __draw_path(self):
        """
        Plot the line from the jetbot's previous square to the current square
        """
        self.__action_ax.plot(*zip(*[self.previous_square, self.jetbot_square]), "bo-") # Previous squares are blue dots
        self.__action_ax.plot(*self.jetbot_square, "ro") # Jetbot's current square is marked by a red dot
        self.__action_ax.get_figure().canvas.draw()
        self.__action_ax.get_figure().canvas.flush_events()

    def visualize(self, visualize_option=Visualize.NOTHING):
        """
        Create the visualization environment surrounding the user's chosen option
        """
        self.__visualize_option = visualize_option

        # Don't visualize anything
        if self.__visualize_option == Visualize.NOTHING:
            if self.__action_ax:
                self.__action_ax.get_figure().close()
                self.__action_ax = None
            if self.__best_ax:
                self.__best_ax.get_figure().close()
                self.__best_ax = None

        # Popup a separate window to show the BEST moves to take within each square
        if self.__visualize_option == Visualize.TRAINING:
            if self.__best_ax is None:
                fig, self.__best_ax = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.manager.set_window_title("Best Move")
                self.__best_ax.set_axis_off()
                self.visualize_q(None)

        # Popup a separate window that shows the Jetbot's current moves (whether during a one-shot or during training)
        if self.__visualize_option in (Visualize.MOVES, Visualize.TRAINING):
            if self.__action_ax is None:
                fig, self.__action_ax = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.manager.set_window_title('QMap')

        plt.show(block=False)

    def step(self, action):
        """
        Take one action and evaluate the reward / next state / associated status
        """
        reward = self.do(action)
        self.__aggregate_reward += reward
        status = self.status()
        state = self.observe_state()
        logging.debug(f'action: {self.actions}')
        return state, reward, status

    def do(self, action):
        """
        Do the action and determine the Jetbot position / associated reward
        """
        possible_actions = self.possible_actions(self.jetbot_square)

        if not possible_actions:
            reward = self.__minimum_reward - 1 # Can't move anywhere --> forces a LOSE status
        elif action in possible_actions:
            current_col, current_row = self.jetbot_square
            if action == Possible_Actions.LEFT:
                current_col -= 1
            elif action == Possible_Actions.UP:
                current_row -= 1
            if action == Possible_Actions.RIGHT:
                current_col += 1
            elif action == Possible_Actions.DOWN:
                current_row += 1

            self.previous_square = self.jetbot_square
            self.jetbot_square = (current_col, current_row)

            if self.__visualize_option != Visualize.NOTHING:
                self.__draw_path()

            if self.jetbot_square == self.goal_square:
                reward = self.goal_reward # Given reward when the jetbot reaches the goal square
            elif self.jetbot_square in self.__visited:
                reward = self.revisit_penalty # Incurred penalty when going back over a square previously traversed
            else:
                reward = self.move_penalty # Incurred penalty when an action is taken that doesn't result in reaching the goal square

            self.__visited.add(self.jetbot_square)

        else:
            reward = self.BLOCKED_penalty # Incurred penalty for hitting a wall

        return reward

    def possible_actions(self, square=None):
        """
        Method to determine whether or not a particular action can be taken given a certain state
        """
        if square is None:
            current_col, current_row = self.jetbot_square
        else:
            current_col, current_row = square

        possible_actions = QMap.actions.copy()

        # Only allow VALID moves
        map_rows, map_cols = self.map.shape
        if current_row == 0 or (current_row > 0 and self.map[current_row - 1, current_col] == Square.BLOCKED):
            possible_actions.remove(Possible_Actions.UP)
        if current_row == map_rows - 1 or (current_row < map_rows - 1 and self.map[current_row + 1, current_col] == Square.BLOCKED):
            possible_actions.remove(Possible_Actions.DOWN)

        if current_col == 0 or (current_col > 0 and self.map[current_row, current_col - 1] == Square.BLOCKED):
            possible_actions.remove(Possible_Actions.LEFT)
        if current_col == map_cols - 1 or (current_col < map_cols - 1 and self.map[current_row, current_col + 1] == Square.BLOCKED):
            possible_actions.remove(Possible_Actions.RIGHT)

        return possible_actions

    def status(self):
        """
        Method to determine the STATUS of the Jetbot - did we win, lose, or are we just going?
        """
        if self.jetbot_square == self.goal_square:
            return Status.GOAL

        if self.__aggregate_reward < self.__minimum_reward: # Force LOSE status after excessive penalty
            return Status.LOSE

        return Status.NAVIGATING

    def observe_state(self):
        """
        Method to realize the Jetbot's position
        """
        return np.array([[*self.jetbot_square]])

    def navigate(self, model, start_square=UL_SQUARE):
        """
        Method to perform a one-shot given a Q-table ("model")
        """
        self.reset(start_square)

        state = self.observe_state()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in (Status.GOAL, Status.LOSE):
                return status

    def reach_goal_from_all_squares(self, model):
        """
        Determine win rate through randomly starting from all possible valid squares in a given map
        """
        previous = self.__visualize_option
        self.__visualize_option = Visualize.NOTHING

        GOAL = 0
        LOSE = 0

        for square in self.open:
            if self.navigate(model, square) == Status.GOAL:
                GOAL += 1
            else:
                LOSE += 1
        self.__visualize_option = previous

        # Calcuate the win rate / success rate 
        SUCCESS_RATE = (GOAL) / (GOAL + LOSE)
        logging.info(f'# of times GOAL reached: {GOAL} | # of times LOST: {LOSE} | SUCCESS RATE: {SUCCESS_RATE}')

        result = True if LOSE == 0 else False

        return result, SUCCESS_RATE

    def visualize_q(self, model):
        # print(f'visualize: {self.__visualize_option}')
        if self.__visualize_option == Visualize.TRAINING:
            map_rows, map_cols = self.map.shape

            # Initialize our visualized map with the representation of our input map
            self.__best_ax.clear()
            self.__best_ax.set_xticks(np.arange(0.5, map_rows, step=1))
            self.__best_ax.set_xticklabels([])
            self.__best_ax.set_yticks(np.arange(0.5, map_cols, step=1))
            self.__best_ax.set_yticklabels([])
            self.__best_ax.grid(True)
            self.__best_ax.plot(*self.goal_square, "rs", markersize=30) # GOAL is a BIG RED SQUARE
            self.__best_ax.text(*self.goal_square, "Goal", ha='center', va='center', color='white')

            # Create a plot depicting the best move that can be taken given a certain state
            for square in self.open:
                q = model.q(square) if model is not None else [0, 0, 0, 0]
                a = np.nonzero(q == np.max(q))[0]

                for action in a:
                    dx = 0
                    dy = 0
                    if action == Possible_Actions.LEFT:
                        dx = -0.2
                    if action == Possible_Actions.RIGHT:
                        dx = 0.2
                    if action == Possible_Actions.UP:
                        dy = -0.2
                    if action == Possible_Actions.DOWN:
                        dy = 0.2

                maxv = 1
                minv = -1
                color = self.clip_output((q[action] - minv) / (maxv - minv)) # Normalizes to [-1, 1]

                self.__best_ax.arrow(*square, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

        if self.__visualize_option != Visualize.NOTHING:
            self.__best_ax.imshow(self.map, cmap='binary')
            self.__best_ax.get_figure().canvas.draw()

    def clip_output(self, n):
        return max(min(1, n), 0)
