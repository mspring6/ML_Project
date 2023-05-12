from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import ReLU, PReLU
import matplotlib.pyplot as plt

visited = 0.8 
agent_mark = 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

epsilon = 0.1

class Qmaze(object):
    def __init__(self, maze, agent=(0,0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        print(nrows, ncols)
        self.target = (nrows-1, ncols-1)   
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not agent in self.free_cells:
            raise Exception("Invalid agent Location: must sit on a free cell")
        self.reset(agent)

    def reset(self, agent):
        self.agent = agent
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = agent
        self.maze[row, col] = agent_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = agent_row, agent_col, mode = self.state

        if self.maze[agent_row, agent_col] > 0.0:
            self.visited.add((agent_row, agent_col))  # mark visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == 0:
                ncol -= 1
            elif action == 1:
                nrow -= 1
            if action == 2:
                ncol += 1
            elif action == 3:
                nrow += 1
        else:                  # invalid action
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        agent_row, agent_col, mode = self.state
        nrows, ncols = self.maze.shape
        if agent_row == nrows-1 and agent_col == ncols-1:
            return 10.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (agent_row, agent_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -1.0
        if mode == 'valid':
            return 2.0

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the agent 
        row, col, valid = self.state
        canvas[row, col] = agent_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        agent_row, agent_col, mode = self.state
        nrows, ncols = self.maze.shape
        if agent_row == nrows-1 and agent_col == ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        return actions

class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    agent_row, agent_col, _ = qmaze.state
    canvas[agent_row, agent_col] = 0.3   
    canvas[nrows-1, ncols-1] = 0.9 
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


def build_model(maze, lr = 0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(ReLU())
    model.add(Dense(maze.size))
    model.add(ReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer = 'adam', loss = 'mse')

    return model



def qtrain(model, maze, **opt):
    global epsilon
    max_number_epochs = 500
    max_number_steps = 1000
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()
    
    #Construct the maze environment
    qmaze = Qmaze(maze)

    experience = Experience(model, max_memory = max_memory)

    win_history = []
    numberFreeCells = len(qmaze.free_cells)
    hsize = qmaze.maze.size//2
    win_rate = 0.0
    imctr = 1
 
    for epoch in range(max_number_epochs):
        print("Epoch Num:", epoch)
        #earlyStop = False
        loss = 0.0
        agent_cell = (0,0)#random.choice(qmaze.free_cells)
        qmaze.reset(agent_cell)
        game_over = False 
        envstate = qmaze.observe()
        total_reward = 0.0
        #while not game_over:
        for step in range(max_number_steps):
            #print("Step:",step)
            valid_act = qmaze.valid_actions()
            if not valid_act: break
            prev_env = envstate
            #get the next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_act)
            else: 
                action = np.argmax(experience.predict(prev_env))
            print(action)
            #best_actions.append(action) #append an action 
            #Take action and get the reward and new state
            envstate, reward, game_status = qmaze.act(action)
            total_reward += reward
            if game_status == 'win':
                #If we win, copy the best actions
                #win_actions = best_actions.copy()
                win_history.append(1)
                print("Game Status:", game_status)
                game_over = True
                #earlyStop = True  #stop the training early if we get a winning condition
                break
                
            elif game_status == 'lose':
                print(envstate)
                print("Game Status:", game_status)
                win_history.append(0)
                game_over = True
                
            else:
                game_over = False

            #store the experience
            episode = [prev_env, action, reward, envstate, game_over]
            experience.remember(episode)
            
            if game_over:
                #If we get a game over, clear the best actions list
                #best_actions.clear()
                break
        
        #Train the neural network model at the end of every epoch, after gathering experience
        inputs, targets = experience.get_data(data_size=data_size)
        h = model.fit(
            inputs,
            targets,
            epochs=8,
            batch_size=16,
            verbose=0,
        )
        loss = model.evaluate(inputs, targets, verbose=0)
        loss_list.append(loss)
        reward_list.append(total_reward)
        episode_list.append(epoch)
        print("Episode Loss:", loss)
        model.save("DeepQ_V3") #Save the model after every epoch 
       
# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

maze = np.array([
    [1.,1.,1.,1.,0.,1.,0.],
    [0.,0.,0.,1.,0.,1.,1.],
    [1.,1.,1.,1.,1.,1.,0.],
    [0.,0.,1.,1.,0.,0.,1.],
    [1.,1.,1.,0.,1.,1.,1.],
    [1.,0.,0.,1.,1.,0.,1.],
    [1.,1.,1.,1.,0.,0.,1.],
])

qmaze = Qmaze(maze)
show(qmaze) 
#plt.show()
loss_list = []
episode_list = []
reward_list = []

#List to save best action
best_actions = []
win_actions = []

#model = build_model(maze)
model = keras.models.load_model("DeepQ_V2")
#model = keras.models.load_model("DeepQ")
qtrain(model, maze, max_memory = 8*maze.size, data_size=32, name='DeepQ')


plt.plot(episode_list, loss_list, color='b')
plt.xlabel('Number of Episodes')
plt.ylabel('Loss')
plt.title('Loss over number of Episodes')
plt.show()
plt.plot(episode_list, reward_list, color='b')
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.show()