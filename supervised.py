# -*- coding: utf-8 -*-
"""
Class : Cosci 5555 Machine Learning
Instructor : Dr. Shukla
Assignment : Final Project
Author : Josh Blaney

Classes and their functions are arranged in alphabetical order and are listed
below.

class nn(robot, inputs, outputs, w=None)
 - backward(y)
 - crossentropy(pred, label)
 - crossentropy_grad(pred, label)
 - embed(row, a=False)
 - forward(x)
 - generate(x, length)
 - label(dmap, s)
 - memory(x, y)
 - softmax(z)
 - train(data, lr=0.01, epochs=1, recurrency=1)
 - test(data)

robot(name, smap, goal=None, acts=None)
 - dijkstra(start) 
 - evaluate(s)
 - move(state, action)
 - path(start, sequence)
 
 - get_state(tmap, rows, cols)
"""

# External Dependencies
import numpy as np
from matplotlib import pyplot as plt

# In House Dependencies


"""
    nn(robot, inputs, outputs, w=None)
    A class to house neural network models. Specifically, recurrent 
    autocorrelative memory based networks or simple single layer networks.
    
inputs:
    - robot (robot): Robot object to handle transitions between states
    - inputs (int): The number of inputs the network has
    - outputs (int): The number of outputs the network has
    - w (numpy array): An optional initial w matrix
outputs:
    - nn (object):
"""
class nn():
    def __init__(self, robot, inputs, outputs, w=None):
        self.robot = robot
        self.inputs = inputs
        self.outputs = outputs
        self.w = np.random.rand(outputs, inputs) if w is None else w
        
        
    """
        backward(y)
        A function to run the network in reverse. Useful in bidirectional 
        networks.
        
    inputs:
        - y (numpy array): The label to run through the network in reverse
    outputs:
        - (numpy array): The corresponding array from the network memory
    """
    def backward(self, y):
        return np.dot(self.w,y)
    
    
    """
        crossentropy(pred, label)
        A function to compute the crossentropy loss between an arbitrary
        number of output neurons and the desired labels.
        
    inputs:
        - pred (numpy array): The models prediction array
        - label (numpy array): The desired label array
    outputs:
        - (float): The crossentropy between predictions and labels
    """
    def crossentropy(self, pred, label):
        return -np.sum(label * np.log(pred + 10**-100))
    
    
    """
        crossentropy_grad(pred, label)
        A function to compute the gradient of the crossentropy loss
        for an arbitrary number of output neurons and desired labels.
        
    inputs:
        - pred (numpy array): The models prediction array
        - label (numpy array): The desired label array
    outputs:
        - (numpy array): The gradient vector for the crossentropy loss
    """
    def crossentropy_grad(self, pred, label):
        return -label/(pred + 10**-100)
    
    
    """
        embed(row, a=False):
        A function to perform a transformation between state vectors and 
        model input vectors. 
        
    inputs:
        - row (numpy array): Either an input vector or a state vector
        - a (bool): input vector to state (True) or state to input vector (False)
    outputs:
        - (numpy array): Either an input vector or a state vector
    """
    def embed(self, row, a=False):
        if a:
            index = np.argmax(row)
            j = int(index % self.robot.cols)
            i = int((index - j) / self.robot.cols)
            return np.array([i, j])
        else:
            vec = np.zeros(self.robot.rows*self.robot.cols)
            vec[row[0] * self.robot.cols + row[1]] = 1
            return vec
        
    
    """
        forward(x)
        A function to calculate the induced field before activation.
        
    inputs:
        - x (numpy array): An input vector for the model
    outputs:
        - (numpy array): The induced field
    """
    def forward(self, x):
        return np.dot(self.w,x)
    
    
    """
        generate(x, length)
        A function to generate a seq given an initial state x and a
        maximum length.
    inputs:
        - x (numpy array): Initial input state in the model input format
        - length (int): The maximum allowable length
    outputs:
        - seq (list): The predicted sequence of actions
    """
    def generate(self, x, length):
        seq = []
        x = self.embed(x) if len(x) == 2 else x
        
        for i in range(length):
            z = self.softmax(self.forward(x))
            y = np.argmax(z)
            s1 = self.embed(x, a=True)
            s2 = self.robot.move(s1, y)
            er = self.robot.evaluate(s2)
            seq.append(y)
            if er == 0.0:
                break
            else:
                x = self.embed(s2)
        return seq
     
    
    """
        label(dmap, s)
        A function to automatically generate the label array based
        on the distance maps information about the available actions.
        
    inputs:
        - dmap (numpy array): A map of distances to the goal
        - s (numpy array): A vector of state
    outputs:
        - label (numpy array): The one hot encoded label array
    """
    def label(self, dmap, s):
        if s[0] + 1 <= self.robot.rows-1:
            op1 = 100 if dmap[s[0]+1,s[1]] < 0 else dmap[s[0]+1,s[1]]
        else:
            op1 = 100
        if s[0] - 1 >= 0:
            op2 = 100 if dmap[s[0]-1,s[1]] < 0 else dmap[s[0]-1,s[1]]
        else:
            op2 = 100
        if s[1] - 1 >= 0:
            op3 = 100 if dmap[s[0],s[1]-1] < 0 else dmap[s[0],s[1]-1]
        else:
            op3 = 100
        if s[1] + 1 <= self.robot.cols-1:
            op4 = 100 if dmap[s[0],s[1]+1] < 0 else dmap[s[0],s[1]+1]
        else:
            op4 = 100
            
        opt = np.array([op1, op2, op3, op4, dmap[s[0],s[1]]])
        label = np.zeros(self.outputs)
        label[np.argmin(opt)] = 1
        return label
    
    
    """
        memory (x, y)
        A function to generate the exact memory (w) to map input x to
        output y. Only useful in specific implementations of BAM.
        
    inputs:
        - x (numpy array): The model input array
        - y (numpy array): The model desired output
    outputs:
        - (numpy array): The memory which will perform the x -> y mapping
    """
    def memory(self, x, y):
        return np.matmul(x,y)
    
    
    """
        softmax(z)
        A function to compute the softmax activation.
        
    inputs:
        - z (numpy array): The array to softmax normalize
    outputs:
        - (numpy array): The softmax normalized array
    """
    def softmax(self, z):
        for i, entry in enumerate(z):
            z[i] = np.exp(entry)
        return z / sum(z)
    
    
    """
        train(data, lr=0.01, epochs=1, recurrency=1)
        A function to automate training of a model on some input data
        using dmap to create desired labels.
        
    inputs:
        - data (numpy array): The matrix of input arrays for training
        - dmap (numpy array): The map of distances from the goal
        - lr (float): The learning rate constant to adjust weight updates
        - epochs (int): The number of epochs to run
        - recurrency (int): The number of recurrent steps to take
    outputs:
        - 
    """
    def train(self, data, lr=0.01, epochs=1, recurrency=1):
        for epoch in range(epochs):
            for row in data:
                x = row
                for i in range(recurrency):
                    z = self.softmax(self.forward(x))
                    y = np.argmax(z)
                    s1 = self.embed(x, a=True)
                    s2 = self.robot.move(s1, y)
                    label = self.label(self.robot.dmap, s1)
                    loss = self.crossentropy(z, label)
                    er = self.crossentropy_grad(z, label)
                    x = np.reshape(x, newshape=(1,x.shape[0]))
                    er = np.reshape(er, newshape=(er.shape[0],1))
                    self.w -= lr * loss * er * x
                    if self.robot.evaluate(s2) == 0.0:
                        break
                    else:
                        x = self.embed(s2)    
                        
    """
        test(data)
        A function to run each input vector in data through the model
        once to make sure that the correct actions are being predicted.
        
    inputs:
        - data (numpy array): The initial states to test
    outputs:
        - seq (list): The list of actions to take
    """                   
    def test(self, data):
        seq = []
        for row in data:
            x = row
            z = self.softmax(self.forward(x))  
            seq.append(z)
        return seq
    
    
"""
    robot(name, smap, goal=None, acts=None)
    A class to house the robot control logic for moving through a maze which
    can be any arbitrary maze.
    
inputs:
    - smap (numpy array): The map of the environment
    - goal (numpy array): The desired position
    - acts (dict): A dictionary of actions and how they alter the state
outputs:
    - (robot): A new robot object
"""
class robot():
    def __init__(self, name, smap, goal=None, acts=None):
        self.name = name
        self.smap = smap
        self.rows = smap.shape[0]
        self.cols = smap.shape[1]
        self.goal = [6, 6] if goal is None else goal
        self.acts = {0: [1, 0], 1:[-1, 0], 2:[0, -1], 3:[0, 1], 4:[0, 0]} if acts is None else acts()
    
    
    """
        dijkstra(start)
        A function to run dijkstra's optimal path algorithm over a smap from 
        the initial position specified by start
    
    inputs:
        - start (numpy array): The initial starting position 
    outpus:
        - 
    """
    def dijkstra(self, start):
        dist = np.ones(4)
        self.dmap = np.ones(self.smap.shape) * 100
        self.dmap[start[0], start[1]] = 0
        for k in range(self.cols):
            for i in range(start[0],-1,-1):
                for j in range(start[1],-1,-1):
                    if not (i == start[0] and j == start[1]):
                        dist[0] = self.dmap[i+1,j] if i+1 < self.rows and smap[i+1,j] == 0 else 1e7
                        dist[1] = self.dmap[i-1,j] if i-1 > -1 and smap[i-1,j] == 0 else 1e7
                        dist[2] = self.dmap[i,j-1] if j-1 > -1 and smap[i,j-1] == 0 else 1e7
                        dist[3] = self.dmap[i,j+1] if j+1 < self.cols and smap[i,j+1] == 0 else 1e7
                        self.dmap[i,j] = min(dist) + 1 if smap[i,j] == 0 else -1
                        
            for i in range(self.rows):
                for j in range(self.cols):
                    if not (i == start[0] and j == start[1]):
                        dist[0] = self.dmap[i+1,j] if i+1 < self.rows and smap[i+1,j] == 0 else 1e7
                        dist[1] = self.dmap[i-1,j] if i-1 > -1 and smap[i-1,j] == 0 else 1e7
                        dist[2] = self.dmap[i,j-1] if j-1 > -1 and smap[i,j-1] == 0 else 1e7
                        dist[3] = self.dmap[i,j+1] if j+1 < self.cols and smap[i,j+1] == 0 else 1e7
                        self.dmap[i,j] = min(dist) + 1 if smap[i,j] == 0 else -1
                        
        norm = np.max(self.dmap)
        for i, row in enumerate(self.dmap):
            for j, value in enumerate(row):
                if value >=0:
                    self.dmap[i,j] /= norm
                    
    
    """
        evaluate(s)
        A function to evaluate how close the robot is to the goal state using
        euclidean distance.
        
    inputs:
        - s (numpy array): The current state of the robot
    outputs:
        - (float): The euclidean distance between the goal and s
    """
    def evaluate(self, s):
        s1 = self.goal[0] - s[0]
        s2 = self.goal[1] - s[1]
        return np.sqrt((s1*s1) + (s2*s2))
    
    
    """
        move(state, action)
        A function to provide the logic for moving the robot to a new state.
        It is assumed that the transition probability is 1.
        
    inputs:
        - state (numpy array): The state to start the robot in
        - action (int): The key from acts to use as the action
    outputs:
        - state (numpy array): The new state of the robot
    """
    def move(self, state, action):
        tstate = state + self.acts[action]
        if tstate[0] < 0 or tstate[1] < 0 or tstate[0] >= self.rows or tstate[1] >= self.cols:
            return state
        
        if self.smap[tstate[0], tstate[1]] == 0:
            return tstate
        else:
            return state
        
        
    """
        path(start, sequence)
        A function to transition the robot along a path using the control 
        sequence specified in sequence.
    
    inputs:
        - start (numpy array): The initial position of the robot
        - sequence (list): The list of integers to specify actions
    outputs:
        - state (numpy array): The final state of the robot
    """
    def path(self, start, sequence):
        state = start
        for act in sequence:
            state = self.move(state, act)
        return state
        

"""
    get_state(tmap, rows, cols)
    A function to get a random valid state within from the map
    
inputs:
    - tmap (numpy array): The map
    - rows (int): The number of rows in the map
    - cols (int): The number of cols in the map
outputs:
    - (list): A state within the map
"""
def get_state(tmap, rows, cols):
    done = False
    row = int(np.random.rand() * rows)
    while not done:
        col = int(np.random.rand()*cols)
        done = True if tmap[row, col] == 0 else False
            
    return [row, col]

init_dict = {}
goal_dict = {}
map_dict = {}

map0 = np.array([[0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0]],dtype=float)

init_dict[0] = np.array([0,0])
goal_dict[0] = np.array([map0.shape[0]-1,map0.shape[1]-1])
map_dict[0] = map0

map1 = np.array([[0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0]],dtype=float)

init_dict[1] = np.array([0,0])
goal_dict[1] = np.array([map1.shape[0]-1,map1.shape[1]-1])
map_dict[1] = map1

map2 = np.array([[0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1],
                 [1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                 [1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0],
                 [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 [0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0]],dtype=float)

init_dict[2] = np.array(get_state(map2, map2.shape[0], map2.shape[1]))    
goal_dict[2] = np.array(get_state(map2, map2.shape[0], map2.shape[1]))
map_dict[2] = map2

map3 = np.array([[1,0,0,0,1,0,0],
                 [1,0,0,0,0,0,0],
                 [1,0,0,1,1,1,1],
                 [1,0,0,0,0,0,0],
                 [0,1,1,1,1,1,0],
                 [0,0,0,0,1,0,0],
                 [0,0,1,0,0,0,0]],dtype=float)

init_dict[3] = np.array([0,5])
goal_dict[3] = np.array([4,0])
map_dict[3] = map3

# Change select to try it on a different map
select = 2
smap = np.copy(map_dict[select])
init = init_dict[select]
goal = goal_dict[select]
size = smap.shape[0] * smap.shape[1]

index = 0
rows, cols = smap.shape
train_data = np.zeros((size,size))
for i, row in enumerate(smap):
    for j, value in enumerate(row):
        if value == 0:
            jndex = i * cols + j
            train_data[index, jndex] = 1    
            index += 1


jb = robot('jb', smap, goal=goal)
jb.dijkstra(goal)
dmap = jb.dmap
state = jb.move(init,3)
print(f'Initial State : {init} | Action : 3 | Final State : {state} \n')


gen1 = nn(robot=jb, inputs=size, outputs=5, w=None)
gen1.train(data=train_data, epochs=100)
seq = gen1.generate(x=init, length=1000)
state = jb.path(init, seq)
error = jb.evaluate(state)
print('Standard NN')
print(f'Sequence : \n{seq}')
print(f'Initial State : {init} | Goal State: {goal} | Final State : {state} | Error : {error} \n')


gen2 = nn(robot=jb, inputs=size, outputs=5, w=None)
gen2.train(data=train_data, recurrency=100)
seq = gen2.generate(x=init, length=1000)
state = jb.path(init, seq)
error = jb.evaluate(state)
print('Recurrent NN')
print(f'Sequence : \n{seq}')
print(f'Initial State : {init} | Goal State: {goal} | Final State : {state} | Error : {error} \n')

smap = -smap
smap[goal[0],goal[1]] += 0.5
smap[init[0],init[1]] += 0.75

fig, ax = plt.subplots(1,2)
ax[0].imshow(smap, cmap='gray')
ax[0].title.set_text("Map")
ax[1].imshow(dmap, cmap='hot')
ax[1].title.set_text("Error Map")
