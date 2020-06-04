#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:50:39 2020

@author: qicai
"""

import time
import random
from collections import namedtuple

import my_graph_library


class env:
    
    def __init__(self, graph, init_matching):
        self.graph = graph
        self.matching = init_matching
        self.matched_nodes = set()
        for edge in init_matching:
            self.matched_nodes.add(edge[0])
            self.matched_nodes.add(edge[1])
        self.avail_edges = set()
        for edge in self.graph.edges:
            check1 = (edge in self.graph.edges)
            check2 = (edge[0] not in self.matched_nodes)
            check3 = (edge[1] not in self.matched_nodes)
            check4 = (edge[0]!=edge[1])
            if (check1*check2*check3*check4) == 1:
                self.avail_edges.add(edge)
        
    def step(self, edge):
        # add an edge to the matching
        if self.is_available(edge):
            self.matching.add(edge)
            self.matched_nodes.add(edge[0])
            self.matched_nodes.add(edge[1])
            self.avail_edges.remove(edge)
             
            node1 = edge[0]
            node2 = edge[1]
            for col_indice in self.graph.mtx.getrow(node1).nonzero()[1]:
                if node1 < col_indice:
                    other_edge = (node1, col_indice) 
                else:
                    other_edge = (col_indice, node1)
                    
                if other_edge in self.avail_edges:
                    self.avail_edges.remove(other_edge)
                    
            for col_indice in self.graph.mtx.getrow(node2).nonzero()[1]:
                if node2 < col_indice:
                    other_edge = (node2, col_indice) 
                else:
                    other_edge = (col_indice, node2)
                    
                if other_edge in self.avail_edges:
                    self.avail_edges.remove(other_edge)       
                    
            reward = 1
            return reward
        else:
            raise ValueError('指令无法接受')
            
    def is_available(self, edge):
        check1 = (edge in self.graph.edges)
        check2 = (edge[0] not in self.matched_nodes)
        check3 = (edge[1] not in self.matched_nodes)
        check4 = (edge[0]!=edge[1])
        if (check1*check2*check3*check4) == 1:
            return True
        else:
            return False
            
    def available_edges(self):
        available = []
        for edge in self.graph.edges:
            if self.is_available(edge):
                available.append(edge)
        return available
    
    def feature(self, edge):
        v = [0,0,0,0,0]
        v[0] = len(self.avail_edges)
        v[1] = self.graph.degrees[edge[0]] + self.graph.degrees[edge[1]]
        v[2] = self.graph.node_num
        v[3] = self.graph.edge_num
        v[4] = len(self.matching)
        return v
    
    
    
    
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)    
    
    
    
    
    
    
def explore(num_training_graphs, num_episodes, size, prob, memory_capacity):
    memory = ReplayMemory(memory_capacity)

    training_graphs = []
    for i in range(num_training_graphs):
        graph = my_graph_library.random_graph(100, 0.1)
        training_graphs.append(graph)
    

    for graph in training_graphs:
        print('Start exploring a new training graph...', end = '')
        if len(graph.edges) == 0:
            continue
        for n_episode in range(num_episodes):
            environment = env(graph, set())
            state = (graph, environment.matching.copy())
            while True:
                available = environment.available_edges()
                action = (random.sample(available, 1))[0]
                reward = environment.step(action) # mdp moves forward

                if len(environment.available_edges()) == 0:
                    next_state = None
                    memory.push(state, action, reward, next_state)
                    break
                else:
                    next_state = (graph, environment.matching.copy())
                    memory.push(state, action, reward, next_state)
                    state = next_state
        print('...completed')
   
    return memory
    
    