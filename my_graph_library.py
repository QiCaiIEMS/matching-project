#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:47:18 2020

@author: qicai
"""

import random
from scipy.sparse import coo_matrix


class graph_diy:

    def __init__(self, mtx):
        self.mtx = mtx
        self.size = mtx.shape
        self.node_num = mtx.shape[0]
        self.node = set([i for i in range(self.node_num)])
        self.degrees = [0 for i in range(self.node_num)]
        self.edge_num = mtx.nnz
        self.edges = set()
        for i in range(mtx.nnz):
            row = mtx.row[i]
            col = mtx.col[i]
            if row < col:
                edge = (row, col)
                self.degrees[row] += 1
                self.degrees[col] += 1
            else:
                continue
            self.edges.add(edge)
            

        


def random_graph(size, prob):
    row, col, data = [], [], []
    for i in range(size):
        for j in range(size):
            toss = random.random()
            if toss < prob:
                row.append(i)
                col.append(j)
                data.append(1)
    mtx = coo_matrix((data, (row, col)), shape=(size, size))
    return graph_diy(mtx)