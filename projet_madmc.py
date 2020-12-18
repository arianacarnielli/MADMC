# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:44:18 2020

@author: Ariana Carnielli
"""
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def gen_vecteur(n, m):
    return [(random.gauss(m, m/4), random.gauss(m, m/4)) for _ in range(n)]   

def non_domine_p_naif(list_vec):
    res = []
    for i in range(len(list_vec)):
        dom = False
        for j in range(len(list_vec)):
            if (list_vec[i][0] > list_vec[j][0] and list_vec[i][1] >= list_vec[j][1]) \
            or (list_vec[i][0] >= list_vec[j][0] and list_vec[i][1] > list_vec[j][1]):
                dom = True
                break
        if not dom:
            res.append(list_vec[i])
    return res 

def non_domine_p(list_vec):
    list_sor = sorted(list_vec)
    res = []
    min2 = float("inf")
    for vec in list_sor:
        if vec[1] < min2:
            res.append(vec)
            min2 = vec[1]
    return res

def pareto_dyn(tab_couts, k):
    n = len(tab_couts)
    T = np.empty((n, k + 1), dtype = object)
    T.fill([])
    T[:, 0].fill([])
    T[0, 0].append((0, 0))
    T[0, 1] = [tab_couts[0]]
    for i in range(1, n):
        for j in range(1, min(k + 1, i + 2)):
            F = [(y1 + tab_couts[i][0], y2 + tab_couts[i][1]) for (y1, y2) in T[i - 1, j - 1]]
            T[i, j] = non_domine_p(F + T[i - 1, j])
    return T[-1, -1]

def f_i(y, I):
    fmin = I[0] * y[0] + (1 - I[0]) * y[1]
    fmax = I[1] * y[0] + (1 - I[1]) * y[1]
    return max (fmin, fmax)

def vec_minimax(list_vec, I):
    res = float("inf")
    vec_res = None
    for y in list_vec:
        fi = f_i(y, I)
        if fi < res:
            res = fi
            vec_res = y
    return vec_res

def pareto_solver(list_vec, k, I):
    list_par = pareto_dyn(list_vec, k)
    return vec_minimax(list_par, I)

def non_domine_i(list_vec, I):
    pi = np.array([[I[0], 1 - I[0]], [I[1], 1 - I[1]]])
    pi_in = np.linalg.inv(pi)
    list_i = [tuple(pi.dot(y)) for y in list_vec]  
    list_par = non_domine_p(list_i)
    return [tuple(pi_in.dot(w)) for w in list_par]

def i_solver(list_vec, k, I):
    pi = np.array([[I[0], 1 - I[0]], [I[1], 1 - I[1]]])
    pi_in = np.linalg.inv(pi)
    list_i = [tuple(pi.dot(y)) for y in list_vec]  
    list_par = pareto_dyn(list_i, k)
    list_res = [tuple(pi_in.dot(w)) for w in list_par]
    return vec_minimax(list_res, I)    

def tester_temps(fonction):
    res = {}
    for n in tqdm(range(200, 10001, 200)):
        temps = time.process_time()
        for _ in range(50):
            vec = gen_vecteur(n, 1000)
            fonction(vec)
        temps = time.process_time() - temps 
        res[n] = temps / 50
    return res




if __name__ == "__main__":
    
    n = 10000
    m = 100
    
    x = gen_vecteur(n, m)
    y = non_domine_p_naif(x)
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    
    ax.plot([w[0] for w in x], [w[1] for w in x], "o")
    ax.plot([w[0] for w in y], [w[1] for w in y], "x")