# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:44:18 2020

@author: arian
"""
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def gen_vecteur(n, m):
    return [(random.gauss(m, m/4), random.gauss(m, m/4)) for _ in range(n)]   

def non_domine_naif(list_vec):
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

def non_domine(list_vec):
    list_sor = sorted(list_vec)
    res = []
    min2 = float("inf")
    for vec in list_sor:
        if vec[1] < min2:
            res.append(vec)
            min2 = vec[1]
    return res

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
    y = non_domine_naif(x)
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    
    ax.plot([w[0] for w in x], [w[1] for w in x], "o")
    ax.plot([w[0] for w in y], [w[1] for w in y], "x")