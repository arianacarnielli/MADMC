# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:44:18 2020

Projet MADMC
Sélection bi-objectifs avec coefficients intervalles

@author: Ariana Carnielli
"""
import random
import time
import numpy as np
from tqdm import tqdm

def gen_vecteur(n, m):
    """
    Crée une liste de n vecteurs.
    Chaque vecteur est représenté comme un tuple de Python à 2 éléments où 
    chaque élément a des valeurs tirées aléatoirement avec la loi normale 
    d'espérance m et d'écart-type m/4.

    Parameters
    ----------
    n : int
        La quantité de vecteurs créés.
    m : int
        L’espérance de la loi normale utilisée. 
               
     Returns
     -------
    _ : list((float, float))
        Liste de taille n avec des tuples de floats de taille 2 représentant 
        les vecteurs. 
    """
    return [(random.gauss(m, m/4), random.gauss(m, m/4)) for _ in range(n)]   

def non_domine_p_naif(list_vec):
    """
    Détermine les vecteurs Pareto non-dominés dans une liste de vecteurs.
    Procède avec des comparaisons par paires systématiques de vecteurs et 
    retourne une liste avec les vecteurs non dominés. 

    Parameters
    ----------
    list_vec : list((float, float))
        Liste avec des tuples de floats de taille 2 représentant des vecteurs. 

    Returns
    -------
    res : list((float, float))
        liste avec les vecteurs non dominés.
    """
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
    """
    Détermine les vecteurs Pareto non-dominés dans une liste de vecteurs.
    Réalise d'abord un tri lexicographique des vecteurs, puis un seul parcours 
    de la liste obtenue pour identifier les vecteurs non-dominés.
    
    Parameters
    ----------
    list_vec : list((float, float))
        Liste avec des tuples de floats de taille 2 représentant des vecteurs. 

    Returns
    -------
    res : list((float, float))
        liste des vecteurs non dominés.
    """
    list_sor = sorted(list_vec)
    res = []
    min2 = float("inf")
    for vec in list_sor:
        if vec[1] < min2:
            res.append(vec)
            min2 = vec[1]
    return res

def pareto_dyn(tab_couts, k):
    """
    Utilise la programmation dynamique pour calculer l'image des sous-ensembles
    Pareto-optimaux de taille k d'un ensemble de taille n d’objets bi-valués. 

    Parameters
    ----------
    tab_couts : list((float, float))
        Liste avec des tuples de floats de taille 2 représentant des coûts des
        objets. 
    k : int
        Quantité d'objets à prend dans la solution.

    Returns
    -------
    _ : list((float, float))
        Liste des images des solutions Pareto non-dominés.
    """
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
    """
    Calcule la valeur de la fonction f_I sur le vecteur y avec l’intervalle I.

    Parameters
    ----------
    y : tuple(float, float)
        Le vecteur qu'on veut appliquer la fonction f_I.
    I : tuple(float, float)
        L'intervalle [alpha_min, alpha_max] utilisé dans le calcul.

    Returns
    -------
    _ : float
        La valeur de la fonction f_i(y) dans l’intervalle I.  
    """
    fmin = I[0] * y[0] + (1 - I[0]) * y[1]
    fmax = I[1] * y[0] + (1 - I[1]) * y[1]
    return max (fmin, fmax)

def vec_minimax(list_vec, I):
    """
    Calcule un vecteur minimax dans une liste de vecteurs.
    Utilise la fonction f_i pour calcular la valeur F_i de chaque vecteur.

    Parameters
    ----------
    list_vec : list((float, float))
        Liste avec des tuples de floats de taille 2 représentant des vecteurs. 
    I : tuple(float, float)
        L'intervalle [alpha_min, alpha_max] utilisé dans le calcul.

    Returns
    -------
    vec_res : tuple(float, float)
        Un vecteur minimax de la liste des vecteurs. 
    """
    res = float("inf")
    vec_res = None
    for y in list_vec:
        fi = f_i(y, I)
        if fi < res:
            res = fi
            vec_res = y
    return vec_res

def pareto_solver(list_vec, k, I):
    """
    Implémente la procédure en deux temps pour déterminer l’image d’une 
    solution minimax dans l’espace des objectifs. 
    Fait comme décrit dans la partie 3 du projet en utilisant la dominance de
    Pareto.

    Parameters
    ----------
    list_vec : list((float, float))
        Liste avec des tuples de floats de taille 2 représentant des coûts des
        objets. 
    k : int
        Quantité d'objets à prend dans la solution.
    I : tuple(float, float)
        L'intervalle [alpha_min, alpha_max] utilisé dans le calcul.

    Returns
    -------
    _ : tuple(float, float)
        Vecteur avec l'image d'une solution minimax dans l'espace des 
        objectifs. 

    """
    list_par = pareto_dyn(list_vec, k)
    return vec_minimax(list_par, I)

def non_domine_i(list_vec, I):
    """
    Détermine les vecteurs non I-dominés dans une liste de vecteurs.
    Réalise d'abord une transformation des vecteurs par la fonction pi, puis 
    calcule les vecteurs Pareto non-dominés parmi ceux-ci et enfin transforme
    ces vecteurs à l'aide de la fonction pi-1 pour identifier les vecteurs non
    I-dominés.
    
    Parameters
    ----------
    list_vec : list((float, float))
        Liste avec des tuples de floats de taille 2 représentant des vecteurs. 
    I : tuple(float, float)
        L'intervalle [alpha_min, alpha_max] utilisé dans le calcul.

    Returns
    -------
    _ : list((float, float))
        liste des vecteurs non I-dominés.
    """
    pi = np.array([[I[0], 1 - I[0]], [I[1], 1 - I[1]]])
    pi_in = np.linalg.inv(pi)
    list_i = [tuple(pi.dot(y)) for y in list_vec]  
    list_par = non_domine_p(list_i)
    return [tuple(pi_in.dot(w)) for w in list_par]

def i_solver(list_vec, k, I):
    """
    Implémente la procédure en deux temps pour déterminer l’image d’une 
    solution minimax dans l’espace des objectifs. 
    Fait comme décrit dans la partie 4 du projet en utilisant la I-dominance.

    Parameters
    ----------
    list_vec : list((float, float))
        Liste avec des tuples de floats de taille 2 représentant des coûts des
        objets. 
    k : int
        Quantité d'objets à prend dans la solution.
    I : tuple(float, float)
        L'intervalle [alpha_min, alpha_max] utilisé dans le calcul.

    Returns
    -------
    _ : tuple(float, float)
        Vecteur avec l'image d'une solution minimax dans l'espace des 
        objectifs. 
    """
    pi = np.array([[I[0], 1 - I[0]], [I[1], 1 - I[1]]])
    pi_in = np.linalg.inv(pi)
    list_i = [tuple(pi.dot(y)) for y in list_vec]  
    list_par = pareto_dyn(list_i, k)
    list_res = [tuple(pi_in.dot(w)) for w in list_par]
    return vec_minimax(list_res, I)    

def tester_temps_n(fonction):
    """
    Implémente le test de temps de calcul démandé à la Question 5.
    Teste une fonction de calcul des points Pareto non-dominés passée en 
    paramètre en créant 50 ensembles de vecteurs tirés aléatoirements pour 
    chaque valeur de n entre 200 et 1000, avec un pas de 200, et en calculant
    le temps moyen de calcul pour chaque valeur de n.

    Parameters
    ----------
    fonction : Function
        La fonction qu'on veut tester le temps d'exécution. 
    Returns
    -------
    res : dict(int : float)
        Dictionnaire dont les clés sont les tailles utilisées dans la création 
        des vecteurs et les valeurs le temps moyen de calcul pour chaque 
        taille.
    """
    res = {}
    for n in tqdm(range(200, 10001, 200)):
        res[n] = 0
        for _ in range(50):
            vec = gen_vecteur(n, 1000)
            temps = time.process_time()
            fonction(vec)
            temps = time.process_time() - temps 
            res[n] += temps
        res[n] = res[n] / 50
    return res

def tester_temps_alpha(fonction):
    """
    Implémente le test de temps de calcul démandé à la Question 12.
    Teste une fonction de calcul d'un image d'une solution minimax dans 
    l'espace des objectifs passée en paramètre en créant 500 ensembles de 
    vecteurs tirés aléatoirements pour chaque valeur de épsilon entre 0.025 et
    0.5, avec un pas de 0.025, et en calculant le temps moyen de calcul pour 
    chaque valeur d'épsilon.

    Parameters
    ----------
    fonction : Function
        La fonction qu'on veut tester le temps d'exécution. 
    Returns
    -------
    res : dict(int : float)
        Dictionnaire dont les clés sont les épsilons utilisés pour créer 
        les intervalles I et les valeurs le temps moyen de calcul pour chaque 
        épsilon.

    """
    res = {}
    for eps in tqdm(range(25, 510, 25)):
        eps = eps / 1000
        res[eps] = 0
        for _ in range(500):
            vec = gen_vecteur(50, 1000)
            temps = time.process_time()
            fonction(vec, 10, (0.5 - eps, 0.5 + eps))
            temps = time.process_time() - temps 
            res[eps] += temps
        res[eps] = res[eps] / 500
    return res
