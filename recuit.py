from utilities import get_dict, plot_sol, max_dist, dist
from energy import Potential
from random import randint, random
import numpy as np
import matplotlib.pyplot as plt

def recuit(T = 100., T_min = 10., lamb = 0.99, pot = Potential(), plotting = False, inst_dict = get_dict(), print_results = True, print_log = False):
    λ = lamb
    def voisin(solution):
        """Renvoie la solution avec deux noeuds aléatories échangés"""
        i = randint(1, len(solution)-1)
        j = randint(1, len(solution)-1) 
        while j==i:
            j = randint(1, len(solution)-1)
        s_perm = [k for k in solution]
        s_perm[i],s_perm[j] = s_perm[j],s_perm[i]
        return s_perm 
    def voisin1(solution):
        """Renvoie la solution avec deux noeuds aléatories  consécutifs échangés"""
        i = randint(1, len(solution)-2)
        s_perm = [k for k in solution]
        s_perm[i],s_perm[i+1] = s_perm[i+1],s_perm[i]
        return s_perm 

    s = list(range(1, len(inst_dict)+1)) #solution initiale 1, 2, ... , n
    penalite = (len(s)+1)/2*max_dist(inst_dict)
    def cost(E, penalite = penalite):
        return 10*E[0] + E[1] + penalite*int(E[0]> 0)
    def cost1(E, penalite = penalite):
        return E[2]/len(s)
    
    s_best = s
    k = 0
    E = pot.evaluate(inst_dict, s, initial_key=1)
    e = cost(E)
    E_best = E
    e_best = cost(E_best)
    e_s = []
    e_bests = []
    print(f'cost initial : {cost(E)}')
    while T > T_min:
        s_new = voisin(s)
        E_new = pot.evaluate(inst_dict, s_new, initial_key=1)
        e_new = cost(E_new)
        Δe = e_new-e
        P = np.exp(-Δe/T)
        tirage_random = random()
        if Δe < 0 :
            if e_new < e_best:
                E_best, e_best, s_best = E_new,e_new, s_new
        if Δe < 0 or tirage_random < P :
            E, s = E_new, s_new

        #print(f'T : {T}, λ*T : {λ*T} , T-λ*T : {T - λ*T}')
        T = λ*T
        E,e, s = E_new,e_new, s_new
        k+=1
        e_s.append(e)
        e_bests.append(e_best)
        if print_log : print(f'iteration {k} : E = {int(E[0]), int(E[1])}, E_best = {int(E_best[0]), int(E_best[1])}')
    if print_results:
        print(f's_best = {s_best} \n ______________________ \n E_best = {int(E_best[0]), int(E_best[1])} en {k} iterations')
    if plotting :
        plt.plot(np.log(e_s), label = 'cost function')
        plt.plot(np.log(e_bests), label = 'best solution')
        plt.legend()
        plt.show()
    return s_best,E_best

if __name__ == "__main__":
    s,E = recuit( T = 100, T_min = 1, lamb = 0.999, plotting = True, pot = Potential(), inst_dict = get_dict(), print_results = True, print_log = False)
    #plot_sol(s, get_dict())
