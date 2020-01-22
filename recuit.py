from utilities import get_dict, plot_sol, max_dist, dist
from energy import Potential
from random import randint

def recuit(T, T_min, λ, pot = Potential(), inst_dict = get_dict(), print_results = True, print_log = False):
    def voisin(solution):
        """Renvoie la solution avec deux noeuds aléatories échangés"""
        i = randint(1, len(solution)+1)
        j = randint(1, len(solution)+1)
        while j==i:
            j = randint(1, len(solution)+1)
        s_perm = [k for k in solution]
        s_perm[i],s_perm[j] = s_perm[j],s_perm[i]
        return s_perm 

    
    s = list(range(1, len(inst_dict)+1)) #solution initiale 1, 2, ... , n
    penalite = (len(s)+1)*max_dist(inst_dict)
    E = pot.evaluate(inst_dict, s, initial_key=1)
    s_best = s
    E_best = E
    k=0

    while T > T_min:
        s_new = voisin(s)
        E_new = pot.evaluate(inst_dict, s_new, initial_key=1)
        ΔE = E[0]*penalite + E[1]
        P = np.exp(-ΔE/T)
        tirage_random = random()
        if E_new < E :
            if E_new < E_best:
                E_best, s_best = E_new, s_new
        if E_new < E_best or tirage_random < P :
            E, s = E_new, s_new

        T = T - λ*T
        E, s = E_new, s_new
        k+=1
        if print_log : print(f'iteration {k} : E = {int(E)}, E_best = {int(E_best)}')
    if print_results:
        print(f's_best = {s_best} \n ______________________ \n E_best = {int(E_best)} en {k} iterations')
    return s_best,E_best
