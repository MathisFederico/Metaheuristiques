from utilities import get_dict, plot_sol, max_dist, draw_animated_solution, extract_inst
from energy import Potential
from random import randint, random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from greedy import fifo_greedy

def recuit(inst_dict, pot, T = 100., T_min = 10., lamb = 0.99, initial_key=1, **kwargs):
    
    greedy_start = kwargs.get('greedy_start', True)
    plotting = kwargs.get('plotting', True)
    print_results = kwargs.get('print_results', True)
    print_log = kwargs.get('print_log', True)

    def voisin(solution):
        """Renvoie la solution avec deux noeuds aléatories échangés"""
        i = randint(1, len(solution)-2)
        s_perm = copy(solution)
        s_perm[i], s_perm[i+1] = s_perm[i+1], s_perm[i]
        return s_perm

    # We take a greedy initial solution
    solution = list(range(1, len(inst_dict)+1)) #solution initiale 1, 2, ... , n
    if greedy_start:
        solution = fifo_greedy(inst_dict)
    penality = (len(solution)+1)*max_dist(inst_dict, pot)/10
    print(penality)

    def cost(E, penality=penality):
        return E[0] + E[1] + E[2]*penality
    
    best_solution = solution

    evaluation = pot.evaluate(inst_dict, solution, initial_key=initial_key)
    energy = cost(evaluation)
    best_evaluation, energy_best = copy(evaluation), copy(energy)

    solutions_energy = []
    best_energies = []
    print(f'Initial energy: {energy}')

    k = 0
    while T > T_min:

        # We evaluate a neighboring solution
        new_solution = voisin(solution)
        new_evaluation = pot.evaluate(inst_dict, new_solution, initial_key=initial_key)
        new_energy = cost(new_evaluation)

        energy_gap = new_energy - energy

        # If the energy is lower we keep the exemple
        if energy_gap < 0 :
            # If the explored solution is the new best, we save it
            if new_energy < energy_best:
                best_evaluation, energy_best, best_solution = new_evaluation, new_energy, new_solution
            evaluation, energy, solution = new_evaluation, new_energy, new_solution
        
        # Else we still have a chance to keep it determined by temperature
        P = np.exp(-energy_gap/T)
        if random() < P :
            evaluation, energy, solution = new_evaluation, new_energy, new_solution
        
        # We decrease the temperature by the decay factor lamb
        T *= lamb

        # We save and print informations
        k += 1
        solutions_energy.append(energy)
        best_energies.append(energy_best)
        if print_log and k%100==0: print(f'Iteration {k} :\t errors = {evaluation[2]} \t P={P} \t {energy} \t {new_energy}')

    if print_results:
        print(f'Best = {best_solution} \nEvaluation = {best_evaluation} in {k} iterations')

    if plotting :
        plt.plot(solutions_energy, label = 'Solutions energy')
        plt.plot(best_energies, label = 'Best_energies', linestyle=':')
        # plt.plot([best_evaluation[1] for _ in best_energies], linestyle='--', label = 'Travel time', color='g')
        # plt.plot([best_evaluation[1] + best_evaluation[2]*penality for _ in best_energies], linestyle='--', label = 'Penality', color='r')
        # plt.plot([best_evaluation[1] + best_evaluation[2]*penality + best_evaluation[0] for _ in best_energies], linestyle='--', label = 'Out window', color='b')
        plt.legend()
        plt.show()

    return best_solution, best_evaluation

if __name__ == "__main__":
    pot = Potential()
    data, official_solution = extract_inst("n200w20.001.txt")
    solution, energy = recuit(data, pot, T=1000, T_min=0.1, lamb=0.999, plotting=True, print_log=True, greedy_start=True)
    draw_animated_solution(data, [solution, official_solution])
    print("Distances evaluations :", pot.dist_count)