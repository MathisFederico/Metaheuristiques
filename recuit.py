from utilities import *
from random import randint, random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from greedy import greedy

def recuit(inst_dict, pot, T=100 ,T_min=10, lamb=0.99, initial_key=1, **kwargs):
    
    greedy_start = kwargs.get('greedy_start', True)
    plotting = kwargs.get('plotting', True)
    print_results = kwargs.get('print_results', True)
    print_log = kwargs.get('print_log', True)

    def random_permute(solution):
        """Renvoie la solution avec deux noeuds aléatories échangés"""
        i = randint(1, len(solution)-1)
        j = randint(1, len(solution)-1)
        while j==i:
            j = randint(1, len(solution)-1)
        s_perm = [k for k in solution]
        s_perm[i],s_perm[j] = s_perm[j],s_perm[i]
        return s_perm
    
    def close_permute(solution):
        """Renvoie la solution avec deux noeuds aléatories échangés"""
        i = randint(1, len(solution)-2)
        s_perm = copy(solution)
        s_perm[i], s_perm[i+1] = s_perm[i+1], s_perm[i]
        return s_perm

    # We take a greedy initial solution
    solution = list(range(1, len(inst_dict)+1)) #solution initiale 1, 2, ... , n
    if greedy_start:
        solution = greedy(inst_dict)
    # penality = (len(solution)+1)*max_dist(inst_dict, pot)/100
    penality = 0

    def cost(E, penality=penality):
        return E[0]/10 + E[1] + E[2]*penality
    
    best_solution = solution

    evaluation = pot.evaluate(inst_dict, solution, initial_key=initial_key)
    energy = cost(evaluation)
    best_evaluation, energy_best = copy(evaluation), copy(energy)

    solutions_energy = []
    best_energies = []
    if print_log: print(f'Initial error and energy: {evaluation[2]} {energy:.1f}')

    k = 0
    while T > T_min:

        # We evaluate a neighboring solution
        new_solution = random_permute(solution)
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
        try:
            P = np.exp(-energy_gap/T)
        except RuntimeWarning:
            P = 0.0
        if random() < P :
            evaluation, energy, solution = new_evaluation, new_energy, new_solution
        
        # We decrease the temperature by the decay factor lamb
        T *= lamb

        # We save and print informations
        k += 1
        solutions_energy.append(energy)
        best_energies.append(energy_best)
        if print_log and k%1000==0: print(f'Iteration {k} :\t errors = {evaluation[2]} \t P={min(1, P):.3f} \t {energy:.1f} \t {new_energy:.1f} \t {energy_gap:.1f}')

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

    return best_solution, best_evaluation, solutions_energy, best_energies

if __name__ == "__main__":

    configs = {
        'n_sim':[1000, 50, 10, 5],
        'Tmax':[1, 1, 10, 200],
        'Tmin':[0.1], 
        'lamb':[1 - 1e-3, 1 - 1e-4, 1 - 5e-5, 1 - 1e-5],
        'greedy_start':[True],
        'plot_energies':[False],
        'nodes':[20, 40, 60, 200],
        'width':[20],
        'instance':['001'],
    }

    nb_configs = np.max([len(configs[param]) for param in configs])
    for i in range(nb_configs):
        n_sim = configs['n_sim'][i%len(configs['n_sim'])]
        Tmax = configs['Tmax'][i%len(configs['Tmax'])]
        Tmin = configs['Tmin'][i%len(configs['Tmin'])]
        lamb = configs['lamb'][i%len(configs['lamb'])]
        greedy_start = configs['greedy_start'][i%len(configs['greedy_start'])]
        plot_energies = configs['plot_energies'][i%len(configs['plot_energies'])]
        nodes = configs['nodes'][i%len(configs['nodes'])]
        width = configs['width'][i%len(configs['width'])]
        instance = configs['instance'][i%len(configs['instance'])]

        instance_name = "n{}w{}.{}.txt".format(nodes, width, instance)
        data, official_solution = extract_inst(instance_name)

        solutions_energy_list, best_energies_list = [], []
        for sim in range(n_sim):
            pot = Potential()
            solution, energy, solutions_energy, best_energies = recuit(data, pot, T=Tmax, T_min=Tmin, lamb=lamb, plotting=False, print_results=True, print_log=False, greedy_start=greedy_start)
            solutions_energy_list.append(deepcopy(solutions_energy))
            best_energies_list.append(deepcopy(best_energies))     
        
        solutions_energy_m = np.mean(np.array(solutions_energy_list), axis=0)
        best_energy_50 = np.quantile(np.array(best_energies_list), q=0.5, axis=0)
        best_energy_90 = np.quantile(np.array(best_energies_list), q=0.9, axis=0)
        best_energy = np.min(solutions_energy_list)
        
        dist_evaluations = np.linspace(0, pot.dist_count, solutions_energy_m.shape[0])

        if greedy_start: greedy_txt = 'greedy'
        else: greedy_txt = ''
        print(f'{instance_name}_{n_sim}_{greedy_txt}_{Tmax}_{Tmin}')

        idx_50 = np.argwhere(best_energy_50==best_energy)
        if len(idx_50) > 0: dist_50 = int(dist_evaluations[idx_50[0]])
        else: dist_50 = 'No'

        idx_90 = np.argwhere(best_energy_90==best_energy)
        if len(idx_90) > 0: dist_90 = int(dist_evaluations[idx_90[0]])
        else: dist_90 = 'No'
        
        print("Distances evaluations for 50% optimal:", dist_50)
        print("Distances evaluations for 90% optimal:", dist_90)

        
        plt.plot(dist_evaluations, solutions_energy_m - best_energy, label='Mean solutions energy', color='b')
        plt.plot(dist_evaluations, best_energy_90 - best_energy, label='Best fot 90% of solutions', color='orange', linestyle='--')
        plt.plot(dist_evaluations, best_energy_50 - best_energy, label='Best fot 50% of solutions', color='g', linestyle='--')

        for solutions_energy in solutions_energy_list:
            plt.plot(dist_evaluations, solutions_energy - best_energy, alpha=0.8/((n_sim)**(3/4)), color='b')

        plt.xlabel('Distance evaluations', fontsize=11)
        plt.ylabel('E - $E_{min}$', fontsize=20)
        if greedy_start: plt.suptitle('Greedy simulated annealing on instance {} '.format(instance_name), fontsize=14)
        else: plt.suptitle('Simulated annealing on instance {} '.format(instance_name), fontsize=14)
        plt.title('with $T \in [{}, {}]$ and $\lambda$ = {}'.format(Tmin, Tmax, lamb), fontsize=10)
        plt.legend(loc='upper right')
        plt.savefig(f'{instance_name[:-4]}_{n_sim}_{greedy_txt}_{dist_50}_{dist_90}.png')
        if plot_energies: plt.show()
        plt.close()