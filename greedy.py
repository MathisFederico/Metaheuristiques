from utilities import get_dict, plot_sol, draw_animated_solution
from energy import Potential
import numpy as np
import operator
from copy import copy

def time_greedy(potential, X_dict):
    pot = potential
    solution = [1]

    n = len(X_dict)
    excluded = []
    prev_key = 1
    t = 0
    for _ in range(2, n+1):
        next_choices = {}
        for key in range(2, n+1):
            if key in excluded:
                continue
            time_to_go = pot.get_time(X_dict, prev_key, key)
            if t + time_to_go >= X_dict[key]['ti']:
                next_choices[key] = time_to_go
            else:
                next_choices[key] = np.inf
        choice = min(next_choices.items(), key=operator.itemgetter(1))[0]
        excluded.append(choice)
        solution.append(choice)
    return solution, pot.dist_count/n

def fifo_greedy(X_dict):
    fifo = np.array([[key, X_dict[key]['ti'] + (X_dict[key]['tf']-X_dict[key]['ti'])/2] for key in X_dict if key != 1])
    fifo = fifo[fifo[:,1].argsort()]
    print(fifo)
    solution = np.concatenate([[1], fifo[:, 0]])
    return solution

def fifo_dist_greedy(X_dict, lenght=3, potential=Potential()):
    fifo = np.array([[key, X_dict[key]['ti']] for key in X_dict if key != 1])
    fifo = fifo[fifo[:,1].argsort()]
    pre_solution = np.concatenate([[1], fifo[:, 0]])
    dists = 1e6 + np.zeros((len(X_dict), lenght, 3))
    for i, key in enumerate(pre_solution):
        for j, next_key in enumerate(pre_solution[key:key+lenght]):
            dists[i, j] = copy(np.array([key, next_key, potential.get_time(X_dict, key, next_key)]))
    
    np.set_printoptions(suppress=True)
    for i, key in enumerate(pre_solution):
        print(dists[i])
        sorted_dist = dists[i][dists[i][:,2].argsort()]
        print(sorted_dist)
        next_key = int(sorted_dist[0, 1])
        print(next_key)
    
    print(potential.dist_count)
    return solution

if __name__ == "__main__":
    pot = Potential()
    X_dict = get_dict("n20w20.001.txt")

    # solution, evaluations = time_greedy(pot, X_dict)

    solution = fifo_greedy(X_dict)
    print(solution)
    print(pot.evaluate(X_dict, solution, 1))
    draw_animated_solution(X_dict, solution, save=False)