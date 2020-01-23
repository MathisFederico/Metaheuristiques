from utilities import get_dict, plot_sol
from energy import Potential
import numpy as np
import operator

def time_greedy(potential, X_dict = get_dict()):
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
            distance = pot.get_dist(X_dict, prev_key, key)
            time_to_go = distance/pot.speed
            if t + time_to_go >= X_dict[key]['ti']:
                next_choices[key] = X_dict[key]['tf']
            else:
                next_choices[key] = np.inf
        choice = min(next_choices.items(), key=operator.itemgetter(1))[0]
        excluded.append(choice)
        solution.append(choice)
    return solution, pot.dist_count/n

if __name__ == "__main__":
    pot = Potential()
    X_dict = get_dict()

    solution, evaluations = time_greedy(pot, X_dict)
    print(solution, evaluations)
    print(pot.evaluate(X_dict, solution, 1))
    plot_sol(X_dict, solution)