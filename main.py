from utilities import extract_inst, plot_sol
from energy import Potential
import numpy as np

X_dict, official_solution = extract_inst("n60w40.001.txt")
potential = Potential()
solution = [1, 4, 34, 41, 8, 30, 56, 18, 19, 33, 12, 45, 7, 59, 22, 29, 9, 57, 25, 20, 16, 13, 40, 5, 26, 53, 3, 15, 43, 17, 28, 55, 44, 58, 2, 37, 48, 54, 10, 49, 50, 23, 46, 35, 39, 47, 60, 6, 38, 27, 42, 61, 21, 31, 11, 24, 14, 32, 51, 36, 52]
score = potential.evaluate(X_dict, solution, initial_key=1)
print(potential.evaluate_count)
print(score)

if official_solution is not None:
    official_score = potential.evaluate(X_dict, official_solution, initial_key=1)
    print(official_score)
    print(np.array(official_score)-np.array(score))

plot_sol(X_dict, solution)

# n20w20.001  [1, 17, 20, 10, 19, 11, 18, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15]