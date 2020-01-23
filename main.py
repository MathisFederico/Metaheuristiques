from utilities import extract_inst, plot_sol
from energy import Potential
import numpy as np

X_dict, official_solution = extract_inst("n20w20.003.txt")
potential = Potential()
solution = [1, 6, 20, 14, 2, 9, 17, 5, 10, 4, 8, 16, 15, 18, 19, 13, 11, 3, 7, 21, 12]
score = potential.evaluate(X_dict, solution, initial_key=1)
print(potential.evaluate_count)
print(score)

if official_solution is not None:
    official_score = potential.evaluate(X_dict, official_solution, initial_key=1)
    print(official_score)
    print(np.array(official_score)-np.array(score))

plot_sol(X_dict, solution)

# n20w20.001  [1, 17, 20, 10, 19, 11, 18, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15]