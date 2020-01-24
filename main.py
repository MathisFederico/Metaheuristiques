from utilities import extract_inst, plot_sol, draw_animated_solution
from energy import Potential
import numpy as np

X_dict, official_solution = extract_inst("n20w20.005.txt")
potential = Potential()
string = '21 16 9 19 17 18 10 5 15 1 11 12 6 13 7 2 4 8 20 3 14'
solution = [int(key) for key in string.split(' ')]
solution = [1, 20, 12, 8, 19, 17, 14, 9, 4, 18, 3, 11, 5, 6, 16, 10, 15, 7, 21, 13, 2]
initial_key = 1
score = potential.evaluate(X_dict, solution, initial_key=initial_key)
print(solution)
print(score)

if official_solution is not None:
    official_score = potential.evaluate(X_dict, official_solution, initial_key=initial_key)
    print(official_solution)
    print(official_score)
    print(np.array(official_score)-np.array(score))

draw_animated_solution(X_dict, solution, initial_key=initial_key)
#16 9 19 17 18 10 5 15 1 11 12 6 13 7 2 4 8 20 3 14
# n20w20.001  [1, 17, 20, 10, 19, 11, 18, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15]