from utilities import *
import numpy as np

nodes = 20
width = 20
instance = '004'
X_dict, official_solution = extract_inst("n{}w{}.{}.txt".format(nodes, width, instance))
potential = Potential()
# string = '21 16 9 19 17 18 10 5 15 1 11 12 6 13 7 2 4 8 20 3 14'
# solution = [int(key) for key in string.split(' ')]
solution = [1, 12, 4, 3, 20, 8, 16, 10, 9, 6, 7, 11, 15, 5, 13, 17, 19, 14, 21, 18, 2]
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