from get_dict import get_dict
from energy import Potential


X_dict = get_dict()
potential = Potential()
solution = [1, 17, 10, 20, 18, 19, 11, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15]

print(potential.evaluate_count)
score = potential.evaluate(X_dict, solution, initial_key=1)

print(potential.evaluate_count)
print(score)