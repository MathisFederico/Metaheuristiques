import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def get_dict(instance_name = "n20w20.001.txt"):
    """Il faut que les instances soient dans metah/DumasEtAl"""
    with open('DumasEtAl/' + instance_name, "r") as f:
        d = {}
        for _ in range(6):
            f.readline()
        for row in f:
            r = [x for x in row.split(' ') if x != '']
            key = int(r[0])
            if key != 999:
                d[key] = {'x':int(float(r[1])), 'y':int(float(r[2])), 'ti':int(float(r[4])), 'tf':int(float(r[5])), 'demand':int(float(r[3])), 't_service':int(float(r[-1][:-2]))}
    return d


def plot_sol(solution, inst_dict):
    times = [0]
    prev_k = 1
    for k in solution[1:]:
        time = times[-1] + np.sqrt((inst_dict[k]['x']-inst_dict[prev_k]['x'])**2 +(inst_dict[k]['y']-inst_dict[prev_k]['y'])**2) 
        
        times.append(time)
        prev_k = k
    label = [] 
    for (t,k) in zip(times, solution):
        valid = (t >= inst_dict[k]['ti'] and t <= inst_dict[k]['tf'])
        if valid : 
            label.append(1)
        else:
            label.append(0)
        
    colors = ['red','green']

    X = [inst_dict[k]['x'] for k in solution]
    Y = [inst_dict[k]['y'] for k in solution]
    
    for i in range(len(X)-1):
        x, dx = X[i], X[i+1]-X[i]
        y, dy = Y[i], Y[i+1]-Y[i]
        plt.arrow(x, y, dx, dy)
    # plt.plot(x, y, 'xb-')
    plt.scatter(X, Y, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    
    plt.show()

if __name__ == "__main__":
    print(get_dict())
    inst = get_dict()
    
    sol = ['1' ,'17', '10', '20', '18', '19', '11', '6', '16', '2', '12', '13', '7', '14', '8', '3', '5', '9', '21', '4' ,'15' ]
    sol = [int(s) for s in sol]
    plot_sol(sol, inst)
