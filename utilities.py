import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Wedge

def isValid(X_dict, solution, initial_key=1):
    every_point_taken = all([key==initial_key or key in solution for key in X_dict])
    try:
        assert(every_point_taken)
    except AssertionError:
        print('Every point must be taken once !')
    return every_point_taken

def get_dict(instance_name = "n20w20.004.txt"):
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

def extract_inst(instance_name):
    X_dict = get_dict(instance_name)
    try :
        with open('DumasEtAl/' + instance_name + '.solution', "r") as f:
            sol = f.readline()
        sol = sol.split(' ')[:-1]
        sol = [int(num) for num in sol]
        return X_dict, sol
            
    except FileNotFoundError:
        return X_dict, None


def plot_sol(X_dict, solution):

    def print_circles(ax, t_tot, X, radius=0.1):
        for key in X:
            center = (X[key]['x'], X[key]['y'])
            thetai = (X[key]['ti']/t_tot) * 360
            if X[key]['tf'] < t_tot: thetaf = (X[key]['tf']/t_tot) * 360
            else: thetaf = 360 - 1e-3
            thetat = (X[key]['t']/t_tot) * 360
            is_in_window = X[key]['ti'] <= X[key]['t'] and X[key]['tf'] >= X[key]['t']
            draw_pretty_circle(center, thetai, thetaf, thetat, is_in_window=is_in_window, radius=radius, ax=ax)

        ax.axis('equal')
 
    def draw_pretty_circle(center, thetai, thetaf, thetat, is_in_window=False, radius=0.1, ax=None,
                        **kwargs):
        """
        Add two half circles to the axes *ax* (or the current axes) with the 
        specified facecolors *colors* rotated at *angle* (in degrees).
        """
        if ax is None:
            ax = plt.gca()

        thetai = (90-(thetai%360))%360
        thetaf = (90-(thetaf%360))%360
        thetat = (90-(thetat%360))%360
        
        if is_in_window: color = 'g'
        else: color = 'r'

        time_window = Wedge(center, radius, thetaf, thetai, fc=color, alpha=0.6, **kwargs)
        ax.add_artist(time_window)
        back_circle = Wedge(center, radius, 90, 90-1, fc=color, alpha=0.3, **kwargs)
        ax.add_artist(back_circle)
        blacktick = Wedge(center, radius, 90-1, 90, fc='black', alpha=1, **kwargs)
        ax.add_artist(blacktick)
        tick = Wedge(center, radius, thetat-0.5, thetat+0.5, fc='b', alpha=1, **kwargs)
        ax.add_artist(tick)

    if not isValid(X_dict, solution):
        return
    times = [0]
    prev_k = 1
    X_dict[prev_k]['t'] = 0
    for k in solution[1:]:
        time = times[-1] + np.sqrt((X_dict[k]['x']-X_dict[prev_k]['x'])**2 + \
                                    (X_dict[k]['y']-X_dict[prev_k]['y'])**2)
        X_dict[k]['t'] = time
        
        times.append(time)
        prev_k = k

    X = [X_dict[k]['x'] for k in solution]
    Y = [X_dict[k]['y'] for k in solution]
    _, ax = plt.subplots()
    for i in range(len(X)-1):
        x, dx = X[i], X[i+1]-X[i]
        y, dy = Y[i], Y[i+1]-Y[i]
        ax.arrow(x, y, dx, dy, shape='full', head_length=2, head_width=0.5, length_includes_head=True, color='black', alpha=0.3)
    print_circles(ax=ax, t_tot=times[-1], X=X_dict, radius=1.2)
    ax.scatter(X,Y)
    plt.show()

def dist(inst_dict, key1, key2):
    return np.sqrt((inst_dict[key1]['x']-inst_dict[key2]['x'])**2 +(inst_dict[key1]['y']-inst_dict[key2]['y'])**2) 
        
def max_dist(inst_dict):
    max_d= 0
    for k1 in inst_dict:
        for k2 in inst_dict:
            d = dist(inst_dict, k1,k2)
            if d >= max_d:
                max_d= d
    return max_d

if __name__ == "__main__":
    inst, sol = extract_inst("n20w20.004.txt")
    print(sol)
    sol = [1, 12, 3, 4, 20, 16, 6, 10, 5, 9, 8, 7, 17, 2, 19, 11, 18, 15, 13, 14, 21]
    if sol is not None:
        plot_sol(inst, sol)