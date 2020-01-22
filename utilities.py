import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Wedge

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

    def print_circles(ax, t_tot, X, radius=0.1):
        print(t_tot)
        for key in X:
            center = (X[key]['x'], X[key]['y'])
            thetai = (X[key]['ti']/t_tot) * 360
            thetaf = (X[key]['tf']/t_tot) * 360
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
        print(thetai, thetaf, thetat)

        
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

    times = [0]
    prev_k = 1
    inst_dict[prev_k]['t'] = 0
    for k in solution[1:]:
        time = times[-1] + np.sqrt((inst_dict[k]['x']-inst_dict[prev_k]['x'])**2 + \
                                    (inst_dict[k]['y']-inst_dict[prev_k]['y'])**2)
        inst_dict[k]['t'] = time
        
        times.append(time)
        prev_k = k

    X = [inst_dict[k]['x'] for k in solution]
    Y = [inst_dict[k]['y'] for k in solution]
    _, ax = plt.subplots()
    for i in range(len(X)-1):
        x, dx = X[i], X[i+1]-X[i]
        y, dy = Y[i], Y[i+1]-Y[i]
        ax.arrow(x, y, dx, dy, shape='full', head_length=2, head_width=0.5, length_includes_head=True, color='black', alpha=0.3)
    print_circles(ax=ax, t_tot=times[-1], X=inst_dict, radius=1.2)
    ax.scatter(X,Y)
    plt.show()

if __name__ == "__main__":
    inst = get_dict()
    
    sol = ['1' ,'17', '10', '20', '18', '19', '11', '6', '16', '2', '12', '13', '7', '14', '8', '3', '5', '9', '21', '4' ,'15' ]
    sol = [int(s) for s in sol]
    plot_sol(sol, inst)
