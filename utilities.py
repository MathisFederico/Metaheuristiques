import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Wedge
from matplotlib import animation, transforms

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

def print_circles(ax, t_tot, X, radius=0.1):
    circles = []
    for key in X:
        center = (X[key]['x'], X[key]['y'])
        thetai = (X[key]['ti']/t_tot) * 360
        if X[key]['tf'] < t_tot: thetaf = (X[key]['tf']/t_tot) * 360
        else: thetaf = 360 - 1e-3
        thetat = (X[key]['t']/t_tot) * 360
        is_in_window = X[key]['ti'] <= X[key]['t'] and X[key]['tf'] >= X[key]['t']
        circles.append(draw_pretty_circle(center, thetai, thetaf, thetat, is_in_window=is_in_window, radius=radius, ax=ax))
    return circles

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
    tick = Wedge(center, radius, thetat-0.2, thetat+0.2, fc='b', alpha=1, **kwargs)
    ax.add_artist(tick)
    return [time_window, back_circle, blacktick, tick]

def plot_sol(X_dict, solution):

    if not isValid(X_dict, solution):
        return
    
    times = [0]
    prev_k = 1
    X_dict[prev_k]['t'] = 0
    for k in solution[1:]:
        time = times[-1] + dist(X_dict, k, prev_k)/1.0
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
    print(times[-1])
    print_circles(ax=ax, t_tot=times[-1], X=X_dict, radius=1.2)
    ax.scatter(X,Y)
    ax.axis('equal')
    plt.show()

def draw_animated_solution(X_dict, solution, initial_key=1, speed=1.0):

    # Check that solution is valid
    isValid(X_dict, solution, initial_key)

    def getEquidistantPoints(p1, p2, parts):
        return (np.linspace(p1[0], p2[0], parts), np.linspace(p1[1], p2[1], parts))

    # First compute the t_tot and t for each point
    t = 0
    prev_key = initial_key
    X_dict[initial_key]['t'] = t
    X_dict[initial_key]['dt'] = 0
    path_X, path_Y = [], []
    for key in solution[1:]:
        distance = dist(X_dict, key, prev_key)
        time_taken = distance/speed
        t += time_taken
        X_dict[key]['dt'] = time_taken
        X_dict[key]['t'] = t
        n_timesteps = int(round(time_taken))
        path_x, path_y = getEquidistantPoints((X_dict[prev_key]['x'], X_dict[prev_key]['y']), (X_dict[key]['x'], X_dict[key]['y']), n_timesteps)
        path_X.append(path_x)
        path_Y.append(path_y)
        prev_key = key 
    t_tot = t

    path_X = np.concatenate(path_X)
    path_Y = np.concatenate(path_Y)

    def getColor(t, t_arrived, dt, ti, tf):
        if t <= ti:
            return (0, 0, 1, 0.3)
        elif ti <= t_arrived and t_arrived <= tf and t >= t_arrived:
            return (0, 0.8, 0, 1)
        elif t >= tf:
            return (0.5, 0, 0, 1)
        else:
            G = (tf - t)/(tf - ti)
            RGBA = (1, G, 0, 1)
            return RGBA

    # Then set up the figure, the axes, and the plot element
    fig = plt.figure(1)
    ax = fig.add_subplot()


    path, = ax.plot([], [], linestyle='--', marker='', color='black', alpha=0.5)
    pretty_circles = print_circles(ax, t_tot, X_dict, radius=1.2)
    ax.set_title("Solution")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_ylim([-5,55])
    ax.set_xlim([-25,65])
    ax.axis('equal')

    # initialization function: plot the background of each frame
    def init():
        path.set_data([], [])
        return path,
    
    # animation function: this is called sequentially
    def animate(t):
        for k in range(len(X_dict)):
            key = k+1
            color = getColor(t, t_arrived=X_dict[key]['t'], dt=X_dict[key]['dt'], ti=X_dict[key]['ti'], tf=X_dict[key]['tf'])
            time_window, back_circle, _, _ = pretty_circles[k]
            back_circle.set_color(color)
            time_window.set_color(color)
            back_circle.set_alpha(color[-1]/2)
            time_window.set_alpha(color[-1])
        path.set_data(path_X[:t], path_Y[:t])

        return tuple(np.concatenate(pretty_circles)) + (path,)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=path_X.size+50, interval=30, blit=True)
    anim.save('solution.gif', codec="libx264", fps=int(1/0.030), bitrate=-1, dpi=-1)
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
    inst, sol = extract_inst("n20w20.001.txt")
    print(sol)
    sol = [1, 17, 20, 10, 19, 11, 18, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15]
    if sol is not None:
        draw_animated_solution(inst, sol)