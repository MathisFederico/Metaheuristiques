import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Wedge
from matplotlib import animation, transforms
from copy import deepcopy

def isValid(X_dict, solution, initial_key=1):
    every_point_taken = all([key==initial_key or key in solution for key in X_dict])
    try:
        assert(every_point_taken)
    except AssertionError:
        print('Every point must be taken once !')
    return every_point_taken


class Potential():

    evaluate_count = 0
    dist_count = 0
    speed = 1.0

    def get_time(self, X_dict:dict, start_key:int, end_key:int, t:float):
        distance = self.get_dist(X_dict, start_key, end_key)
        return max(distance/self.speed, X_dict[end_key]['ti'] - t)

    def get_dist(self, X_dict:dict, key1:int, key2:int):
        self.dist_count += 1
        distance = ( (X_dict[key1]['x'] - X_dict[key2]['x'])**2 \
                    + (X_dict[key1]['y'] - X_dict[key2]['y'])**2 )**.5
        return int(distance)

    @staticmethod
    def in_window(t, ti, tf):
        return (t >= ti) and (t <= tf)

    @staticmethod
    def distance_to_window(t, ti, tf):
        if t < ti:
            return ti - t
        else:
            return t - tf

    def evaluate(self, X_dict:dict, solution:list, initial_key=1):

        # Check that every point is taken once
        if not isValid(X_dict, solution, initial_key):
            return (-1, -1, -1)

        self.evaluate_count += 1

        oow_cost, time, errors = 0, 0, 0
        prev_key = initial_key
        
        for key in solution:

            # Pass the first key if it is the initial key
            if key == prev_key:
                continue

            # Compute time taken
            time_taken = self.get_time(X_dict, prev_key, key, time)
            time += time_taken
            
            # Compute out of window cost if not in window
            ti, tf = X_dict[key]['ti'], X_dict[key]['tf']
            if not self.in_window(time, ti, tf):
                errors +=1
                oow_cost += self.distance_to_window(time, ti, tf)

            # Update the previous key
            prev_key = key    
            
        return oow_cost, time, errors


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


def plot_sol(X_dict, solution, potential=Potential()):

    if not isValid(X_dict, solution):
        return
    
    times = [0]
    prev_k = 1
    X_dict[prev_k]['t'] = 0
    for k in solution[1:]:
        time = times[-1] + potential.get_dist(X_dict, prev_k, k)/1.0
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


def draw_animated_solution(instance_dict:dict, solutions:list, potential=Potential(), initial_key=1, speed=1.0, save=False):
    if type(solutions[0]) == int:
        solutions = [solutions]

    # Check that every solution is valid
    for solution in solutions:
        isValid(instance_dict, solution, initial_key)

    def getEquidistantPoints(p1, p2, parts):
        return (np.linspace(p1[0], p2[0], parts), np.linspace(p1[1], p2[1], parts))

    # First compute the t_tot and t for each point
    sol_dicts = []
    sol_paths = []
    sol_tots = []
    for solution in solutions:
        sol_dict = deepcopy(instance_dict)
        t = 0
        prev_key = initial_key
        sol_dict[initial_key]['t'] = t
        path_X, path_Y = [], []
        for key in solution[1:]:
            distance = potential.get_dist(instance_dict, prev_key, key)
            time_taken = max(distance/speed, sol_dict[key]['ti'] - t)
            n_timesteps_path = max(1, int(round(distance/speed)))
            n_timesteps_wait = int(max(0, sol_dict[key]['ti'] - t - distance/speed))
            t += time_taken
            sol_dict[key]['t'] = t
            path_x, path_y = getEquidistantPoints((sol_dict[prev_key]['x'], sol_dict[prev_key]['y']), (sol_dict[key]['x'], sol_dict[key]['y']), n_timesteps_path)
            if n_timesteps_wait > 0:
                path_x = np.concatenate([path_x, [path_x[-1] for _ in range(n_timesteps_wait)]])
                path_y = np.concatenate([path_y, [path_y[-1] for _ in range(n_timesteps_wait)]])
            path_X.append(path_x)
            path_Y.append(path_y)
            prev_key = key 
        
        t_tot = t
        path_X = np.concatenate(path_X)
        path_Y = np.concatenate(path_Y)

        sol_tots.append(deepcopy(t_tot))
        sol_dicts.append(deepcopy(sol_dict))
        sol_paths.append(deepcopy((path_X, path_Y)))

    def getColor(t, t_arrived, ti, tf):
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
    fig, axs = plt.subplots(nrows=1, ncols=len(solutions), sharey=True)
    paths = []
    pretty_circles_list = []
    titles = ["Solution", "Official Solution"]
    for i, sol_dict in enumerate(sol_dicts):
        t_tot = sol_tots[i]
        if len(solutions) > 1: ax = axs[i]
        else: ax = axs
        paths.append(ax.plot([], [], linestyle='--', marker='', color='black', alpha=0.5)[0])
        pretty_circles_list.append(print_circles(ax, t_tot, sol_dict, radius=1.2))
        ax.set_title(titles[i])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_ylim([-5,55])
        ax.set_xlim([-25,65])
        ax.axis('equal')

    # initialization function: plot the background of each frame
    def init():
        for i, _ in enumerate(sol_dicts):
            paths[i].set_data([], [])
        return tuple(paths)
    
    # animation function: this is called sequentially
    def animate(t):
        for i, sol_dict in enumerate(sol_dicts):
            for k in range(len(sol_dict)):
                key = k+1
                color = getColor(t, t_arrived=sol_dict[key]['t'], ti=sol_dict[key]['ti'], tf=sol_dict[key]['tf'])
                time_window, back_circle, _, _ = pretty_circles_list[i][k]
                back_circle.set_color(color)
                time_window.set_color(color)
                back_circle.set_alpha(color[-1]/2)
                time_window.set_alpha(color[-1])
            path_X, path_Y = sol_paths[i]
            paths[i].set_data(path_X[:t], path_Y[:t])

        return tuple(np.concatenate([np.concatenate(pretty_circles_list[i]) for i in range(len(solutions))])) + tuple(paths)

    nb_frames = np.max([sol_paths[i][0].size+50 for i in range(len(solutions))])
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nb_frames, interval=30, blit=True)
    plt.show()
    if save: anim.save('solution.gif', codec="libx264", fps=int(1/0.030), bitrate=-1, dpi=-1)


def max_dist(inst_dict, potential=Potential()):
    max_d= 0
    for k1 in inst_dict:
        for k2 in inst_dict:
            d = potential.get_dist(inst_dict, k1, k2)
            if d >= max_d:
                max_d= d
    return max_d


if __name__ == "__main__":
    inst, official_sol = extract_inst("n20w20.001.txt")
    print("Official:", official_sol)
    sol = [1, 17, 15, 20, 11, 19, 6, 18, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 10]
    print("Local:",sol)
    if official_sol is not None:
        draw_animated_solution(inst, [sol, official_sol], save=False)
    else: draw_animated_solution(inst, [sol])