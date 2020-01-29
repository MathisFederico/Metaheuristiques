from utilities import *
from gym import Env, spaces
import numpy as np
from numpy.random import choice
from copy import deepcopy

class MCTS_DAG():

    class Node:
        action_indexes = {}
        UCB = []
        N = []
        Q = []
        P = []
        marked = False
        value = None
        
        def __init__(self, isFinal=False):
            self.isFinal = isFinal
        
        def initialiseNode(self, legal_actions, P, value):
            legal_actions = np.array(legal_actions)
            n_actions = np.size(legal_actions)
            self.action_indexes = {str(action):i for i,action in enumerate(legal_actions)}
            self.actions = [str(action) for action in legal_actions]
            self.N = np.zeros((n_actions,))
            self.Q = np.zeros((n_actions,))
            self.P = P
            self.value = value
            if len(legal_actions) == 0: 
                self.isFinal = True
                self.value = -10

        def update(self, action, value):
            action_idx = self.action_indexes[str(action)]
            self.N[action_idx] += 1
            self.Q[action_idx] += (value - self.Q[action_idx])/self.N[action_idx]
        
        def get_UCB_action(self, c=1):
            if np.sum(self.N) == 0:
                return self.actions[np.random.choice(np.array(range(self.Q.size)))]
            self.UCB = self.Q + c*self.P*np.sqrt(np.log(np.sum(self.N))/(1+self.N))
            action_idx = np.argmax(self.UCB)
            return self.actions[action_idx]
    
    def __init__(self):
        self.nodes = {}

    def render(self):
        print({node:(self.nodes[node].N) for node in self.nodes})
        print({node:(self.nodes[node].Q) for node in self.nodes})

    def add_node(self, observation, legal_actions, P=None, value=None):
        n_actions = np.size(legal_actions)
        node = self.Node()
        if P is None:
            P = np.ones((n_actions,))
        node.initialiseNode(legal_actions, P, value)
        self.nodes[self.hash(observation)] = node
    
    def get_node(self, observation):
        return self.nodes[self.hash(observation)]
    
    def get_childrens(self, observation):
        return self.nodes[self.hash(observation)].childrens
    
    def is_in_nodes(self, observation):
        return self.hash(observation) in self.nodes
    
    def hash(self, observation):
        return str(observation)

class MctsEnv():

    def __init__(self, gym_env, model=None):
        self.env = deepcopy(gym_env)
        self.env.early_end = True
        self.initial_observation = self.env.reset()
        self.tree = MCTS_DAG()
        self.model = model
        self.c = 3
        self.reward, self.done = 0, False

    def _is_leaf(self, observation):
        return self.done or np.sum(self.tree.get_node(observation).N)==0 or self.tree.get_node(observation).isFinal

    def _simulate(self):
        self.done = False
        self.reward = 0
        history = []
        observation = self.env.reset()
        
        if not self.tree.is_in_nodes(observation):
            # print("Added at the beginning of simulation : ", observation)
            self.add_node(observation)
        
        while not self._is_leaf(observation):

            # Action selection using upper confidence bound
            action = int(self.tree.get_node(observation).get_UCB_action(c=self.c))
            history.append(deepcopy((observation, action)))

            # Environement step
            # print("Observation : ", observation)
            # print("Action : ", action, self.tree.get_node(observation).Q)
            observation, reward, self.done, _ = self.env.step(action)
            self.reward += reward

            if not self.tree.is_in_nodes(observation):
                # print("Added at the while in simulation : ", observation)
                self.add_node(observation)

        if not (self.done or self.tree.get_node(observation).isFinal):

            # Last action selection using upper confidence bound
            action = int(self.tree.get_node(observation).get_UCB_action())
            history.append(deepcopy((observation, action)))

            # Last environement step
            # print("Last step taken ! {} {}".format(observation, self.tree.get_node(observation).N))
            # print("At :", observation, "Action :", action, self.tree.get_node(observation).Q)
            # print(self.env.legal_actions(observation))
            # self.tree.render()
            observation, reward, self.done, _ = self.env.step(action)
            self.reward += reward

            # Add extanded node
            # print("Added by extanding : ", observation)
            self.add_node(observation)

        return history, observation
    
    def add_node(self, observation):
        if self.tree.is_in_nodes(observation):
            # print("{} is already in nodes !".format(observation))
            return
        P, value = None, None
        if self.model is not None:
            x = np.array([observation])
            P, value = self.model.predict(x)
            P, value = P[0], value[0, 0]
        self.tree.add_node(observation, self.env.legal_actions(observation), P, value)

    def _backup(self, history, last_observation):
        # Get last_observation value
        action_space_size = self.env.action_space.n
        if self.tree.get_node(last_observation).value is not None:
            value = self.tree.get_node(last_observation).value
        else:
            if self.model is None:
                _, value = np.ones(action_space_size), self.reward
            else:
                x = np.array([last_observation])
                _, value = self.model.predict(x)
                value = value[0,0]
                # print("After sim : For leaf : ", x[0], " value is ", value)
        
        # Update nodes in history
        for observation, action in history:
            node = self.tree.get_node(observation)
            node.update(action, value)
    
    def build_policy(self, temperature=1):
        root = self.tree.get_node(self.initial_observation)
        N = np.array(root.N)
        if np.size(N) == 0: return 1
        if temperature > 0:
            P = np.divide(np.power(N, 1/temperature), np.sum(np.power(N, 1/temperature)))
            if np.any(np.isnan(P)):
                P = np.zeros((N.size,))
                P[np.argmax(N)] = 1
        else:
            P = np.zeros((N.size,))
            P[np.argmax(N)] = 1
        return P

    def run_search(self, n_simulation=1600, temperature=1):
        for sim in range(n_simulation):
            if 100*sim/n_simulation % 10 == 0:
                # print("\nSimulation {}/{}".format(sim, n_simulation))
                pass
            history, observation = self._simulate()
            self._backup(history, observation)

        policy = self.build_policy(temperature=temperature)
        actions = self.tree.get_node(self.initial_observation).actions
        
        if len(actions) == 0:
            print(self.initial_observation, actions)
            actions = [[action, self.env.X_dict[action]['tf']-self.env.time] for action in range(1, len(self.env.X_dict)+1) if action not in self.initial_observation]
            actions = np.array(actions)
            action = actions[actions[:,1].argsort()][0, 0]
            return action, policy
        action = choice(actions, p=policy)
        # print("Choice", self.initial_observation, actions, action)
        return action, policy
    
    def resetEnv(self, observation):
        # print("Initial state : ", observation)
        self.env.initial_state = observation
        self.initial_observation = self.env.reset()
        self.reward, self.done = 0, False
    
    def resetTree(self, observation):
        self.tree = MCTS_DAG()
        self.add_node(observation)

class TSPTW_Env(Env):

    potential = Potential()
    early_end = False

    def __init__(self, X_dict):
        self.X_dict = X_dict
        self.initial_state = [1]
        self.path = deepcopy(self.initial_state)
        self.time = 0
        self.action_space = spaces.Discrete(len(self.X_dict)-1)
        self.observation_space = spaces.Discrete(len(self.X_dict))
    
    def legal_actions(self, observation):
        legal_actions = []
        for action in range(len(self.X_dict)+2):
            if (action not in observation) and (action in self.X_dict):
                t = self.time + self.potential.get_time(self.X_dict, observation[-1], action, self.time)
                if t >= self.X_dict[action]['ti'] and t <= self.X_dict[action]['tf']:
                    legal_actions.append(action)
        return legal_actions

    def isLegal(self, action:int):
        return (action not in self.path) and (int(action) in self.X_dict)

    def step(self, action:int):
        real_action = int(action)
        try:
            assert(self.isLegal(real_action))
        except AssertionError:
            raise AssertionError('\n Non-Legal action : {}\n'.format(real_action))

        time_to_go = self.potential.get_time(self.X_dict, self.path[-1], real_action, self.time)
        self.path.append(real_action)
        observation = self.path

        self.time += time_to_go
        reward = 0
        
        done = len(self.path) == len(self.X_dict)
        if done:
            errors = self.potential.evaluate(self.X_dict, self.path)[2]
            if errors == 0:
                global solution_found
                solution_found = deepcopy(self.path)
                # print('Solution found! : {}'.format(solution_found))
                # print('N_distances: {}'.format(self.potential.dist_count))
                return
            else:
                reward = -errors**2

        if not self.potential.in_window(self.time, self.X_dict[real_action]['ti'], self.X_dict[real_action]['tf']):
            reward += -.1*self.potential.distance_to_window(self.time, self.X_dict[real_action]['ti'], self.X_dict[real_action]['tf'])
            reward += -10
    
        
        # if len(self.legal_actions(observation)) == 0 and not done:
        #     if self.early_end:
        #         done = True
        #     reward += -10

        # print(observation, reward, done)
        return observation, reward, done, {}

    def reset(self):
        self.path = deepcopy(self.initial_state)
        self.time = 0
        return self.path

def mc_backup(history, mcts_tree, G, alpha=0.1, decay=0.3):
    for i, observation in enumerate(history):
        print("MC on ", observation[-1], mcts_tree.get_node(observation).value, G*decay**i)
        if mcts_tree.get_node(observation).value is not None:
            mcts_tree.get_node(observation).value += alpha*(G*decay**i - mcts_tree.get_node(observation).value)
        else:
            mcts_tree.get_node(observation).value = G*decay**i


if __name__ == "__main__":
    nodes = 20
    width = 20
    instance = '004'
    X_dict, official_sol = extract_inst("n{}w{}.{}.txt".format(nodes, width, instance))
    err = nodes
    env = TSPTW_Env(X_dict)
    mcts_env = MctsEnv(env)
    observation = env.reset()
    done = False
    hist = [deepcopy(observation)]
    G = 0
    while not done:
        n_simulation = int(10000/np.log(1+len(observation)))
        mcts_env.resetEnv(observation)
        try: 
            action, _ = mcts_env.run_search(n_simulation=n_simulation, temperature=0)
            np.set_printoptions(precision=2, suppress=True)
            print(action, mcts_env.tree.get_node(observation).actions, mcts_env.tree.get_node(observation).UCB,
                    mcts_env.tree.get_node(observation).N/np.sum(mcts_env.tree.get_node(observation).N), mcts_env.tree.get_node(observation).Q)
            
            observation, reward, done, _ = env.step(action)
            if not done:
                hist.append(deepcopy(observation))
            G += reward
        except TypeError:
            done = True
            observation = solution_found
            break
        # mc_backup(hist, mcts_env.tree, G)

    print('Final solution : {}'.format(observation))    
    print('Distance evaluations:', TSPTW_Env.potential.dist_count) 
    a, b, err = Potential().evaluate(X_dict, observation)
    print(a, b, err)
    draw_animated_solution(X_dict, [observation, official_sol], save=False)
