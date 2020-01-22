from utilities import plot_sol, get_dict
from energy import Potential
from gym import Env, spaces
import numpy as np
from numpy.random import choice
from copy import deepcopy

class MCTS_DAG():

    class Node:
        action_indexes = {}
        N = []
        Q = []
        P = []
        marked = False
        value = None
        
        def __init__(self, isFinal=False):
            self.isFinal = isFinal
        
        def initialiseNode(self, legal_actions, P, value):
            n_actions = np.size(np.array(legal_actions))
            self.action_indexes = {str(action):i for i,action in enumerate(legal_actions)}
            self.indexes_action = {i:str(action) for i,action in enumerate(legal_actions)}
            self.N = np.zeros((n_actions,))
            self.Q = np.zeros((n_actions,))
            self.P = P
            self.value = value

        def update(self, action, value):
            action_idx = self.action_indexes[str(action)]
            self.N[action_idx] += 1
            self.Q[action_idx] += (value - self.Q[action_idx])/self.N[action_idx]
        
        def get_UCB_action(self, c=1):
            if np.sum(self.N) == 0:
                return self.indexes_action[np.random.choice(np.array(range(self.Q.size)))]
            UCB = self.Q + c*self.P*np.sqrt(np.log(np.sum(self.N))/(0.01+self.N))
            action_idx = np.argmax(UCB)
            return self.indexes_action[action_idx]
    
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
        self.env = gym_env
        self.initial_observation = self.env.reset()
        self.tree = MCTS_DAG()
        self.model = model
        self.c = 2
        self.reward, self.done = 0, False

    def _is_leaf(self, observation):
        return self.done or np.sum(self.tree.get_node(observation).N)==0

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

        if not self.done:

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
                print("\nSimulation {}/{}".format(sim, n_simulation))
            history, observation = self._simulate()
            self._backup(history, observation)

        policy = self.build_policy(temperature=temperature)
        action = choice(range(len(policy)), p=policy)
        return action, policy
    
    def resetEnv(self, observation):
        self.env.initial_state = observation
        self.initial_observation = self.env.reset()
        self.reward, self.done = 0, False
    
    def resetTree(self, observation):
        self.tree = MCTS_DAG()
        self.add_node(observation)

class TSPTW_Env(Env):

    potential = Potential()

    def __init__(self, X_dict=get_dict()):
        self.X_dict = X_dict
        self.initial_state = [1]
        self.path = [1]
        self.time = 0
        self.action_space = spaces.Discrete(len(self.X_dict)-1)
        self.observation_space = spaces.Discrete(len(self.X_dict))
    
    def legal_actions(self, observation):
        legal_actions = []
        for action in range(len(self.X_dict)+2):
            if (action not in observation) and (action in self.X_dict):
                legal_actions.append(action)
        return legal_actions

    def isLegal(self, action:int):
        return (action not in self.path) and (action in self.X_dict)

    def step(self, action:int):
        try:
            assert(self.isLegal(action))
        except AssertionError:
            print('\n Non-Legal action : {}\n'.format(action))

        time_to_go = self.potential.get_time(self.X_dict, self.path[-1], action)
        self.path.append(action)
        observation = self.path

        self.time += time_to_go
        reward = 0
        
        done = len(self.path) == len(self.X_dict)
        if len(self.path) > 16:
            print(len(self.path))
        if done and self.potential.in_window(self.time, self.X_dict[action]['ti'], self.X_dict[action]['tf']):
            print('Solution found! : {}'.format(self.path))
            reward = 1000

        if not self.potential.in_window(self.time, self.X_dict[action]['ti'], self.X_dict[action]['tf']):
            reward = len(self.path) - len(self.X_dict) - 10
            done = True

        # print(observation, reward, done)
        return observation, reward, done, {}

    def reset(self):
        self.path = [1]
        self.time = 0
        return self.path


env = TSPTW_Env()
mcts_env = MctsEnv(env)
print(mcts_env.run_search(n_simulation=200000, temperature=0))
print(mcts_env.tree.get_node([1]).N)
print(mcts_env.tree.get_node([1]).Q)
print(TSPTW_Env.potential.dist_count)
