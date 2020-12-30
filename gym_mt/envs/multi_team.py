from gym import Env, spaces
from random import randint
import numpy as np
from scipy.spatial.distance import cityblock


class MultiTeam(Env):
    def __init__(self, teams, grid, n_steps=100):
        self.teams = teams
        self.grid = grid
        self.n_steps = n_steps
        self.team_shape = [team.size for team in teams]
        self.grid_shape = grid.shape
        self.n_teams = len(teams)
        self.n_agents = int(np.sum(self.team_shape))
        self.n_actions = len(grid.actions)
        self.n_cells = int(np.prod(self.grid_shape))
        assert self.n_teams > 1, \
            'Number of teams should be greater than one'
        assert self.n_agents <= self.n_cells, \
            'Number of agents should be less than number of cells'
        self.count = None
        self.state = [
            [None] * self.team_shape[team_i] for team_i in range(self.n_teams)
        ]
        self.action_space = spaces.Tuple([
            spaces.Tuple([spaces.Discrete(self.n_actions)] * self.team_shape[team_i])
            for team_i in range(self.n_teams)
        ])
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(self.team_shape[team_i]) for team_i in range(self.n_teams)]
        )
        # self.observation_space = spaces.Tuple([
        #     spaces.Tuple([spaces.Discrete(self.n_cells)] * self.team_shape[team_i])
        #     for team_i in range(self.n_teams)
        # ])
        self.reset()

    def reset(self):
        self.count = 0
        for team_i in range(self.n_teams):
            for agent_i in range(self.team_shape[team_i]):
                while True:
                    state = randint(0, self.n_cells - 1)
                    if self.is_available(state):
                        self.state[team_i][agent_i] = state
                        break
        return self.state

    def step(self, action):
        assert self.count is not None, \
            'Reset the environment before executing the step method'
        self.count += 1

        for team_i in range(self.n_teams):
            for agent_i in range(self.team_shape[team_i]):
                self.update(team_i, agent_i, action[team_i][agent_i])

        observation = self.get_observation()
        reward = self.get_reward()
        done = True
        info = {}
        return observation, reward, done, info

    def render(self, mode):
        pass

    def close(self):
        pass

    def is_valid(self, state):
        return 0 <= state < self.n_cells

    def is_vacant(self, state):
        return not max([state in team_state for team_state in self.state])

    def is_available(self, state):
        return self.is_valid(state) and self.is_vacant(state)

    def state2coord(self, state):
        return state % self.grid_shape[0], int(state / self.grid_shape[0])

    def coord2state(self, coord):
        return coord[0] + coord[1] * self.grid_shape[0]

    def update(self, team_i, agent_i, action_i):
        state_curr = self.state[team_i][agent_i]
        x_curr, y_curr = self.state2coord(state_curr)
        x_next, y_next = None, None
        label = self.grid.actions[action_i]
        if label == 'stop':
            x_next, y_next = x_curr, y_curr
        elif label == 'up':
            x_next, y_next = x_curr + 1, y_curr
        elif label == 'down':
            x_next, y_next = x_curr - 1, y_curr
        elif label == 'right':
            x_next, y_next = x_curr, y_curr + 1
        elif label == 'left':
            x_next, y_next = x_curr, y_curr - 1
        else:
            raise Exception('Action not found!')
        state_next = self.coord2state([x_next, y_next])
        if self.is_available(state_next):
            self.state[team_i][agent_i] = state_next

    def __repr__(self):
        return 'Env({}, {})'.format(self.teams, self.grid)

    def __str__(self):
        return 'Env({}, {})'.format(self.teams, self.grid)

    def binvec(self, states):
        return [1 if i in states else 0 for i in range(self.n_cells)]

    def get_distance_matrix(self, team_i, team_j):
        matrix = np.zeros(self.team_shape, int)
        for agent_i in range(self.team_shape[team_i]):
            for agent_j in range(self.team_shape[team_j]):
                coord_i = self.state2coord(self.state[team_i][agent_i])
                coord_j = self.state2coord(self.state[team_j][agent_j])
                distance = cityblock(coord_i, coord_j)
                matrix[agent_i, agent_j] = distance
        return matrix

    def get_observation(self):
        observation = []
        # for team_i in range(self.n_teams):
        #     for agent_i in range(self.team_shape[team_i]):
        #         team_state = self.state[team_i][:]
        #         agent_state = self.state[team_i][agent_i]
        #         team_state.remove(agent_state)
        #         observation.append(self.binvec(team_state))
        # return observation
        return self.state

    def get_reward(self):
        reward = 0
        team_i = 0
        team_j = 1

        distance_matrix = self.get_distance_matrix(team_i, team_j)

        for agent_i in range(self.team_shape[team_i]):
            reward += min(distance_matrix[agent_i, :])

        for agent_j in range(self.team_shape[team_j]):
            reward -= min(distance_matrix[agent_j, :])

        return reward
