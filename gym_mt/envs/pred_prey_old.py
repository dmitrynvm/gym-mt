import copy
from datetime import datetime
import os
import random
import gym
from PIL import ImageColor
from gym import spaces
from .utils import draw_grid, fill_cell, draw_circle, write_cell_text


class PredPrey(gym.Env):
    '''
    Environment implements pursuit-evasion multi-agent games.
    '''
    def __init__(self, shape=(5, 5), n_preds=2, n_preys=1, n_steps=100, mode=None):
        '''
        Predator-Prey environment constructor creates a new space free of agents.
        :param shape: the shape of grid space
        :param n_preds: number of predators
        :param n_preys: number of
        :param n_steps: maximum number of steps
        :param mode: fixed, random, e
        '''
        self.shape = shape
        self.n_cells = shape[0] * shape[1]
        self.n_preds = n_preds
        self.n_preys = n_preys
        self.n_steps = n_steps
        self.count = None
        self.space = None
        self.preds = {_: None for _ in range(self.n_preds)}
        self.preys = {_: None for _ in range(self.n_preys)}
        self.state = None
        self._subdir = None

    def reset(self):
        self._grid = [[AGENTS['none'] for _ in range(self.shape[1])] for row in range(self.shape[0])]
        self.space = draw_grid(self.shape[0], self.shape[1], cell_size=CELL_SIZE, fill='white')

        for pred_i in range(self.n_preds):
            while True:
                pos = random.randint(0, self.n_cells)
                if self._is_available(pos):
                    self.preds[pred_i] = pos
                    break
            # self._grid[self.pred_pos[pred_i][0]][self.pred_pos[pred_i][1]] = AGENTS['pred'] + str(pred_i + 1)

        for prey_i in range(self.n_preys):
            while True:
                pos = random.randint(0, self.n_cells)
                if self._is_available(pos):
                    self.preys[prey_i] = pos
                    break
            # self._grid[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = AGENTS['prey'] + str(prey_i + 1)

        self._step_count = 0
        self._pred_dones = [False for _ in range(self.n_preds)]
        self._prey_alive = [True for _ in range(self.n_preys)]
        self._subdir = datetime.now().strftime('%d-%m-%Y_%I-%M-%S')
        print(self.preds)
        print(self.preys)
        print(self._grid)

    def step(self, agents_action):
        self._step_count += 1
        pass
        # self._step_count += 1
        # rewards = [self._step_cost for _ in range(self.n_preds)]
        #
        # for agent_i, action in enumerate(agents_action):
        #     if not (self._agent_dones[agent_i]):
        #         self.__update_agent_pos(agent_i, action)
        #
        # for prey_i in range(self.n_preys):
        #     if self._prey_alive[prey_i]:
        #         predator_neighbour_count, n_i = self._neighbour_agents(self.prey_pos[prey_i])
        #
        #         if predator_neighbour_count >= 1:
        #             _reward = self._penalty if predator_neighbour_count == 1 else self._prey_capture_reward
        #             self._prey_alive[prey_i] = (predator_neighbour_count == 1)
        #
        #             for agent_i in range(self.n_preds):
        #                 rewards[agent_i] += _reward
        #
        #         prey_move = None
        #         if self._prey_alive[prey_i]:
        #             # 5 trails : we sample next move and check if prey (smart) doesn't go in neighbourhood of predator
        #             for _ in range(5):
        #                 _move = np.random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
        #                 if self._neighbour_agents(self.__next_pos(self.prey_pos[prey_i], _move))[0] == 0:
        #                     prey_move = _move
        #                     break
        #             prey_move = 4 if prey_move is None else prey_move  # default is no-op(4)
        #
        #         self.__update_prey_pos(prey_i, prey_move)
        #
        # if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
        #     for i in range(self.n_preds):
        #         self._agent_dones[i] = True
        #
        # for i in range(self.n_preds):
        #     self._total_episode_reward[i] += rewards[i]
        # return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

    def render(self):
        img = copy.copy(self.space)
        #
        # for agent_i in range(self.n_preds):
        #     draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
        #     write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
        #                     fill='white', margin=0.4)
        #
        # for prey_i in range(self.n_preys):
        #     if self._prey_alive[prey_i]:
        #         draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
        #         write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
        #                         fill='white', margin=0.4)
        # self._curstep += 1
        return img

    def export(self, pardir='results', prefix='screen', digits=2):
        curdir = os.path.join(pardir, self._subdir)
        if not os.path.exists(curdir):
            os.makedirs(curdir)
        img = self.render()
        file = '{}_{}.png'.format(prefix, str(self._step_count).rjust(digits, '0'))
        path = os.path.join(curdir, file)
        img.save(path)

    # def _is_valid(self, pos):
    #     return
    #
    # def _is_vacant(self, pos):
    #     return self._grid[pos[0]][pos[1]] == AGENTS['none']

    def _is_available(self, pos):
        return 0 <= pos < self.n_cells


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTIONS = {
    0: "D",
    1: "L",
    2: "U",
    3: "R",
    4: "S",
}

AGENTS = {
    'pred': 'A',
    'prey': 'B',
    'wall': 'W',
    'none': 'O'
}
