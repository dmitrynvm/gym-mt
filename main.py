import gym
import gym_mt

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

import ray
from ray import tune
from ray.tune import grid_search

env = gym.make('MultiTeam-4v3-5x5-v0')
action = env.action_space.sample()
# print(env.state)
observation, reward, done, info = env.step(action)
# print(observation, reward, done, info)
env.close()


class ShallowModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("shallow_model", ShallowModel)
    tune.run(
        "PG",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": "MultiTeam-4v3-5x5-v0",
            "model": {
                "custom_model": "shallow_model",
            },
            "vf_share_layers": True,
            "lr": grid_search([1e-2, 1e-4, 1e-6]),
            "num_workers": 1,
        },
    )
