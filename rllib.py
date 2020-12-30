from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.tune import grid_search



class CustomModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PG",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": SimpleCorridor,
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "lr": grid_search([1e-2, 1e-4, 1e-6]),
            "num_workers": 1,
        },
    )
