# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
from crowd_nav.policy.helpers import mlp


class ValueEstimator(nn.Module):
    def __init__(self, config, graph_model):
        super().__init__()
        self.graph_model = graph_model
        self.value_network = mlp(config.gcn.X_dim, config.model_predictive_rl.value_network_dims)

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation
        state = self.trans_features(state)
        state_embedding = self.graph_model(state)[:, 0, :]
        value = self.value_network(state_embedding)
        return value

    # robot: px, py, vx, vy, radius, gx, gy, v_pref, theta. 训练时的shape为[100, 1, 9]
    # human: px, py, vx, vy, radius. 训练时的shape为[100, 5, 5]
    def trans_features(self, state_tensor):
        robot_state_tensor = state_tensor[0]
        human_states_tensor = state_tensor[1]
        dx = (robot_state_tensor[..., 5:6] - robot_state_tensor[..., 0:1])
        dy = (robot_state_tensor[..., 6:7] - robot_state_tensor[..., 1:2])
        rot = torch.atan2(dy, dx)
        dg = torch.norm(torch.cat([dx, dy], dim=-1), 2, dim=-1, keepdim=True)
        v_pref = robot_state_tensor[..., 7:8]
        vx = robot_state_tensor[..., 2:3] * torch.cos(rot) + robot_state_tensor[..., 3:4] * torch.sin(rot)
        vy = robot_state_tensor[..., 3:4] * torch.cos(rot) - robot_state_tensor[..., 2:3] * torch.sin(rot)
        radius = robot_state_tensor[..., 4:5]
        theta = (robot_state_tensor[..., -1:] - rot) % (2 * np.pi)

        vx1 = human_states_tensor[..., 2:3] * torch.cos(rot) + human_states_tensor[..., 3:4] * torch.sin(rot)
        vy1 = human_states_tensor[..., 3:4] * torch.cos(rot) - human_states_tensor[..., 2:3] * torch.sin(rot)
        px1 = (human_states_tensor[..., 0:1] - robot_state_tensor[..., 0:1]) * torch.cos(rot) \
              + (human_states_tensor[..., 1:2] - robot_state_tensor[..., 1:2]) * torch.sin(rot)
        py1 = (human_states_tensor[..., 1:2] - robot_state_tensor[..., 1:2]) * torch.cos(rot) \
              - (human_states_tensor[..., 0:1] - robot_state_tensor[..., 0:1]) * torch.sin(rot)

        new_robot_state_tensor = torch.cat([dg, v_pref, theta, radius, vx, vy], dim=-1)
        new_human_states_tensor = torch.cat([px1, py1, vx1, vy1, human_states_tensor[..., -1:]], dim=-1)
        return new_robot_state_tensor, new_human_states_tensor