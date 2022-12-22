from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from l5kit.planning.vectorized.common import (build_matrix, build_target_normalization, pad_avail, pad_points,
                                              transform_points)
from l5kit.planning.vectorized.global_graph import VectorizedEmbedding
from l5kit.planning.vectorized.local_graph import LocalSubGraph, SinusoidalPositionalEmbedding

from .safepathnet import MultimodalDecoder, TrajectoryMatcher


class SafePathNetModel(nn.Module):
    """ SafePathNet model - Unified prediction and planning model with multimodal output.
    """

    def __init__(
        self,
        history_num_frames_ego: int,
        history_num_frames_agents: int,
        num_timesteps: int,
        weights_scaling: List[float],
        criterion: nn.Module,  # criterion is only needed for training and not for evaluation
        disable_other_agents: bool,
        disable_map: bool,
        disable_lane_boundaries: bool,
        agent_num_trajectories: int,
        max_num_agents: int = 30,
        cost_prob_coeff: float = 0.01,
    ) -> None:
        """ Initializes the model.

        :param history_num_frames_ego: number of history ego frames to include
        :param history_num_frames_agents: number of history agent frames to include
        :param num_timesteps: number of predicted future steps
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param disable_other_agents: ignore agents
        :param disable_map: ignore map
        :param disable_lane_boundaries: ignore lane boundaries
        :param agent_num_trajectories: number of predicted trajectories per agent
        :param max_num_agents: maximum number of agents per scene
        :param cost_prob_coeff: weight of the probability cost in the TrajectoryMatcher and in the overall loss
        """
        super().__init__()
        self.disable_map = disable_map
        self.disable_other_agents = disable_other_agents
        self.disable_lane_boundaries = disable_lane_boundaries

        self._history_num_frames_ego = history_num_frames_ego
        self._history_num_frames_agents = history_num_frames_agents

        self.max_num_agents = max_num_agents

        # Model parameters
        self._d_local = 128
        self._d_global = 256
        self._d_feedforward = 1024
        self._d_num_layers = 3
        self._d_num_decode_layers = 3

        self._agent_features = ["start_x", "start_y", "yaw"]
        self._lane_features = ["start_x", "start_y", "tl_feature"]
        self._vector_agent_length = len(self._agent_features)
        self._vector_lane_length = len(self._lane_features)
        self._subgraph_layers = 3

        self.register_buffer("weights_scaling", torch.as_tensor(weights_scaling))
        self.criterion = criterion

        self.normalize_targets = False
        num_outputs = len(weights_scaling)
        self.num_timesteps = num_timesteps
        self._num_targets = len(weights_scaling)

        scale = build_target_normalization(num_timesteps)
        if not self.normalize_targets:
            scale.fill_(1.)
        self.register_buffer("xy_scale", scale)

        # normalization buffers
        self.register_buffer("agent_std", torch.tensor([1.6919, 0.0365, 0.0218]))
        self.register_buffer("other_agent_std", torch.tensor([33.2631, 21.3976, 1.5490]))

        self.input_embed = nn.Linear(self._vector_agent_length, self._d_local)
        self.positional_embedding = SinusoidalPositionalEmbedding(self._d_local)
        self.type_embedding = VectorizedEmbedding(self._d_global)

        self.disable_pos_encode = False

        self.local_subgraph = LocalSubGraph(num_layers=self._subgraph_layers, dim_in=self._d_local)

        self.global_head = MultimodalDecoder(
            dim_in=self._d_local, projection_dim=self._d_global, dim_feedforward=self._d_feedforward,
            num_layers=self._d_num_layers, num_decode_layers=self._d_num_decode_layers,
            num_outputs=num_outputs, future_num_frames=num_timesteps, agent_future_num_frames=num_timesteps,
            agent_num_trajectories=agent_num_trajectories,
            max_num_agents=self.max_num_agents
        )

        self.agent_traj_matcher = TrajectoryMatcher(cost_prob_coeff=cost_prob_coeff)
        self.eps = float(torch.finfo(torch.float).eps)

    def embed_polyline(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Embeds the inputs, generates the positional embedding and calls the local subgraph.

        :param features: input features
        :tensor features: [batch_size, num_elements, max_num_points, max_num_features]
        :param mask: availability mask
        :tensor mask: [batch_size, num_elements, max_num_points]

        :return tuple of local subgraph output, (in-)availability mask
        """
        # embed inputs
        # [batch_size, num_elements, max_num_points, embed_dim]
        polys = self.input_embed(features)
        # calculate positional embedding
        # [1, 1, max_num_points, embed_dim]
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)
        # [batch_size, num_elements, max_num_points]
        invalid_mask = ~mask
        invalid_polys = invalid_mask.all(-1)
        # input features to local subgraph and return result -
        # local subgraph reduces features over elements, i.e. creates one descriptor
        # per element
        # [batch_size, num_elements, embed_dim]
        polys = self.local_subgraph(polys, invalid_mask, pos_embedding)
        return polys, invalid_polys

    def model_call(
        self,
        agents_polys: torch.Tensor,
        static_polys: torch.Tensor,
        agents_avail: torch.Tensor,
        static_avail: torch.Tensor,
        type_embedding: torch.Tensor,
        lane_bdry_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Encapsulates calling the global_head and preparing needed data.

        :param agents_polys: dynamic elements - i.e. vectors corresponding to agents
        :param static_polys: static elements - i.e. vectors corresponding to map elements
        :param agents_avail: availability of agents
        :param static_avail: availability of map elements
        :param type_embedding: agent type embeddings
        :param lane_bdry_len: number of map elements
        """
        # Standardize inputs
        agents_polys_feats = torch.cat(
            [agents_polys[:, :1] / self.agent_std, agents_polys[:, 1:] / self.other_agent_std], dim=1
        )
        static_polys_feats = static_polys / self.other_agent_std

        all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1)
        all_avail = torch.cat([agents_avail, static_avail], dim=1)

        # Embed inputs, calculate positional embedding, call local subgraph
        all_embs, invalid_polys = self.embed_polyline(all_polys, all_avail)
        if hasattr(self, "global_from_local"):
            all_embs = self.global_from_local(all_embs)

        all_embs = F.normalize(all_embs, dim=-1) * (self._d_global ** 0.5)
        # all_embs = all_embs.transpose(0, 1)

        other_agents_len = agents_polys.shape[1] - 1

        # disable certain elements on demand
        if self.disable_other_agents:
            invalid_polys[:, 1: (1 + other_agents_len)] = 1  # agents won't create attention

        if self.disable_map:  # lanes (mid), crosswalks, and lanes boundaries.
            invalid_polys[:, (1 + other_agents_len):] = 1  # lanes won't create attention

        if self.disable_lane_boundaries:
            type_embedding = type_embedding[:-lane_bdry_len]

        invalid_polys[:, 0] = 0  # make AoI always available in global graph

        # call and return global graph
        outputs: Tuple[torch.Tensor, torch.Tensor] = self.global_head(all_embs, type_embedding, invalid_polys)
        return outputs

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Load and prepare vectors for the model call, split into map and agents

        # ==== LANES ====
        # batch size x num lanes x num vectors x num features
        polyline_keys = ["lanes_mid", "crosswalks"]
        if not self.disable_lane_boundaries:
            polyline_keys += ["lanes"]
        avail_keys = [f"{k}_availabilities" for k in polyline_keys]

        max_num_vectors = max([data_batch[key].shape[-2] for key in polyline_keys])

        map_polys = torch.cat([pad_points(data_batch[key], max_num_vectors) for key in polyline_keys], dim=1)
        map_polys[..., -1].fill_(0)
        # batch size x num lanes x num vectors
        map_availabilities = torch.cat([pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1)

        # ==== AGENTS ====
        # batch_size x (1 + M) x seq len x self._vector_length
        agents_polys = torch.cat(
            [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
        )
        # batch_size x (1 + M) x num vectors x self._vector_length
        agents_polys = pad_points(agents_polys, max_num_vectors)

        # batch_size x (1 + M) x seq len
        agents_availabilities = torch.cat(
            [
                data_batch["agent_polyline_availability"].unsqueeze(1),
                data_batch["other_agents_polyline_availability"],
            ],
            dim=1,
        )
        # batch_size x (1 + M) x num vectors
        agents_availabilities = pad_avail(agents_availabilities, max_num_vectors)

        # batch_size x (1 + M) x num features
        type_embedding = self.type_embedding(data_batch)  # .transpose(0, 1)
        lane_bdry_len = data_batch["lanes"].shape[1]

        # call the model with these features
        # [batch_size, num_agents, num_traj, num_timesteps, num_outputs], [batch_size, num_agents, num_traj]
        pred_agents, pred_traj_logits = self.model_call(
            agents_polys, map_polys, agents_availabilities, map_availabilities, type_embedding, lane_bdry_len
        )

        # used to make targets relative to agents / output relative to ego
        agents_t0_xy = data_batch["all_other_agents_history_positions"][:, :, 0]
        agents_t0_yaw = data_batch["all_other_agents_history_yaws"][:, :, 0].squeeze(-1)

        # calculate loss or return predicted position for inference
        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # ego
            # loss_ego = 0.  # we don't have a planning ego loss here

            # agents
            # gt is in ego-relative coords
            # [batch_size, num_agents, agent_num_timesteps, num_outputs]
            agents_xy = data_batch["all_other_agents_future_positions"]
            agents_yaw = data_batch["all_other_agents_future_yaws"]

            # make targets relative to agents
            # get the transformation matrix
            agents_t0_xy_exp = agents_t0_xy.reshape(-1, 2)
            agents_t0_yaw_exp = agents_t0_yaw.reshape(-1)
            _, inverse_matrix = build_matrix(translation=agents_t0_xy_exp, angle=agents_t0_yaw_exp)
            # get the correct matrix shape
            inverse_matrix = inverse_matrix.view(list(agents_t0_xy.shape[:2]) + [3, 3])
            inverse_matrix = inverse_matrix.unsqueeze(2).expand(
                list(agents_t0_xy.shape[:-1]) + [self.num_timesteps, 3, 3]).reshape(-1, 3, 3)
            # transform the points
            agents_points = torch.cat((agents_xy, agents_yaw), -1)
            agents_points_flattened = agents_points.reshape(-1, 1, 1, 3)
            transformed_points = transform_points(agents_points_flattened, inverse_matrix,
                                                  torch.ones_like(agents_points_flattened[..., 0], dtype=torch.bool),
                                                  agents_yaw.reshape(-1, 1, 1, 1))
            agents_xy = transformed_points[..., :2].reshape(agents_xy.shape)
            agents_yaw = transformed_points[..., 2:].reshape(agents_yaw.shape)

            # normalize targets
            if self.normalize_targets:
                agents_xy /= self.xy_scale
            targets = torch.cat((agents_xy, agents_yaw), dim=-1)
            targets[..., :2] /= 125.
            targets[..., 2] /= np.math.pi  # no need for complex angle normalization, we predict offsets

            # [batch_size, num_agents, num_timesteps]
            target_avails = data_batch["all_other_agents_future_availability"]

            # [batch_size, num_agents, agent_num_trajectories, num_timesteps, 3]
            targets = targets.unsqueeze(dim=2).expand(-1, -1, self.global_head.agent_num_trajectories, -1, -1)
            # [batch_size, num_agents, agent_num_trajectories, num_timesteps]
            target_avails = target_avails.unsqueeze(dim=2).expand(-1, -1, self.global_head.agent_num_trajectories, -1)

            # compute loss on trajectories
            agent_loss = self.criterion(pred_agents, targets)
            agent_loss *= target_avails.unsqueeze(-1)
            pred_num_valid_targets = target_avails.sum().float()

            pred_num_valid_targets /= self.global_head.agent_num_trajectories
            any_target_avails = torch.any(target_avails, dim=-1)

            # [batch_size, num_agents, agent_num_trajectories]
            loss_pred_per_trajectory = agent_loss.sum(-1).sum(-1) * any_target_avails

            # [batch_size, num_agents]
            pred_loss_argmin_idx = self.agent_traj_matcher(
                loss_pred_per_trajectory / (target_avails.sum(-1) + self.eps), pred_traj_logits)

            # compute loss on probability distribution for valid targets only
            pred_prob_loss = F.cross_entropy(pred_traj_logits[any_target_avails[..., 0]],
                                             pred_loss_argmin_idx[any_target_avails[..., 0]])

            # compute the final loss only on the trajectory with the lowest loss
            # [batch_size, num_agents, num_traj]
            pred_traj_loss_batch = agent_loss.sum(-1).sum(-1)
            # [batch_size, num_agents]
            # NOTE torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
            pred_traj_loss_batch = torch.gather(pred_traj_loss_batch, dim=-1,
                                                index=pred_loss_argmin_idx[:, :, None]).squeeze(-1)
            # zero out invalid targets (agents)
            pred_traj_loss_batch = pred_traj_loss_batch * any_target_avails[..., 0]
            # [1]
            pred_traj_loss = pred_traj_loss_batch.sum() / (pred_num_valid_targets + self.eps)
            # compute overall loss
            pred_loss = pred_traj_loss + pred_prob_loss * self.agent_traj_matcher.cost_prob_coeff

            train_dict = {
                "loss": pred_loss,
                "loss/agents_traj": pred_traj_loss,
                "loss/agents_traj_prob": pred_prob_loss,
            }
            return train_dict
        else:
            # ego
            # ego is ignored in these experiments (we don't include planning).
            # ego future is predicted considering ego as a road agent (ego is the first agent)

            # agents
            # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
            pred_agents[..., :2] *= 125.
            pred_agents[..., 2] *= np.math.pi  # no need for complex angle normalization, we predict offsets

            # getting the agents availabilities at the current timestep
            # [batch_size, num_agents]
            pred_agents_avails = data_batch['all_other_agents_history_availability'][:, :, 0]
            pred_agents[~pred_agents_avails] = 0.

            # make predictions relative to ego
            # get the transformation matrix
            agents_t0_xy_reshaped = agents_t0_xy.reshape(-1, 2)
            agents_t0_yaw_reshaped = agents_t0_yaw.reshape(-1)
            matrix, _ = build_matrix(translation=agents_t0_xy_reshaped, angle=agents_t0_yaw_reshaped)
            # get the correct matrix shape
            matrix = matrix.view(list(agents_t0_xy.shape[:2]) + [3, 3])
            matrix = matrix.unsqueeze(2).unsqueeze(2).expand(list(pred_agents.shape[:-1]) + [3, 3]).reshape(-1, 3, 3)
            # transform the points
            pred_agents_flattened = pred_agents.reshape(-1, 1, 1, 3)
            transformed_points = transform_points(pred_agents_flattened, matrix,
                                                  torch.ones_like(pred_agents_flattened[..., 0], dtype=torch.bool),
                                                  pred_agents_flattened[..., 2:])
            # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
            transformed_points = transformed_points.reshape(pred_agents.shape)
            all_pred_agents_xy = transformed_points[..., :2]
            all_pred_agents_yaw = transformed_points[..., 2:]

            # get the highest probability trajectory
            # [batch_size, num_agents]
            if self.global_head.agent_multimodal_predictions:
                # just pick the trajectory with highest probability
                pred_traj_index = pred_traj_logits.argmax(dim=-1)
            else:
                # pick the only predicted trajectory
                pred_traj_index = torch.zeros(pred_agents.shape[:2], dtype=torch.long, device=pred_agents.device)
            # [batch_size, num_agents, 1, 1, 1]
            pred_traj_index_expanded = pred_traj_index[:, :, None, None, None]
            # [batch_size, num_agents, num_future_frames, 3]
            # NOTE torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
            pred_agents_xy = torch.gather(
                all_pred_agents_xy, dim=2,
                index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents_xy.shape[-2:])))).squeeze(2)
            pred_agents_yaw = torch.gather(
                all_pred_agents_yaw, dim=2,
                index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents_yaw.shape[-2:])))).squeeze(2)

            eval_dict = {
                "agent_positions": pred_agents_xy,
                "agent_yaws": pred_agents_yaw,
                "all_agent_positions": all_pred_agents_xy,
                "all_agent_yaws": all_pred_agents_yaw,
                "agent_traj_logits": pred_traj_logits,
            }

            return eval_dict
