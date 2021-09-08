from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


from .local_graph import LocalSubGraph_MHA, LocalSubGraph, SinusoidalPositionalEmbedding
from .global_graph import MultiheadAttentionGlobalHead, VectorizedEmbedding
from .common import pad_avail, pad_points, build_target_normalization


class VectorizedModel(nn.Module):
    """ Vectorized architecture with subgraph and global attention
    """

    def __init__(
        self,
        model_arch: str,
        subgraph_arch: str,
        history_num_frames_ego: int,
        history_num_frames_agents: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,  # criterion is only needed for training and not for evaluation
        disable_other_agents: bool,
        disable_map: bool,
        disable_lane_boundaries: bool,
        skip_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.disable_map = disable_map
        self.disable_other_agents = disable_other_agents
        self.disable_lane_boundaries = disable_lane_boundaries

        self._model_arch = model_arch
        self.subgraph_arch = subgraph_arch
        self._history_num_frames_ego = history_num_frames_ego
        self._history_num_frames_agents = history_num_frames_agents
        # change output size
        # X, Y  * number of future states
        self._num_targets = num_targets

        self._d_local = 256
        self._d_global = 256

        self._agent_features = ["start_x", "start_y", "yaw"]
        self._lane_features = ["start_x", "start_y", "tl_feature"]
        self._vector_agent_length = len(self._agent_features)
        self._vector_lane_length = len(self._lane_features)
        self._subgraph_layers = 3

        self.register_buffer("weights_scaling", torch.as_tensor(weights_scaling))
        self.criterion = criterion

        self.normalize_targets = True
        num_outputs = len(weights_scaling)
        num_timesteps = num_targets // num_outputs

        if self.normalize_targets:
            scale = build_target_normalization(num_timesteps)
            self.register_buffer("xy_scale", scale)

        # normalization buffers
        self.register_buffer("agent_std", torch.tensor([1.6919, 0.0365, 0.0218]))
        self.register_buffer("other_agent_std", torch.tensor([33.2631, 21.3976, 1.5490]))

        self.input_embed = nn.Linear(self._vector_agent_length, self._d_local)
        self.positional_embedding = SinusoidalPositionalEmbedding(self._d_local)
        self.type_embedding = VectorizedEmbedding(self._d_global)

        self.disable_pos_encode = False

        if self.subgraph_arch == "local_subgraph":
            self.local_subgraph = LocalSubGraph(num_layers=self._subgraph_layers, dim_in=self._d_local)
        elif self.subgraph_arch == "local_subgraph_mha":
            self.local_subgraph = LocalSubGraph_MHA(
                num_layers=self._subgraph_layers, dim_in=self._d_local, disable_pos_encode=self.disable_pos_encode,
            )
        else:
            raise ValueError(f"Subgraph arch {subgraph_arch} unknown in agent_prediction/model.py")

        if self._d_global != self._d_local:
            self.global_from_local = nn.Linear(self._d_local, self._d_global)

        dropout = 0.1
        if model_arch == "vectorized_mha":
            self.global_head = MultiheadAttentionGlobalHead(
                self._d_global, num_timesteps, num_outputs, dropout=dropout
            )
        else:
            raise ValueError(f"Model arch {model_arch} unknown in agent_prediction/model.py")

    def embed_polyline(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        polys = self.input_embed(features)
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)
        # batch_size x total num polys x num vecs
        invalid_mask = ~mask
        invalid_polys = invalid_mask.all(-1)
        # batch_size x total num polys x num vecs
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
        """
        agents_polys_feats = torch.cat(
            [agents_polys[:, :1] / self.agent_std, agents_polys[:, 1:] / self.other_agent_std], dim=1
        )
        static_polys_feats = static_polys / self.other_agent_std

        all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1)
        all_avail = torch.cat([agents_avail, static_avail], dim=1)

        # Input embedding, positional embedding, and local subgraph
        all_embs, invalid_polys = self.embed_polyline(all_polys, all_avail)
        if hasattr(self, "global_from_local"):
            all_embs = self.global_from_local(all_embs)

        # transformer
        all_embs = F.normalize(all_embs, dim=-1) * (self._d_global ** 0.5)
        all_embs = all_embs.transpose(0, 1)

        other_agents_len = agents_polys.shape[1] - 1
        if self.disable_other_agents:
            invalid_polys[:, 1 : (1 + other_agents_len)] = 1  # agents won't create attention

        if self.disable_map:  # lanes (mid), crosswalks, and lanes boundaries.
            invalid_polys[:, (1 + other_agents_len) :] = 1  # lanes won't create attention

        if self.disable_lane_boundaries:
            type_embedding = type_embedding[:-lane_bdry_len]

        invalid_polys[:, 0] = 0  # make AoI always available in global graph

        return self.global_head(all_embs, type_embedding, invalid_polys)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        type_embedding = self.type_embedding(data_batch).transpose(0, 1)
        lane_bdry_len = data_batch["lanes"].shape[1]

        outputs, attns = self.model_call(
            agents_polys, map_polys, agents_availabilities, map_availabilities, type_embedding, lane_bdry_len
        )

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            xy = data_batch["target_positions"]
            yaw = data_batch["target_yaws"]
            if self.normalize_targets:
                xy /= self.xy_scale
            targets = torch.cat((xy, yaw), dim=-1)
            target_weights = data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            pred_positions, pred_yaws = outputs[..., :2], outputs[..., 2:3]
            if self.normalize_targets:
                pred_positions *= self.xy_scale

            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            if attns is not None:
                eval_dict["attention_weights"] = attns
            return eval_dict