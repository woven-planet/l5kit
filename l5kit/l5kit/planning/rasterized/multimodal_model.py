from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RasterizedPlanningModel


class RasterizedMultiModalPlanningModel(RasterizedPlanningModel):
    """Raster-based multimodal planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        future_num_frames: int,
        num_outputs: int,
        num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        coef_alpha: float = 0.5,
    ) -> None:
        """Initializes the multimodal planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param future_num_frames: number of future frames to predict
        :param num_outputs: number of output dimensions, by default is 3: x, y, heading
        :param num_modes: number of modes in predicted outputs
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        :param coef_alpha: hyper-parameter used to trade-off between trajectory distance loss and classification
        cross-entropy loss
        """
        num_targets = (future_num_frames * num_outputs + 1) * num_modes
        super().__init__(
            model_arch,
            num_input_channels,
            num_targets,
            weights_scaling,
            criterion,
            pretrained,
        )
        self.num_modes = num_modes
        self.num_timestamps = future_num_frames
        self.num_outputs = num_outputs
        self.coef_alpha = coef_alpha

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        batch_size = len(image_batch)

        # [batch_size, (num_timestamps * num_outputs * num_modes) + num_modes]
        outputs_all = self.model(image_batch)

        # [batch_size, num_timestamps * num_outputs * num_modes]
        outputs = outputs_all[:, :-self.num_modes]
        # [batch_size, num_modes]
        outputs_nll = outputs_all[:, -self.num_modes:]

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, num_timestamps * num_modes, num_outputs]
            outputs = outputs.view(
                batch_size, self.num_timestamps * self.num_modes, self.num_outputs
            )

            # [batch_size, num_timestamps, 2]
            xy = data_batch["target_positions"]
            # [batch_size, num_timestamps, 1]
            yaw = data_batch["target_yaws"]

            # [batch_size, num_timestamps, num_outputs]
            targets = torch.cat((xy, yaw), dim=-1)
            # [batch_size, num_timestamps]
            target_weights = data_batch["target_availabilities"] > 0.5
            # [batch_size, num_timestamps, num_outputs]
            target_weights = target_weights.unsqueeze(-1) * self.weights_scaling
            # [batch_size, num_timestamps * num_modes, num_outputs]
            losses = self.criterion(
                outputs, targets.repeat(1, self.num_modes, 1)
            ) * target_weights.repeat(1, self.num_modes, 1)
            # [batch_size, num_modes]
            cost_dist = losses.view(batch_size, self.num_modes, -1).mean(dim=-1)
            # [batch_size,]
            assignment = cost_dist.argmin(dim=-1)
            # [1]
            loss_dist = cost_dist[torch.arange(batch_size, device=outputs.device), assignment].mean()
            # [1]
            loss_nll = F.cross_entropy(outputs_nll, assignment)
            return {"loss": loss_dist + self.coef_alpha * loss_nll}
        else:
            outputs = outputs.view(batch_size, self.num_modes, self.num_timestamps, self.num_outputs)
            outputs_selected = outputs[torch.arange(batch_size, device=outputs.device), outputs_nll.argmax(-1)]
            pred_positions, pred_yaws = outputs_selected[..., :2], outputs_selected[..., 2:3]
            pred_pos_all = outputs[..., :2]
            pred_yaw_all = outputs[..., 2:3]
            return {
                "positions": pred_positions,
                "yaws": pred_yaws,
                "positions_all": pred_pos_all,
                "yaws_all": pred_yaw_all,
            }
