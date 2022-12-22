import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> nn.Module:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _init_transformer_encoder_params(module: nn.Module) -> None:
    """Initialize weights specific to the transformer encoder.

    Linear and embedding layers are initialized by normal distributions. And the MultiheadAttention is initialized
    by its own initialization method.

    This implementation is inspired by the code piece here
    https://github.com/pytorch/fairseq/blob/c2e8904b6072d8eddab362ac50b324e374b5951d/fairseq/modules/transformer_sentence_encoder.py#L21
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight.data, mean=0., std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.)
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean=0., std=0.02)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    if isinstance(module, nn.MultiheadAttention):
        if module._qkv_same_embed_dim:
            nn.init.xavier_uniform_(module.in_proj_weight)
        else:
            nn.init.xavier_uniform_(module.q_proj_weight)
            nn.init.xavier_uniform_(module.k_proj_weight)
            nn.init.xavier_uniform_(module.v_proj_weight)

        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0.)
            nn.init.constant_(module.out_proj.bias, 0.)
        if module.bias_k is not None:
            nn.init.xavier_normal_(module.bias_k)
        if module.bias_v is not None:
            nn.init.xavier_normal_(module.bias_v)


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers. See PyTorch docs for more details."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def reset_parameters(self) -> None:
        self.apply(_init_transformer_encoder_params)

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None) -> Tensor:
        """Computes the forward of the module.

        :param src: module input, shape [num_elements, batch_size, num_features]
        :param mask: attention mask, shape [(batch_size,) num_elements, num_elements]
        :param src_key_padding_mask: key padding mask, shape [batch_size, num_elements]
        :param pos: positional embedding tensor, it will be added to src. shape [num_elements, batch_size, num_features]
        :return: tuple containing the module output and a list of attention weights (one for each encoder layer)
                 shape [num_elements, batch_size, num_features], List[[batch_size, num_elements, num_elements]]
        """
        output = src

        for layer in self.layers:
            output, _ = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """A stack of N transformer decoder layers. See PyTorch docs for more details."""

    def __init__(
            self,
            decoder_layer: nn.Module,
            num_layers: int,
            norm: Optional[nn.Module] = None,
            return_intermediate: Optional[bool] = False) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def reset_parameters(self) -> None:
        """See https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L78"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None) -> Tensor:
        """Computes the forward of the module.

        :param tgt: decoder layer input, used as input and queries
        :param memory: output of the last encoder layer, used as keys and values
        :param tgt_mask: attention mask for tgt
        :param memory_mask: attention mask for memory
        :param tgt_key_padding_mask: key padding mask for tgt
        :param memory_key_padding_mask: key padding mask for memory
        :param pos: positional embedding tensor, it will be added to memory
        :param query_pos: query positional embedding tensor, it will be added to tgt
        :return: the module output
        """
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                assert self.norm is not None
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer composed of self-attn and feedforward network. See PyTorch docs for more details."""

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Optional[str] = "relu",
            normalize_before: Optional[bool] = False) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        assert activation is not None
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src: Tensor,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward normalizing after the self-attention module"""
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

    def forward_pre(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward normalizing before the self-attention module"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn_weights

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the forward of the module.

        :param src: input of shape [num_elements, batch_size, num_features]
        :param src_mask: attention mask [batch_size, num_elements, num_elements]
        :param src_key_padding_mask: key padding mask [batch_size, num_elements]
        :param pos: positional embedding tensor, it will be added to src
        :return: tuple containing the layer output and the computed attention weights
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer composed of self-attn, multi-head-attn and feedforward network.
    See PyTorch docs for more details."""

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: Optional[int] = 2048,
            dropout: float = 0.1,
            activation: Optional[str] = "relu",
            normalize_before: Optional[bool] = False) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        assert activation is not None
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None) -> Tensor:
        """Forward normalizing after the self-attention module"""
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None) -> Tensor:
        """Forward normalizing before the self-attention module"""
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None) -> Tensor:
        """Computes the forward of the module.

        :param tgt: decoder layer input, used as input and queries
        :param memory: output of the last encoder layer, used as keys and values
        :param tgt_mask: attention mask for tgt
        :param memory_mask: attention mask for memory
        :param tgt_key_padding_mask: key padding mask for tgt
        :param memory_key_padding_mask: key padding mask for memory
        :param pos: positional embedding tensor, it will be added to memory
        :param query_pos: query positional embedding tensor, it will be added to tgt
        :return: the layer output
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
