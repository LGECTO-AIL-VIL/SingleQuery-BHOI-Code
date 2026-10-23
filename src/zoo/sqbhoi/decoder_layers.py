"""Decoder building blocks of SQ-BHOI: MLP, multi-scale deformable attention, decoder layer and
the OiH transformer decoder (body pose and hand-object reference points)."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid

__all__ = ["MLP", "MSDeformableAttention", "TransformerDecoderLayer", "OiHTransformerDecoder"]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile(
            [1, self.num_levels, self.num_points, 1]
        )
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.to(value.dtype).unsqueeze(-1)
            value = value * value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=reference_points.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            if len(reference_points.shape) == 4:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / self.num_points
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
                )
            else:
                sampling_locations = (
                    reference_points[:, :, :, :, None, :2]
                    + sampling_offsets
                    / self.num_points
                    * reference_points[:, :, :, :, None, 2:]
                    * 0.5
                )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.n_head = n_head

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        tgt,
        reference_points,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        if attn_mask is None:
            tgt2, _ = self.self_attn(q, k, value=tgt, average_attn_weights=False)
        else:
            tgt2, _ = self.self_attn(
                q, k, value=tgt, attn_mask=attn_mask, average_attn_weights=False
            )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class OiHTransformerDecoder(nn.Module):
    """TransformerDecoder extended with body pose and OiH reference points."""

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        num_layers,
        eval_idx=-1,
        n_head=32,
        kpts_n_head=17,
        oih_config=None,
    ):
        super(OiHTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.n_head = n_head
        self.kpts_n_head = kpts_n_head

        if oih_config is None:
            oih_config = {}
        for k, v in oih_config.items():
            setattr(self, k, v)
        self.oih_config = oih_config

        self.sigmoid = nn.Sigmoid()
        self.sigmoid_bbox_head = nn.ModuleList([nn.Sigmoid() for _ in range(num_layers)])
        self.sigmoid_bbox_head_2 = nn.ModuleList([nn.Sigmoid() for _ in range(num_layers - 1)])

        if oih_config.get("oih_sim", False):
            self.sigmoid_oih = nn.Sigmoid()
            self.sigmoid_oih_head = nn.ModuleList([nn.Sigmoid() for _ in range(num_layers)])

        # These will be set by the parent SQBHOIOiHTransformer
        self.bbox_head = None
        self.score_head = None
        self.query_pos_head = None
        self.vis_head = None
        self.kpts_hw = None
        self.pose_head = None
        self.oih_head = None
        self.obj_embed_head = None
        self.hand_embed_head = None
        self.oih_bias = None

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        body_points_unact=None,
        attn_mask=None,
        memory_mask=None,
        oih_points_unact=None,
        dn_meta=None,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_seg = []
        dec_out_poses = []
        dec_out_vis = []
        dec_out_oih = []

        ref_points_detach = self.sigmoid(ref_points_unact)
        bs, nq = ref_points_detach.shape[:2]

        if self.pose_head is not None:
            body_points_detach = body_points_unact.clone()
            _, _, nk = body_points_detach.shape

        if self.oih_head is not None:
            oih_dim = self.oih_config.get("oih_dim", 8)
            if self.oih_config.get("oih_vis", False):
                oih_points_detach = oih_points_unact[:, :nq].reshape(bs, nq, 2, (oih_dim - 2) // 2)
            else:
                oih_points_detach = oih_points_unact[:, :nq].reshape(bs, nq, 2, oih_dim // 2)
            oih_points_detach = self.sigmoid_oih(oih_points_detach)

        # previous layer's refined predictions (set at the end of each layer)
        ref_points = body_points = oih_points = None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if self.pose_head is not None:
                body_xy_ref_points = body_points_detach.reshape(bs, nq, nk // 2, 2)
                body_hw_ref_points = ref_points_input.detach()[
                    ..., 2:
                ] * self.kpts_hw.weight.unsqueeze(0).unsqueeze(1).repeat(bs, nq, 1, 1)
                body_ref_points_input = torch.cat([body_xy_ref_points, body_hw_ref_points], dim=-1)

                if self.oih_head is not None:
                    n_head = self.n_head - self.kpts_n_head - 4
                else:
                    n_head = self.n_head - self.kpts_n_head

                ref_points_input = torch.cat(
                    [
                        ref_points_input.repeat(1, 1, n_head, 1),
                        body_ref_points_input,
                    ],
                    dim=2,
                )

                if self.oih_head is not None:
                    oih_points_input = oih_points_detach.repeat(1, 1, 2, 1)
                    ref_points_input = torch.cat([ref_points_input, oih_points_input], dim=2)

                ref_points_input = ref_points_input.unsqueeze(3)
            else:
                ref_points_input = ref_points_input.unsqueeze(3)

            query_pos_embed = self.query_pos_head(ref_points_detach)

            if attn_mask is None:
                output = layer(
                    output,
                    ref_points_input,
                    memory,
                    memory_spatial_shapes,
                    memory_level_start_index,
                    query_pos_embed=query_pos_embed,
                )
            else:
                output = layer(
                    output,
                    ref_points_input,
                    memory,
                    memory_spatial_shapes,
                    memory_level_start_index,
                    attn_mask=attn_mask,
                    query_pos_embed=query_pos_embed,
                )

            dec_out_seg.append(output)

            if self.pose_head is not None:
                inter_ref_pose = self.pose_head[i](output) + body_points_detach

            inter_ref_bbox = self.sigmoid_bbox_head[i](
                self.bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )

            if self.oih_head is not None:
                inter_ref_oih = self.oih_head[i](output)
                inter_ref_oih = inter_ref_oih.reshape(bs, nq, 2, oih_dim // 2)
                if self.oih_config.get("oih_vis", False):
                    oih_vis_out = inter_ref_oih[..., -1:]
                    inter_ref_oih = inter_ref_oih[..., :-1]
                inter_ref_oih = inter_ref_oih + inverse_sigmoid(oih_points_detach)
                inter_ref_oih = self.sigmoid_oih_head[i](inter_ref_oih)

                # Calculate similarity map
                oih_obj_embed = self.obj_embed_head[i](output)
                oih_hand_embed = self.hand_embed_head[i](output)
                oih_hand_embed = oih_hand_embed.reshape(bs, nq, 2, -1)
                # Only use object + hands-off query
                if dn_meta is not None:
                    num_dn = dn_meta["dn_num_split"][0]
                    num_q = dn_meta["dn_num_split"][1]
                    oih_obj_embed = oih_obj_embed[:, num_dn : num_dn + num_q + 1]
                oih_embed = torch.einsum("bqhd,bnd->bqhn", oih_hand_embed, oih_obj_embed)
                oih_embed = oih_embed / math.sqrt(oih_hand_embed.shape[-1])
                oih_embed = oih_embed + self.oih_bias[i].to(oih_embed.device)

            if self.training:
                dec_out_logits.append(self.score_head[i](output))
                if self.vis_head is not None:
                    dec_out_vis.append(self.vis_head[i](output))
                if self.oih_head is not None:
                    if i == 0:
                        if self.oih_config.get("oih_vis", False):
                            dec_out_oih.append(
                                torch.cat([oih_embed, inter_ref_oih, oih_vis_out], dim=-1)
                            )
                        else:
                            dec_out_oih.append(torch.cat([oih_embed, inter_ref_oih], dim=-1))
                    else:
                        oih_out = self.oih_head[i](output)
                        oih_out = oih_out.reshape(bs, nq, 2, oih_dim // 2)
                        if self.oih_config.get("oih_vis", False):
                            oih_vis_out = oih_out[..., -1:]
                            oih_out = oih_out[..., :-1]
                            oih_out = oih_out + inverse_sigmoid(oih_points)
                            oih_out = self.sigmoid_oih_head[i - 1](oih_out)  # use i-1 for train aux
                            dec_out_oih.append(torch.cat([oih_embed, oih_out, oih_vis_out], dim=-1))
                        else:
                            oih_out = oih_out + inverse_sigmoid(oih_points)
                            oih_out = self.sigmoid_oih_head[i - 1](oih_out)
                            dec_out_oih.append(torch.cat([oih_embed, oih_out], dim=-1))

                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                    if self.pose_head is not None:
                        dec_out_poses.append(inter_ref_pose)
                else:
                    dec_out_bboxes.append(
                        self.sigmoid_bbox_head_2[i - 1](
                            self.bbox_head[i](output) + inverse_sigmoid(ref_points)
                        )
                    )
                    if self.pose_head is not None:
                        dec_out_poses.append(self.pose_head[i](output) + body_points)

            elif i == self.eval_idx:
                dec_out_logits.append(self.score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                if self.vis_head is not None:
                    dec_out_vis.append(self.vis_head[i](output))
                if self.oih_head is not None:
                    if self.oih_config.get("oih_vis", False):
                        dec_out_oih.append(
                            torch.cat([oih_embed, inter_ref_oih, oih_vis_out], dim=-1)
                        )
                    else:
                        dec_out_oih.append(torch.cat([oih_embed, inter_ref_oih], dim=-1))
                if self.pose_head is not None:
                    dec_out_poses.append(inter_ref_pose)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox
            if self.pose_head is not None:
                body_points = inter_ref_pose
                body_points_detach = inter_ref_pose.detach() if self.training else inter_ref_pose
            if self.oih_head is not None:
                oih_points = inter_ref_oih
                oih_points_detach = inter_ref_oih.detach() if self.training else inter_ref_oih

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_seg),
            torch.stack(dec_out_poses) if self.pose_head is not None else None,
            torch.stack(dec_out_vis) if self.vis_head is not None else None,
            torch.stack(dec_out_oih) if self.oih_head is not None else None,
        )
