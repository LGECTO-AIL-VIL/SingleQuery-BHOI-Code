"""
SQ-BHOI (Object in Hands) transformer decoder head: queries, reference points, denoising and
body pose / OiH heads on top of the layers in `decoder_layers.py`.
"""

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.core import register

from .decoder_layers import MLP, OiHTransformerDecoder, TransformerDecoderLayer
from .denoising import get_contrastive_denoising_training_group
from .utils import bias_init_with_prob

__all__ = ["SQBHOIOiHTransformer"]

REPO_ROOT = Path(__file__).resolve().parents[3]


@register
class SQBHOIOiHTransformer(nn.Module):
    __share__ = [
        "num_classes",
        "task",
        "num_kpts",
        "oih_config",
        "pose_relative",
    ]

    def __init__(
        self,
        num_classes=81,
        hidden_dim=256,
        num_queries=300,
        position_embed_type="sine",
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        task=["detection"],
        num_kpts=17,
        kpts_nhead=17,
        init_2dkpt_file=None,
        pose_relative=None,
        batchnorm=False,
        oih_config=None,
    ):
        super(SQBHOIOiHTransformer, self).__init__()
        assert position_embed_type in ["sine", "learned"]
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.kpts_nhead = kpts_nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.task = task if task is not None else ["detection"]
        self.num_kpts = num_kpts
        self.pose_relative = pose_relative
        self.batchnorm = batchnorm

        if oih_config is None:
            oih_config = {"oih_sim": True, "oih_sim_dim": 256, "oih_vis": False}
        for k, v in oih_config.items():
            setattr(self, k, v)
        self.oih_config = oih_config

        if "oih" in self.task:
            assert self.oih_sim, (
                "Only the similarity-based OiH head is supported; "
                "set `oih_sim: True` in oih_config."
            )
            assert "body" in self.task, (
                "OiH requires the 'body' task (hand anchors use wrist keypoints)."
            )
            self.oih_dim = 8
            if oih_config.get("oih_vis", False):
                self.oih_dim += 2
            self.oih_config["oih_dim"] = self.oih_dim

        if "body" in self.task:
            assert nhead > kpts_nhead, (
                f"Num of total heads {nhead} > num of kpts heads {kpts_nhead}"
            )

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_decoder_points,
        )
        self.decoder = OiHTransformerDecoder(
            hidden_dim,
            decoder_layer,
            num_decoder_layers,
            eval_idx,
            n_head=nhead,
            kpts_n_head=kpts_nhead,
            oih_config=self.oih_config,
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )

        # encoder head
        if self.batchnorm:
            self.enc_output = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)),
                        ("norm", nn.BatchNorm1d(hidden_dim)),
                    ]
                )
            )
        else:
            self.enc_output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        self.pose_represent = None

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)]
        )

        # Body heads
        if "body" in self.task:
            self.dim_vis = num_kpts
            self.enc_vis_head = nn.Linear(hidden_dim, self.dim_vis)
            self.dec_vis_head = nn.ModuleList(
                [nn.Linear(hidden_dim, self.dim_vis) for _ in range(num_decoder_layers)]
            )
            if init_2dkpt_file is not None and init_2dkpt_file != "Learn":
                kpt_path = Path(init_2dkpt_file)
                if not kpt_path.is_absolute():
                    kpt_path = REPO_ROOT / kpt_path
                pose_represent = torch.Tensor(np.load(kpt_path))
                if self.pose_relative == "box":
                    pose_min = pose_represent.min(0)[0]
                    pose_max = pose_represent.max(0)[0]
                    pose_center = (pose_min + pose_max) / 2
                    pose_size = 1.2 * (pose_max - pose_min)
                    pose_represent = (pose_represent - pose_center) / pose_size
                self.pose_represent = pose_represent.reshape(1, 1, -1)
                assert self.pose_represent.shape[-1] == (self.num_kpts * 2)

            self.enc_pose_head = MLP(hidden_dim, hidden_dim, num_kpts * 2, num_layers=3)
            self.kpts_hw = nn.Embedding(kpts_nhead, 2)
            self.dec_pose_head = nn.ModuleList(
                [
                    MLP(hidden_dim, hidden_dim, num_kpts * 2, num_layers=3)
                    for _ in range(num_decoder_layers)
                ]
            )
        else:
            self.dec_pose_head = None
            self.kpts_hw = None
            self.dec_vis_head = None

        # OiH heads
        self.enc_oih_head = None
        self.dec_oih_head = None
        self.oih_hw = None
        self.enc_obj_embed_head = None
        self.dec_obj_embed_head = None
        self.enc_hand_embed_head = None
        self.dec_hand_embed_head = None
        self.enc_oih_bias = None
        self.dec_oih_bias = None

        if "oih" in self.task:
            self.enc_oih_head = MLP(hidden_dim, hidden_dim, self.oih_dim, num_layers=3)
            self.dec_oih_head = nn.ModuleList(
                [
                    MLP(hidden_dim, hidden_dim, self.oih_dim, num_layers=3)
                    for _ in range(num_decoder_layers)
                ]
            )
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.enc_oih_bias = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
            self.dec_oih_bias = [
                nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
                for _ in range(num_decoder_layers)
            ]
            self.off_embed = nn.Embedding(1, hidden_dim)
            self.pnt_oih = nn.Embedding(1, 4)
            self.pnt_oih_hand = nn.Embedding(1, 4)
            if "body" in self.task:
                self.pnt_oih_body = nn.Embedding(1, self.num_kpts * 2)
            self.enc_obj_embed_head = MLP(
                hidden_dim,
                hidden_dim,
                self.oih_sim_dim,
                num_layers=3,
            )
            self.dec_obj_embed_head = nn.ModuleList(
                [
                    MLP(hidden_dim, hidden_dim, self.oih_sim_dim, num_layers=3)
                    for _ in range(num_decoder_layers)
                ]
            )
            self.enc_hand_embed_head = MLP(
                hidden_dim,
                hidden_dim,
                self.oih_sim_dim * 2,
                num_layers=3,
            )
            self.dec_hand_embed_head = nn.ModuleList(
                [
                    MLP(hidden_dim, hidden_dim, self.oih_sim_dim * 2, num_layers=3)
                    for _ in range(num_decoder_layers)
                ]
            )
            self.oih_hw = nn.Embedding(2, 2)
            self.sigmoid_oih = nn.Sigmoid()

        # Init anchors
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()
            if "body" in self.task:
                self.anchors_body, _ = self._generate_pose_anchors()

        self._reset_parameters()

        # Wire up decoder heads
        self.decoder.bbox_head = self.dec_bbox_head
        self.decoder.score_head = self.dec_score_head
        self.decoder.query_pos_head = self.query_pos_head
        self.decoder.vis_head = self.dec_vis_head
        self.decoder.kpts_hw = self.kpts_hw
        self.decoder.pose_head = self.dec_pose_head
        self.decoder.oih_head = self.dec_oih_head
        self.decoder.obj_embed_head = self.dec_obj_embed_head
        self.decoder.hand_embed_head = self.dec_hand_embed_head
        self.decoder.oih_bias = self.dec_oih_bias

        # State dict hook: also load checkpoints that store the heads as `decoder.dec_*_head`
        def modify_decoder_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            state_dict_name = list(state_dict.keys())
            for state_name in state_dict_name:
                if state_name.startswith("decoder.dec_bbox_head"):
                    copy_state_name = state_name.replace("dec_bbox_head", "bbox_head")
                    copy_state_name = "decoder." + copy_state_name
                    if copy_state_name not in state_dict_name:
                        state_dict[copy_state_name] = state_dict[state_name]
                elif state_name.startswith("decoder.dec_score_head"):
                    copy_state_name = state_name.replace("dec_score_head", "score_head")
                    copy_state_name = "decoder." + copy_state_name
                    if copy_state_name not in state_dict_name:
                        state_dict[copy_state_name] = state_dict[state_name]
                elif state_name.startswith("decoder.query_pos_head"):
                    copy_state_name = "decoder." + state_name
                    if copy_state_name not in state_dict_name:
                        state_dict[copy_state_name] = state_dict[state_name]
                elif state_name.startswith("decoder.dec_vis_head"):
                    copy_state_name = state_name.replace("dec_vis_head", "vis_head")
                    copy_state_name = "decoder." + copy_state_name
                    if copy_state_name not in state_dict_name:
                        state_dict[copy_state_name] = state_dict[state_name]
                elif state_name.startswith("decoder.kpts_hw"):
                    copy_state_name = "decoder." + state_name
                    if copy_state_name not in state_dict_name:
                        state_dict[copy_state_name] = state_dict[state_name]
                elif state_name.startswith("decoder.dec_pose_head"):
                    copy_state_name = state_name.replace("dec_pose_head", "pose_head")
                    copy_state_name = "decoder." + copy_state_name
                    if copy_state_name not in state_dict_name:
                        state_dict[copy_state_name] = state_dict[state_name]

        self._register_load_state_dict_pre_hook(modify_decoder_state_dict)

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_ in self.dec_score_head:
            init.constant_(cls_.bias, bias)
        for reg_ in self.dec_bbox_head:
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        if not self.batchnorm:
            init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

        if "body" in self.task:
            init.constant_(self.enc_vis_head.weight, 0)
            init.constant_(self.enc_vis_head.bias, 0)
            init.constant_(self.enc_pose_head.layers[-1].weight, 0)
            init.constant_(self.enc_pose_head.layers[-1].bias, 0)
            if "oih" in self.task:
                init.constant_(self.enc_oih_head.layers[-1].weight, 0)
                init.constant_(self.enc_oih_head.layers[-1].bias, 0)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    in_channels, self.hidden_dim, 3, 2, padding=1, bias=False
                                ),
                            ),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])

        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.max(anchors))
        return anchors, valid_mask

    def _generate_pose_anchors(self, spatial_shapes=None, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        num_kpts = self.num_kpts
        for h, w in spatial_shapes:
            if self.pose_relative == "box":
                anchor_re = torch.zeros(1, h * w, 2 * num_kpts)
            else:
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(end=h, dtype=dtype),
                    torch.arange(end=w, dtype=dtype),
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], -1)
                valid_WH = torch.tensor([w, h]).to(dtype)
                grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
                anchor = grid_xy.reshape(-1, h * w, 2)
                if self.pose_relative == "root":
                    anchor_re = torch.zeros(
                        (anchor.shape[0], anchor.shape[1], anchor.shape[2] * num_kpts)
                    )
                    anchor_re[..., :2] = anchor
                else:
                    anchor_re = anchor.repeat(1, 1, num_kpts)
            anchors.append(anchor_re)

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)

        if self.pose_represent is not None:
            pose_offset = self.pose_represent.repeat(1, anchors.size(1), 1).to(device)
            anchors += pose_offset
        else:
            anchors = torch.zeros(anchors.shape[0], anchors.shape[1], num_kpts * 2, dtype=dtype)
        return anchors, valid_mask

    def _get_decoder_input(
        self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None
    ):
        bs = memory.shape[0]

        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype) * memory

        if self.batchnorm:
            output_memory = self.enc_output(memory.transpose(1, 2).contiguous())
            output_memory = output_memory.transpose(1, 2).contiguous()
        else:
            output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        target = output_memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        undetached_target = target
        target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]),
        )

        topk_anchors = anchors.repeat((bs, 1, 1)).gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]),
        )
        reference_points_unact = self.enc_bbox_head(undetached_target) + topk_anchors

        # Body pose processing
        oih_points_unact_detached = None
        enc_topk_oih = None

        if "body" in self.task:
            if self.training or self.eval_spatial_size is None:
                anchors_body, _ = self._generate_pose_anchors(spatial_shapes, device=memory.device)
            else:
                anchors_body = self.anchors_body
            anchors_body = anchors_body.to(memory.device)

            topk_anchors_body = anchors_body.repeat((bs, 1, 1)).gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, anchors_body.shape[-1]),
            )

            if "oih" in self.task:
                enc_topk_oih = self.enc_oih_head(undetached_target)
                off_embed = self.off_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
                enc_topk_obj = self.enc_obj_embed_head(
                    torch.cat([undetached_target, off_embed], dim=1)
                )
                enc_topk_hand = self.enc_hand_embed_head(undetached_target).reshape(
                    bs, self.num_queries, 2, self.oih_sim_dim
                )
                enc_topk_embed = torch.einsum("bqhd,bnd->bqhn", enc_topk_hand, enc_topk_obj)
                enc_topk_embed = enc_topk_embed / math.sqrt(enc_topk_hand.shape[-1])
                enc_topk_embed = enc_topk_embed + self.enc_oih_bias.to(enc_topk_embed.device)

                # Use body anchor for hand positions (wrist keypoints)
                topk_anchors_oih = torch.cat(
                    [topk_anchors_body[..., 20:22], topk_anchors_body[..., 18:20]], dim=-1
                ).reshape(bs, self.num_queries, 2, 2)

                oih_hw_points_unact = reference_points_unact.unsqueeze(-2)[
                    ..., 2:
                ] * self.oih_hw.weight.unsqueeze(0).unsqueeze(1).repeat(bs, self.num_queries, 1, 1)
                topk_anchors_oih = torch.cat([topk_anchors_oih, oih_hw_points_unact], dim=-1)

                enc_topk_oih = enc_topk_oih.reshape(bs, self.num_queries, 2, -1) + topk_anchors_oih
                oih_points_unact = enc_topk_oih.reshape(bs, self.num_queries, -1).clone()
                enc_topk_oih = torch.cat([enc_topk_embed, self.sigmoid_oih(enc_topk_oih)], dim=-1)

                # Add off query pnt
                pnt_oih_hand = (
                    self.pnt_oih_hand.weight.unsqueeze(0)
                    .unsqueeze(2)
                    .repeat(bs, 1, 2, 1)
                    .reshape(bs, 1, -1)
                )
                oih_points_unact = torch.cat([oih_points_unact, pnt_oih_hand], dim=1)

                if denoising_bbox_unact is not None:
                    B, Q, _ = denoising_bbox_unact.size()
                    oih_points_unact = torch.cat(
                        [
                            torch.zeros(B, Q, self.oih_dim).to(oih_points_unact),
                            oih_points_unact.clone(),
                        ],
                        1,
                    )
                oih_points_unact_detached = oih_points_unact.detach()

                target = torch.cat([target, off_embed], dim=1)

            body_points_unact = self.enc_pose_head(undetached_target) + topk_anchors_body
            enc_topk_poses = body_points_unact
            enc_topk_vis = self.enc_vis_head(undetached_target)

            if denoising_bbox_unact is not None:
                B, Q, _ = denoising_bbox_unact.size()
                body_points_unact = torch.cat(
                    [
                        torch.zeros(B, Q, self.num_kpts * 2).to(body_points_unact),
                        body_points_unact,
                    ],
                    1,
                )
            body_points_unact_detached = body_points_unact.detach()

            if "oih" in self.task:
                body_points_unact_detached = torch.concat(
                    [
                        body_points_unact_detached,
                        self.pnt_oih_body.weight.unsqueeze(0).repeat(bs, 1, 1),
                    ],
                    1,
                )
        else:
            enc_topk_poses = None
            body_points_unact_detached = None
            enc_topk_vis = None
            enc_topk_oih = None
            oih_points_unact_detached = None

        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        reference_points_unact_detached = reference_points_unact.detach()
        if "oih" in self.task:
            reference_points_unact_detached = torch.concat(
                [
                    reference_points_unact_detached,
                    self.pnt_oih.weight.unsqueeze(0).repeat(bs, 1, 1),
                ],
                1,
            )

        return (
            target,
            reference_points_unact_detached,
            enc_topk_bboxes,
            enc_topk_logits,
            undetached_target,
            enc_topk_poses,
            enc_topk_vis,
            body_points_unact_detached,
            enc_topk_oih,
            oih_points_unact_detached,
        )

    def forward(self, feats, targets=None):
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # Prepare denoising
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    oih_sim=self.oih_sim,
                )
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        (
            target,
            init_ref_points_unact,
            enc_topk_bboxes,
            enc_topk_logits,
            undetached_target,
            enc_topk_poses,
            enc_topk_vis,
            body_points_unact,
            enc_topk_oih,
            oih_points_unact,
        ) = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_class,
            denoising_bbox_unact,
        )

        # Decoder
        spatial_shapes_t = torch.tensor(spatial_shapes)
        level_start_index_t = torch.tensor(level_start_index)
        decoder_output = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes_t,
            level_start_index_t,
            body_points_unact=body_points_unact,
            attn_mask=attn_mask,
            oih_points_unact=oih_points_unact,
            dn_meta=dn_meta,
        )

        out_bboxes, out_logits, out_segs, out_poses, out_vis, out_oih = decoder_output

        if "oih" in self.task:
            out_bboxes = out_bboxes[:, :, :-1, :]
            out_logits = out_logits[:, :, :-1, :]
            if out_poses is not None:
                out_poses = out_poses[:, :, :-1, :]
                out_vis = out_vis[:, :, :-1, :]
            if out_oih is not None:
                out_oih = out_oih[:, :, :-1, :]

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_poses = [None for _ in range(len(dn_out_bboxes))]
            dn_out_vis = [None for _ in range(len(dn_out_bboxes))]
            dn_out_oih = [None for _ in range(len(out_bboxes))]
            if "body" in self.task and out_poses is not None:
                dn_out_poses, out_poses = torch.split(out_poses, dn_meta["dn_num_split"], dim=2)
                dn_out_vis, out_vis = torch.split(out_vis, dn_meta["dn_num_split"], dim=2)
            if "oih" in self.task and out_oih is not None:
                dn_out_oih, out_oih = torch.split(out_oih, dn_meta["dn_num_split"], dim=2)

        if "body" not in self.task:
            out_poses = [None for _ in range(len(out_bboxes))]
            out_vis = [None for _ in range(len(out_bboxes))]
        if "oih" not in self.task:
            out_oih = [None for _ in range(len(out_bboxes))]

        out = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
            "pred_poses": out_poses[-1]
            if isinstance(out_poses, torch.Tensor)
            else out_poses[-1]
            if out_poses
            else None,
            "pred_vis": out_vis[-1]
            if isinstance(out_vis, torch.Tensor)
            else out_vis[-1]
            if out_vis
            else None,
            "pred_oih": out_oih[-1]
            if isinstance(out_oih, torch.Tensor)
            else out_oih[-1]
            if out_oih
            else None,
        }

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                out_logits[:-1],
                out_bboxes[:-1],
                out_poses[:-1],
                out_vis[:-1],
                out_oih[:-1],
            )
            out["aux_outputs"].extend(
                self._set_aux_loss(
                    [enc_topk_logits],
                    [enc_topk_bboxes],
                    [enc_topk_poses],
                    [enc_topk_vis],
                    [enc_topk_oih],
                )
            )

            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_poses,
                    dn_out_vis,
                    dn_out_oih,
                )
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_bboxes, outputs_poses, outputs_vis, outputs_oih):
        return [
            {
                "pred_logits": c,
                "pred_boxes": b,
                "pred_poses": p,
                "pred_vis": v,
                "pred_oih": o,
            }
            for c, b, p, v, o in zip(
                outputs_class,
                outputs_bboxes,
                outputs_poses,
                outputs_vis,
                outputs_oih,
            )
        ]
