# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math

import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed

# `detr` is pip-installed as its own package, so my_utils may not be importable
# when it is used standalone. Degrade to no-op logging rather than blow up.
try:
    from my_utils.ori_logging import (get_logger, log_tensor, log_module_shapes,
                                      TRACE, StepGate)
except ImportError:  # pragma: no cover
    import logging as _logging
    TRACE = 5

    def get_logger(name):
        return _logging.getLogger("ori." + name)

    def log_tensor(*a, **k):
        pass

    def log_module_shapes(*a, **k):
        pass

    class StepGate:
        def __init__(self, **k):
            pass

        def __call__(self):
            return False

log = get_logger("model")
# forward() runs every step; only trace the first few calls + a periodic sample.
_fwd_gate = StepGate(first_n=2, every=1000)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ContinuousTimeEncoding(nn.Module):
    """Sinusoidal positional encoding keyed on continuous elapsed time (seconds),
    not a discrete index. Needed for image history (configs.IMAGE_HISTORY_*):
    samples are drawn from stochastic Gaussian modes, so actual elapsed time
    varies per sample and per batch row -- a learned/fixed embedding indexed
    0..T-1 would be wrong. Frequencies are scaled to the actual time range of
    interest (min_dt..max_dt), not the large integer-index range the classic
    Transformer PE (get_sinusoid_encoding_table above) assumes.
    """
    def __init__(self, hidden_dim, min_dt=0.05, max_dt=6.0):
        super().__init__()
        assert hidden_dim % 2 == 0
        n_freqs = hidden_dim // 2
        omega = torch.exp(torch.linspace(
            math.log(2 * math.pi / max_dt), math.log(2 * math.pi / min_dt), n_freqs))
        self.register_buffer('omega', omega, persistent=False)

    def forward(self, elapsed_sec):
        """elapsed_sec: [...] (any leading dims) -> [..., hidden_dim]."""
        args = elapsed_sec.unsqueeze(-1) * self.omega
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class HistoryTorqueTactileFusion(nn.Module):
    """Fuses per-camera image-history embeddings with tactile (+ optionally
    torque) per-timestep sequences into one 'decision latent' vector.
    One-directional cross-attention (camera embeddings query the tactile/torque
    sequence) -- only the camera side feeds the pooled output below, so unlike
    TransformerEncoderLayer's cross_attn_1/cross_attn_2 pair (which needs both
    sides updated since both survive into its output), a second cross-attention
    updating the tactile/torque side would be pure wasted compute here.
    """
    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, cam_history_embeds, tactile_per_step, torque_per_step=None):
        """
        cam_history_embeds: [B, num_cam, hidden_dim]
        tactile_per_step: [B, 18, hidden_dim]
        torque_per_step: [B, T_torque, hidden_dim] or None
        Returns: [B, hidden_dim] decision latent.
        """
        query = cam_history_embeds.permute(1, 0, 2)  # [num_cam, B, D]
        kv_parts = [tactile_per_step.permute(1, 0, 2)]
        if torque_per_step is not None:
            kv_parts.append(torque_per_step.permute(1, 0, 2))
        kv = torch.cat(kv_parts, dim=0)  # [T_total, B, D]

        query2 = self.cross_attn(query=query, key=kv, value=kv)[0]
        query = self.norm(query + query2)
        pooled = query.mean(dim=0)  # [B, D], camera side now tactile/torque-informed
        return self.pool_proj(pooled)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, use_tactile, proprioceptive_temporal_horizon, action_dim=None, tactile_mode='predict', use_decision_fusion=False):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        assert num_queries == 100, "num_queries should be non-zero"
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        # 'predict' runs the transformer twice: once to produce tactile_hat,
        # once for the actions with that prediction as an extra token. 'input'
        # keeps tactile as a plain observation token and runs ONE pass -- the
        # aux head is what makes the second pass exist.
        self.tactile_mode = tactile_mode
        self.action_dim = int(action_dim) if action_dim else state_dim
        self.action_head = nn.Linear(hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.use_tactile = use_tactile

        qpos_dim = state_dim * proprioceptive_temporal_horizon #348 WTH why hardcode it
        tactile_dim = 120
        tactile_dim_all = 18 * tactile_dim
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(qpos_dim, hidden_dim)
            self.qpos_mask_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)

            # Torque as a model input (--torque_input CLI arg). Same [65]
            # per-frame shape as qpos (confirmed from meta/info.json -- identical
            # joint names), kept PER-TIMESTEP (not flattened, unlike
            # input_proj_robot_state) since Phase 4's fusion cross-attends over
            # a torque-history sequence, weights shared across timesteps.
            self.input_proj_torque_per_step = nn.Linear(state_dim, hidden_dim)

            # Image history: present-past cross-attention per camera (see
            # configs.IMAGE_HISTORY_*). Built unconditionally -- cheap, and only
            # exercised when forward() receives a 6D image tensor
            # [B,num_cam,T,C,H,W] instead of the normal 5D [B,num_cam,C,H,W]; a
            # plain single-frame forward pass never touches it.
            self.image_time_encoding = ContinuousTimeEncoding(hidden_dim)
            _history_layer = TransformerEncoderLayer(
                hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=0.1, activation='relu')
            self.history_attn = TransformerEncoder(_history_layer, num_layers=1)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(qpos_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        if use_tactile:
            self.input_proj_tactile = nn.Linear(tactile_dim_all, hidden_dim)
            self.tactile_head = nn.Linear(hidden_dim, tactile_dim)
            self.query_embed_tactile = nn.Embedding(18, hidden_dim)
            # Per-timestep projection, additive to input_proj_tactile above (which
            # stays on the flattened [B,18*tactile_dim] path used by the existing
            # dual-pass). This one keeps the 18 steps separate -- [B,18,hidden_dim]
            # -- for cross-attention against image/torque history (weights shared
            # across timesteps, same convention as input_proj_robot_state).
            self.input_proj_tactile_per_step = nn.Linear(tactile_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self.action_dim, hidden_dim)  # CVAE encoder sees the same dims we predict
        self.encoder_joint_proj = nn.Linear(qpos_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # Fusion mechanically requires tactile (it cross-attends camera-history
        # against a tactile[+torque] sequence) -- degrade gracefully rather than
        # building a slot for an ingredient that doesn't exist.
        self.use_decision_fusion = use_decision_fusion and use_tactile
        if use_tactile:
            self.additional_pos_embed = nn.Embedding(5 if self.use_decision_fusion else 4, hidden_dim)
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        if self.use_decision_fusion:
            self.fusion = HistoryTorqueTactileFusion(hidden_dim)

        # ---- architecture summary (once, at build time) ----
        log.info("DETRVAE built:")
        log.info("  hidden_dim=%d  state_dim=%d  action_dim=%d  num_queries=%d  latent_dim=%d",
                 hidden_dim, state_dim, self.action_dim, num_queries, self.latent_dim)
        log.info("  qpos_dim=%d (= state_dim %d x proprio_horizon %d) -> input_proj_robot_state",
                 qpos_dim, state_dim, proprioceptive_temporal_horizon)
        log.info("  cameras=%d %s  (ONE shared backbone is used for all of them)",
                 len(camera_names), camera_names)
        log.info("  backbone out channels=%s -> input_proj -> hidden_dim",
                 backbones[0].num_channels if backbones else None)
        log.info("  use_tactile=%s", use_tactile)
        if use_tactile:
            log.info("    tactile_dim=%d per step, flattened=%d (18 x %d) -> input_proj_tactile",
                     tactile_dim, tactile_dim_all, tactile_dim)
            log.info("    tactile_head -> %d ; query_embed_tactile=18 queries", tactile_dim)
            if self.use_decision_fusion:
                log.info("    additional_pos_embed slots: [0]=tactile [1]=tactile_pred [2]=latent "
                         "[3]=proprio [4]=decision_latent")
            else:
                log.info("    additional_pos_embed slots: [0]=tactile [1]=tactile_pred [2]=latent [3]=proprio")
        else:
            log.info("    additional_pos_embed slots: [0]=latent [1]=proprio")
        log_module_shapes(log, TRACE, self, max_params=200)


    def forward(self, qpos, image, env_state, tactile=None, actions=None, is_pad=None, tactile_next=None, epoch=0, qpos_mask=None, image_history_elapsed_sec=None, torque=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width -- OR, when image history is
            enabled (configs.IMAGE_HISTORY_*), batch, num_cam, T, channel, height,
            width (6D). image_history_elapsed_sec: [B,T] realized seconds-into-
            the-past per sample (required whenever image is 6D).
        env_state: None
        actions: batch, seq, action_dim
        tactile: batch, tactile_dim*seq_tactile
        torque: batch, qpos_dim (same flattened [state_dim x horizon] layout as
            qpos), or None when --torque_input is off
        tactile_next: batch, seq_tactile, tactile_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape

        _trace = _fwd_gate() and log.isEnabledFor(TRACE)
        if _trace:
            log.log(TRACE, "--- DETRVAE.forward  bs=%d is_training=%s epoch=%s ---", bs, is_training, epoch)
            log_tensor(log, TRACE, "in/qpos", qpos)
            log_tensor(log, TRACE, "in/image", image)
            log_tensor(log, TRACE, "in/actions", actions)
            log_tensor(log, TRACE, "in/is_pad", is_pad)
            log_tensor(log, TRACE, "in/tactile", tactile)
            log_tensor(log, TRACE, "in/tactile_next", tactile_next)

        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
            if _trace:
                log.log(TRACE, "CVAE encoder: input %s (cls + qpos + %d actions)",
                        tuple(encoder_input.shape), action_embed.shape[1])
                log_tensor(log, TRACE, "cvae/mu", mu)
                log_tensor(log, TRACE, "cvae/logvar", logvar)
                log_tensor(log, TRACE, "cvae/latent_input", latent_input)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            if _trace:
                log.log(TRACE, "CVAE encoder skipped (inference): latent sampled as zeros")

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            # camera count is whatever the run recorded; --cameras may select a
            # subset. Was hardcoded to 4.
            assert len(self.camera_names) >= 1
            all_cam_history_embeds = []
            for cam_id, cam_name in enumerate(self.camera_names):
                cam_image = image[:, cam_id]  # [B,C,H,W] or [B,T,C,H,W] (history)
                if cam_image.dim() == 5:
                    # Image history path: batch the backbone forward across T,
                    # take the LAST (=current) timestep for the main 196-token
                    # vision path (unchanged downstream), and separately GAP +
                    # history_attn the full T-length sequence into one
                    # history-informed embedding per camera.
                    Bh, Th, Ch, Hh, Wh = cam_image.shape
                    flat = cam_image.reshape(Bh * Th, Ch, Hh, Wh)
                    feat_flat, pos_flat = self.backbones[0](flat)
                    feat_flat = self.input_proj(feat_flat[0])  # [B*T,hidden_dim,gh,gw]
                    # PositionEmbeddingSine always returns batch=1 regardless of
                    # input batch size (it's a pure function of H,W, broadcast to
                    # the real batch later inside Transformer.forward via
                    # .repeat(1,bs,1)) -- use it as-is, exactly like the
                    # single-frame path below does, not reshaped by Bh/Th.
                    pos_now = pos_flat[0]  # [1,hidden_dim,gh,gw]
                    _, hd, gh, gw = feat_flat.shape
                    feat_all = feat_flat.view(Bh, Th, hd, gh, gw)

                    all_cam_features.append(feat_all[:, -1])
                    all_cam_pos.append(pos_now)

                    assert image_history_elapsed_sec is not None, \
                        "image_history_elapsed_sec is required when image has a time dimension"
                    pooled = feat_all.mean(dim=(-2, -1))  # [B,T,hidden_dim]
                    pooled = pooled + self.image_time_encoding(image_history_elapsed_sec)
                    pooled_seq = pooled.permute(1, 0, 2)  # [T,B,hidden_dim]
                    hist_out = self.history_attn(pooled_seq)  # [T,B,hidden_dim]
                    all_cam_history_embeds.append(hist_out[-1])  # [B,hidden_dim], "now" slot
                    if _trace:
                        log_tensor(log, TRACE, f"backbone/cam{cam_id}({cam_name.split('.')[-1]}) history",
                                   feat_all)
                else:
                    features, pos = self.backbones[0](cam_image) # HARDCODED
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
                    if _trace:
                        log_tensor(log, TRACE, f"backbone/cam{cam_id}({cam_name.split('.')[-1]})", features)
            # cam_history_embeds/torque_embeds are LOCAL to this call on purpose
            # (not read back from self._debug_last_* below) -- an instance
            # attribute would carry stale data from a PREVIOUS call into a
            # later one that doesn't have history/torque this time.
            cam_history_embeds = torch.stack(all_cam_history_embeds, dim=1) if all_cam_history_embeds else None
            if cam_history_embeds is not None:
                self._debug_last_history_embeds = cam_history_embeds  # [B,num_cam,hidden_dim], for verification scripts only
            torque_embeds = None
            if torque is not None:
                # torque arrives flattened [B, qpos_dim] like qpos -- reshape back
                # to per-timestep for the per-step projection.
                B_t = torque.shape[0]
                torque_per_step = torque.view(B_t, -1, self.input_proj_torque_per_step.in_features)
                torque_embeds = self.input_proj_torque_per_step(torque_per_step)  # [B,horizon,hidden_dim]
                self._debug_last_torque_embeds = torque_embeds  # for verification scripts only
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            if qpos_mask is not None and qpos_mask.any():
                proprio_input = torch.where(qpos_mask.unsqueeze(-1), self.qpos_mask_embed, proprio_input)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            if _trace:
                log.log(TRACE, "vision: %d cams concatenated along W -> src %s (=%d image tokens)",
                        len(all_cam_features), tuple(src.shape), src.shape[2] * src.shape[3])
                log_tensor(log, TRACE, "proprio_input", proprio_input)
            if self.use_tactile:
                tactile_input = self.input_proj_tactile(tactile)
                if _trace:
                    log_tensor(log, TRACE, "tactile_input (pass1)", tactile_input)
                # Per-timestep projection, additive to tactile_input above.
                B_tac = tactile.shape[0]
                tactile_per_step = tactile.view(B_tac, 18, self.input_proj_tactile_per_step.in_features)
                tactile_per_step_embeds = self.input_proj_tactile_per_step(tactile_per_step)  # [B,18,hidden_dim]
                self._debug_last_tactile_per_step_embeds = tactile_per_step_embeds  # for verification scripts only

                decision_latent = None
                if self.use_decision_fusion:
                    assert cam_history_embeds is not None, (
                        "use_decision_fusion=True but this forward call's image has no "
                        "time dimension (--image_history must be on) -- "
                        "decision_latent cannot be computed without image history.")
                    decision_latent = self.fusion(cam_history_embeds, tactile_per_step_embeds, torque_embeds)
                    if _trace:
                        log_tensor(log, TRACE, "decision_latent", decision_latent)

                if self.tactile_mode == "input":
                    # one pass, tactile as a plain observation token. The
                    # tactile_pred=None branch in transformer.py already lays the
                    # tokens out as [tactile, latent, proprio(, decision_latent)] + image.
                    hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, tactile_input, None, decision_latent)[0]
                    tactile_hat = None
                else:
                    hs_tactile = self.transformer(src, None, self.query_embed_tactile.weight, pos,
                                                  latent_input, proprio_input, self.additional_pos_embed.weight,
                                                  tactile_input, None, decision_latent)[0]

                    tactile_hat = self.tactile_head(hs_tactile)  ##[bs, 18, tactile_dim]
                    B, T, D = tactile_hat.shape
                    _teacher_forced = epoch < 75
                    if _teacher_forced:
                        tactile_pred_input = tactile_next.view(B, T * D)
                    else:
                        tactile_pred_input = tactile_hat.view(B, T * D)
                    tactile_pred_input = self.input_proj_tactile(tactile_pred_input)
                    if _trace:
                        log.log(TRACE, "pass1 -> tactile_hat %s ; pass2 tactile_pred source=%s (epoch=%s, threshold=75)",
                                tuple(tactile_hat.shape),
                                "GROUND TRUTH tactile_next" if _teacher_forced else "SELF-PREDICTED tactile_hat",
                                epoch)
                        log_tensor(log, TRACE, "tactile_hat", tactile_hat)
                        log_tensor(log, TRACE, "tactile_pred_input (pass2)", tactile_pred_input)
                    hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, tactile_input, tactile_pred_input, decision_latent)[0]
            else:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
                tactile_hat = None
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        if _trace:
            log_tensor(log, TRACE, "decoder/hs", hs)
            log_tensor(log, TRACE, "out/a_hat", a_hat)
            log_tensor(log, TRACE, "out/tactile_hat", tactile_hat)
            log.log(TRACE, "--- DETRVAE.forward done ---")
        return a_hat, is_pad_hat, [mu, logvar], tactile_hat



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    # state_dim = 58 # TODO hardcode
    

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    # print("use_tactile:", args.use_tactile)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=args.state_dim,
        action_dim=getattr(args, 'action_dim', None),
        tactile_mode=getattr(args, 'tactile_mode', 'predict'),
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        use_tactile = args.use_tactile,
        proprioceptive_temporal_horizon = args.proprioceptive_temporal_horizon,
        use_decision_fusion = getattr(args, 'use_decision_fusion', False),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("ACT model built: %.2fM trainable parameters", n_parameters / 1e6)
    log.info("  transformer: d_model=%d nhead=%d enc_layers=%d dec_layers=%d ffn=%d dropout=%s pre_norm=%s",
             args.hidden_dim, args.nheads, args.enc_layers, args.dec_layers,
             args.dim_feedforward, args.dropout, args.pre_norm)
    log.info("  cvae encoder: %d layers (shares TransformerEncoderLayer, use_tactile=False)", args.enc_layers)
    log.info("  backbone=%s lr_backbone=%s dilation=%s position_embedding=%s",
             args.backbone, args.lr_backbone, args.dilation, args.position_embedding)

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
