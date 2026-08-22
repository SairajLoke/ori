# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
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


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, use_tactile, proprioceptive_temporal_horizon, action_dim=None, tactile_mode='predict'):
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

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self.action_dim, hidden_dim)  # CVAE encoder sees the same dims we predict
        self.encoder_joint_proj = nn.Linear(qpos_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        if use_tactile:
            self.additional_pos_embed = nn.Embedding(4, hidden_dim)
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

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
            log.info("    additional_pos_embed slots: [0]=tactile [1]=tactile_pred [2]=latent [3]=proprio")
        else:
            log.info("    additional_pos_embed slots: [0]=latent [1]=proprio")
        log_module_shapes(log, TRACE, self, max_params=200)


    def forward(self, qpos, image, env_state, tactile=None, actions=None, is_pad=None, tactile_next=None, epoch=0):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        tactile: batch, tactile_dim*seq_tactile
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
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                if _trace:
                    log_tensor(log, TRACE, f"backbone/cam{cam_id}({cam_name.split('.')[-1]})", features)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
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
                if self.tactile_mode == "input":
                    # one pass, tactile as a plain observation token. The
                    # tactile_pred=None branch in transformer.py already lays the
                    # tokens out as [tactile, latent, proprio] + image.
                    hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, tactile_input, None)[0]
                    tactile_hat = None
                else:
                    hs_tactile = self.transformer(src, None, self.query_embed_tactile.weight, pos, 
                                                  latent_input, proprio_input, self.additional_pos_embed.weight, 
                                                  tactile_input, None)[0]
                
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
                    hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, tactile_input, tactile_pred_input)[0]
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
        proprioceptive_temporal_horizon = args.proprioceptive_temporal_horizon
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
