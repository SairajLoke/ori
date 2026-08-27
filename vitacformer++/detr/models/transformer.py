# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython
e = IPython.embed

# `detr` is pip-installed as its own package; degrade gracefully if my_utils
# is not on the path (see detr_vae.py for the same shim).
try:
    from my_utils.ori_logging import get_logger, log_tensor, TRACE, StepGate
except ImportError:  # pragma: no cover
    import logging as _logging
    TRACE = 5

    def get_logger(name):
        return _logging.getLogger("ori." + name)

    def log_tensor(*a, **k):
        pass

    class StepGate:
        def __init__(self, **k):
            pass

        def __call__(self):
            return False

log = get_logger("transformer")
_tf_gate = StepGate(first_n=4, every=2000)     # 2 passes/step, so 4 == first 2 steps
_enc_gate = StepGate(first_n=2, every=0)       # token-split layout only, it never changes


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, use_tactile=False,
                 explicit_flash_attn=False, use_decision_fusion=False):
        super().__init__()

        # explicit_flash_attn -> need_weights=False -> nn.MultiheadAttention may
        # dispatch to fused SDPA. Without it the fast path is unreachable, since
        # returning weights is incompatible with flash/mem-efficient kernels.
        _nw = not explicit_flash_attn
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, use_tactile,
                                                need_weights=_nw, use_decision_fusion=use_decision_fusion)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                need_weights=_nw)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None, tactile=None, tactile_pred=None, decision_latent=None):

        _trace = _tf_gate() and log.isEnabledFor(TRACE)

        if len(src.shape) == 4: # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            if _trace:
                log.log(TRACE, "Transformer.forward: image src [%d,%d,%d,%d] -> %d tokens x bs=%d x d=%d ; %d decoder queries",
                        bs, c, h, w, h * w, bs, c, query_embed.shape[0])

            if tactile is None:
                additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1) # seq, bs, dim
                pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

                addition_input = torch.stack([latent_input, proprio_input], axis=0)
                src = torch.cat([addition_input, src], axis=0)
                pred_action = True
            else:
                # additional_pos_embed has 4 rows normally, 5 when decision_latent
                # is used (see DETRVAE's use_decision_fusion) -- this assertion
                # catches a lockstep drift against TransformerEncoderLayer's
                # use_decision_fusion flag at the source, loudly, instead of
                # silently mis-slicing downstream.
                expected_slots = 5 if decision_latent is not None else 4
                assert additional_pos_embed.shape[0] == expected_slots, (
                    f"additional_pos_embed has {additional_pos_embed.shape[0]} slots but "
                    f"decision_latent={'present' if decision_latent is not None else 'None'} "
                    f"expects {expected_slots} -- DETRVAE's use_decision_fusion and this call "
                    f"disagree on the token layout.")

                tokens = [tactile]
                pos_list = [additional_pos_embed[0]]

                if tactile_pred is not None:
                    tokens.append(tactile_pred)
                    pos_list.append(additional_pos_embed[1])
                    pred_action = True
                else:
                    pred_action = False

                tokens.extend([latent_input, proprio_input])
                pos_list.extend([additional_pos_embed[2], additional_pos_embed[3]])
                if decision_latent is not None:
                    tokens.append(decision_latent)
                    pos_list.append(additional_pos_embed[4])

                addition_input = torch.stack(tokens, dim=0)  # [3 + 1 (+1), B, D]
                pos_list = torch.stack(pos_list, dim=0).unsqueeze(1).repeat(1, bs, 1)  # [N_token, B, D]

                pos_embed = torch.cat([pos_list, pos_embed], dim=0)  # [N_token + HW, B, D]
                src = torch.cat([addition_input, src], dim=0)  # [N_token + HW, B, D]

                if _trace:
                    _layout = (["tactile"] + (["tactile_pred"] if tactile_pred is not None else [])
                               + ["latent", "proprio"] + (["decision_latent"] if decision_latent is not None else [])
                               + [f"image x{src.shape[0] - len(tokens)}"])
                    log.log(TRACE, "  encoder token layout (pred_action=%s): %s  -> total src %s",
                            pred_action, " | ".join(_layout), tuple(src.shape))
        else:
            assert len(src.shape) == 3
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, pred_action=pred_action)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        if _trace:
            log_tensor(log, TRACE, "  encoder memory", memory)
            log_tensor(log, TRACE, "  decoder hs (all layers)", hs)
        return hs

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, pred_action=False):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, pred_action=pred_action)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


# nn.MultiheadAttention defaults to need_weights=True, which materialises the
# full [T,T] attention matrix per head and blocks the fused SDPA (flash /
# mem-efficient) kernels. EVERY call site here takes [0] and discards the
# weights, so that work was always wasted.
#
# Carried per-LAYER rather than as a module global: a single process can build
# several models (the eval scripts do), and a global would let the last build
# silently change the attention path of the earlier ones.


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_tactile=False,
                 need_weights=True, use_decision_fusion=False):
        super().__init__()
        self.need_weights = need_weights
        # print("use_tactile encoder", use_tactile)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        if use_tactile:
            self.cross_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.cross_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        if use_tactile:
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        if use_tactile:
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.use_tactile = use_tactile
        # Fixed at build time (unlike pred_action, which varies per forward call
        # between pass 1/pass 2) -- whether Transformer.forward's token assembly
        # includes a decision_latent token in the "middle" group (alongside
        # latent/proprio). Shifts the middle/other slice boundary below by +1
        # when true. MUST match Transformer.forward's token count exactly, or
        # this cross-attends the wrong tokens with no error -- see the
        # slice-boundary comment in forward_post.
        self.use_decision_fusion = use_decision_fusion

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, pred_action=False):
        # ==== Self-attention ====
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              need_weights=self.need_weights)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.use_tactile:
            # ==== Cross-attention ====
            # Middle group is [latent, proprio] normally, [latent, proprio,
            # decision_latent] when use_decision_fusion -- Transformer.forward
            # appends decision_latent to the END of that group, so only the
            # upper boundary (where "other"/image tokens start) shifts by +1;
            # the tactile_token slice at the front is untouched either way.
            _mid_extra = 1 if self.use_decision_fusion else 0
            if pred_action:
                tactile_token = src[:2]
                tactile_pos = pos[:2]
                other_tokens = src[4 + _mid_extra:]
                other_pos = pos[4 + _mid_extra:]
                middle_tokens = src[2:4 + _mid_extra]
                middle_pos = pos[2:4 + _mid_extra]
            else:
                tactile_token = src[:1]
                tactile_pos = pos[:1]
                other_tokens = src[3 + _mid_extra:]
                other_pos = pos[3 + _mid_extra:]
                middle_tokens = src[1:3 + _mid_extra]
                middle_pos = pos[1:3 + _mid_extra]

            if _enc_gate() and log.isEnabledFor(TRACE):
                # The slice boundaries here MUST match the token order built in
                # Transformer.forward. Log them so a mismatch is obvious.
                log.log(TRACE,
                        "  enc layer split (pred_action=%s, use_decision_fusion=%s): tactile=%s "
                        "middle(latent,proprio[,decision_latent])=%s other(image)=%s",
                        pred_action, self.use_decision_fusion, tuple(tactile_token.shape),
                        tuple(middle_tokens.shape), tuple(other_tokens.shape))

            tactile_token2 = self.cross_attn_1(
                query=self.with_pos_embed(tactile_token, tactile_pos),
                key=self.with_pos_embed(other_tokens, other_pos),
                value=other_tokens, need_weights=self.need_weights)[0]
            
            other_tokens2 = self.cross_attn_2(
                query=self.with_pos_embed(other_tokens, other_pos),
                key=self.with_pos_embed(tactile_token, tactile_pos),
                value=tactile_token, need_weights=self.need_weights)[0]
            
            tactile_token = tactile_token + self.dropout2(tactile_token2)
            tactile_token = self.norm2(tactile_token)
            other_tokens = other_tokens + self.dropout3(other_tokens2)
            other_tokens = self.norm3(other_tokens)

            src = torch.cat([tactile_token, middle_tokens, other_tokens], dim=0)

        # ==== Feedforward ====
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        src = self.norm4(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              need_weights=self.need_weights)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm4(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout4(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, pred_action=False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, pred_action=pred_action)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, need_weights=True):
        super().__init__()
        self.need_weights = need_weights
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

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   need_weights=self.need_weights)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        use_tactile=args.use_tactile,
        explicit_flash_attn=getattr(args, 'explicit_flash_attn', False),
        use_decision_fusion=getattr(args, 'use_decision_fusion', False),
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
