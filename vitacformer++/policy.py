from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch

from detr.main import build_ACT_model_and_optimizer
from my_utils.ori_logging import get_logger, log_tensor, TRACE, StepGate
import IPython
e = IPython.embed

log = get_logger("policy")
_loss_gate = StepGate(first_n=3, every=500)


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.tac_weight = float(args_override.get('tac_weight', 1.0))

        # Per-dimension weights for the action L1 term. A buffer so it follows
        # .to(device), but persistent=False so it never enters the state_dict:
        # the weighting is a training hyperparameter (recorded in
        # training_configs.json), and keeping it out of the checkpoint means
        # enabling it does not break strict loading of older checkpoints.
        # Which contract columns this model predicts. None => all of them.
        # When a subset, targets are sliced to match and the weight vector is
        # sliced the same way, so both stay indexed by MODEL output position.
        pad = args_override.get('predicted_action_dims')
        self.register_buffer(
            'predicted_action_dims',
            None if pad is None else torch.as_tensor(pad, dtype=torch.long),
            persistent=False,
        )
        step_w = args_override.get('action_step_weights')
        self.register_buffer(
            'action_step_weights',
            None if step_w is None else torch.as_tensor(step_w, dtype=torch.float32),
            persistent=False,
        )
        dim_weights = args_override.get('action_dim_weights')
        if dim_weights is not None and pad is not None:
            dim_weights = [dim_weights[i] for i in pad]
        self.register_buffer(
            'action_dim_weights',
            None if dim_weights is None else torch.as_tensor(dim_weights, dtype=torch.float32),
            persistent=False,
        )

        log.info("ACTPolicy: kl_weight=%s  tac_weight=%s  num_queries=%s  state_dim=%s  use_tactile=%s",
                 self.kl_weight, self.tac_weight, args_override.get('num_queries'),
                 args_override.get('state_dim'), args_override.get('use_tactile'))
        log.info("  optimizer=%s  lr=%s  lr_backbone=%s",
                 type(optimizer).__name__, args_override.get('lr'), args_override.get('lr_backbone'))
        if self.action_dim_weights is None:
            log.info("  action dim weights: uniform")
        else:
            w = self.action_dim_weights
            log.info("  action dim weights: mean=%.3f min=%.3f max=%.3f, %d dim(s) zeroed",
                     w.mean().item(), w.min().item(), w.max().item(), int((w == 0).sum().item()))
            from train_eval_utils import JOINT_GROUPS
            for g, idx in JOINT_GROUPS.items():
                log.info("    %-11s -> %s", g,
                         sorted({round(float(w[i]), 4) for i in idx if i < w.numel()}))


    @staticmethod
    def _masked_l1(pred, target, is_pad, dim_weights=None, step_weights=None):
        """Mean L1 over the UNPADDED entries only, optionally per-dim weighted.

        `is_pad` is [B, T] (True = fabricated/out-of-episode). The naive
        `(err * ~is_pad).mean()` zeroes the padded terms but still divides by
        the full B*T*D count, so the reported loss shrinks purely because a
        batch happened to contain more padding. Divide by the number of real
        elements instead: n_valid_timesteps * D.

        `dim_weights` is a [D] vector with mean 1 over its non-zero entries, so
        the weighted loss stays on the same scale as the unweighted one and the
        balance against kl_weight/tac_weight is preserved. The denominator uses
        sum(dim_weights) rather than D for the same reason.
        """
        err = F.l1_loss(pred, target, reduction='none')          # [B, T, D]

        if dim_weights is None:
            per_step = err.sum(-1)                               # [B, T]
            width = float(err.shape[-1])
        else:
            w = dim_weights.to(dtype=err.dtype, device=err.device)
            per_step = (err * w).sum(-1)                          # [B, T]
            width = float(w.sum())

        # Per-timestep weighting. Late chunk steps are worth less: only
        # action_horizon of num_queries rows are ever returned, a receding-horizon
        # caller may consume just a prefix of those, and open-loop error roughly
        # doubles by depth 50. Folded into the denominator the same way
        # dim_weights is, so the L1 term keeps its scale against kl_weight.
        if step_weights is not None:
            sw = step_weights.to(dtype=per_step.dtype, device=per_step.device)  # [T]
            sw = sw[:per_step.shape[1]]
            per_step = per_step * sw
            step_scale = sw
        else:
            step_scale = None

        if is_pad is None:
            n = per_step.numel() * width
            if step_scale is not None:
                n = per_step.shape[0] * float(step_scale.sum()) * width
            return per_step.sum() / max(n, 1.0)

        valid = (~is_pad).to(per_step.dtype)                      # [B, T]
        if step_scale is None:
            denom = (valid.sum() * width).clamp(min=1.0)
        else:
            denom = ((valid * step_scale).sum() * width).clamp(min=1.0)
        return (per_step * valid).sum() / denom

    def __call__(self, qpos, image, actions=None, is_pad=None, device=None, tactile=None,
                 tactile_next=None, tactile_next_pad=None, epoch=0, return_a_hat=False,
                 qpos_mask=None):
        env_state = None

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            if self.predicted_action_dims is not None:
                actions = actions.index_select(-1, self.predicted_action_dims.to(actions.device))

            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # print('model_num_queries', self.model.num_queries)
            # _stats('actions', actions)
            # _stats('image', image)
            # _stats('env_state', env_state)
            # _stats('tactile', tactile)
            # _stats('tactile_next', tactile_next)
            

            a_hat, is_pad_hat, (mu, logvar), tac_hat = self.model(qpos, image, env_state, tactile, actions, is_pad, tactile_next, epoch=epoch, qpos_mask=qpos_mask)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            loss_dict['l1'] = self._masked_l1(a_hat, actions, is_pad, self.action_dim_weights,
                                              self.action_step_weights)
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            if tac_hat is not None:
                # The tactile target lives on its own time axis (18 future
                # steps), so it needs its own mask -- the action mask has a
                # different meaning and a different length.
                loss_dict['l1_tac'] = self._masked_l1(tac_hat, tactile_next, tactile_next_pad)
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['l1_tac'] * self.tac_weight

            if _loss_gate() and log.isEnabledFor(TRACE):
                _n_pad = int(is_pad.sum().item())
                _n_tac_pad = int(tactile_next_pad.sum().item()) if tactile_next_pad is not None else -1
                log.log(TRACE, "loss: l1=%.5f kl=%.5f (xw=%s -> %.5f)%s | total=%.5f | "
                               "padded action steps=%d/%d padded tactile steps=%d",
                        loss_dict['l1'].item(), loss_dict['kl'].item(), self.kl_weight,
                        (loss_dict['kl'] * self.kl_weight).item(),
                        f" l1_tac={loss_dict['l1_tac'].item():.5f}" if 'l1_tac' in loss_dict else "",
                        loss_dict['loss'].item(), _n_pad, is_pad.numel(), _n_tac_pad)
                log_tensor(log, TRACE, "loss/a_hat", a_hat)
                log_tensor(log, TRACE, "loss/actions(target)", actions)
                if tac_hat is not None:
                    log_tensor(log, TRACE, "loss/tac_hat", tac_hat)
                    log_tensor(log, TRACE, "loss/tactile_next(target)", tactile_next)

            # Validation asks for the predictions too, so it can break the error
            # down per joint group without a second forward pass.
            if return_a_hat:
                return loss_dict, a_hat
            return loss_dict
        else: # inference time
            # epoch>=75 disables teacher forcing so the model uses its own predicted tactile_next
            a_hat, _, (_, _), _ = self.model(qpos, image, env_state, tactile, epoch=999, tactile_next=tactile_next) # no action, sample from prior
            return a_hat


    def configure_optimizers(self):
        return self.optimizer
    
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

