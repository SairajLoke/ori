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
        log.info("ACTPolicy: kl_weight=%s  num_queries=%s  state_dim=%s  use_tactile=%s",
                 self.kl_weight, args_override.get('num_queries'),
                 args_override.get('state_dim'), args_override.get('use_tactile'))
        log.info("  optimizer=%s  lr=%s  lr_backbone=%s",
                 type(optimizer).__name__, args_override.get('lr'), args_override.get('lr_backbone'))


    @staticmethod
    def _masked_l1(pred, target, is_pad):
        """Mean L1 over the UNPADDED entries only.

        `is_pad` is [B, T] (True = fabricated/out-of-episode). The naive
        `(err * ~is_pad).mean()` zeroes the padded terms but still divides by
        the full B*T*D count, so the reported loss shrinks purely because a
        batch happened to contain more padding. Divide by the number of real
        elements instead: n_valid_timesteps * D.
        """
        err = F.l1_loss(pred, target, reduction='none')          # [B, T, D]
        if is_pad is None:
            return err.mean()
        valid = (~is_pad).unsqueeze(-1).to(err.dtype)            # [B, T, 1]
        denom = valid.sum() * err.shape[-1]                      # real elements
        return (err * valid).sum() / denom.clamp(min=1.0)

    def __call__(self, qpos, image, actions=None, is_pad=None, device=None, tactile=None,
                 tactile_next=None, tactile_next_pad=None, epoch=0):
        env_state = None

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # print('model_num_queries', self.model.num_queries)
            # _stats('actions', actions)
            # _stats('image', image)
            # _stats('env_state', env_state)
            # _stats('tactile', tactile)
            # _stats('tactile_next', tactile_next)
            

            a_hat, is_pad_hat, (mu, logvar), tac_hat = self.model(qpos, image, env_state, tactile, actions, is_pad, tactile_next, epoch=epoch)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            loss_dict['l1'] = self._masked_l1(a_hat, actions, is_pad)
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            if tac_hat is not None:
                # The tactile target lives on its own time axis (18 future
                # steps), so it needs its own mask -- the action mask has a
                # different meaning and a different length.
                loss_dict['l1_tac'] = self._masked_l1(tac_hat, tactile_next, tactile_next_pad)
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['l1_tac']

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

