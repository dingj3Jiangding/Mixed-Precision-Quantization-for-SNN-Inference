import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron


class SJTimeWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        T: int = 16,
        encoder: str = "direct",          # "direct" or "poisson"
        force_step_mode: str = "s",       # "s" | "m" | "auto"
        aggregate: str | None = "mean",   # None -> return [T,B,C]; "mean"/"sum"/"last"
        collect_spike_rate: bool = False,
    ):
        super().__init__()
        self.base = base_model
        self.T = int(T)
        self.encoder = encoder
        self.force_step_mode = force_step_mode
        self.aggregate = aggregate
        self.collect_spike_rate = collect_spike_rate

        # Spike stats buffers (for logging)
        self._spike_sum = 0.0
        self._spike_cnt = 0.0
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Set step mode safely
        self._configure_step_mode()

        # Optional: attach spike hooks
        if self.collect_spike_rate:
            self._attach_spike_hooks()

    def _configure_step_mode(self):
        mode = self.force_step_mode
        if mode not in ("s", "m", "auto"):
            raise ValueError("force_step_mode must be 's', 'm', or 'auto'.")

        # Practical default:
        # - If you plan to loop in wrapper -> enforce 's'
        # - If you want base model to consume [T,B,...] directly -> enforce 'm'
        if mode == "auto":
            # Conservative: default to single-step to avoid accidental double time-loop
            mode = "s"

        functional.set_step_mode(self.base, step_mode=mode)
        self._step_mode = mode  # for internal logic

    def _encode(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        x: [B,C,H,W] -> x_seq: [T,B,C,H,W]
        """
        if self.encoder == "direct":
            return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)

        if self.encoder == "poisson":
            # NOTE: This assumes x is in [0,1] as firing probability.
            # If you Normalize inputs, poisson should be applied BEFORE Normalize.
            if x.min() < 0 or x.max() > 1:
                raise ValueError(
                    "Poisson encoding expects inputs in [0,1]. "
                    "Apply poisson before Normalize or rescale appropriately."
                )
            # Bernoulli sampling per timestep
            rand = torch.rand((T,) + x.shape, device=x.device, dtype=x.dtype)
            return (rand < x.unsqueeze(0)).to(x.dtype)      # poisson sampling: x 就是伯努利参数 p -- >x 输出1 <x 输出0

        raise ValueError("encoder must be 'direct' or 'poisson'.")

    def reset(self):
        # SpikingJelly canonical reset
        functional.reset_net(self.base)

        # reset spike stats per batch if enabled
        if self.collect_spike_rate:
            self._spike_sum = 0.0
            self._spike_cnt = 0.0

    def _attach_spike_hooks(self):
        # Remove existing hooks if any
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        def hook_fn(module, inp, out):
            # out could be Tensor or tuple/list; we count spikes in Tensor
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                return
            # For spiking neurons, out is typically spikes (0/1 float/bool)
            # Count mean firing activity
            self._spike_sum += float(out.detach().sum().item())
            self._spike_cnt += float(out.detach().numel())

        for m in self.base.modules():
            if isinstance(m, neuron.BaseNode):
                self._hooks.append(m.register_forward_hook(hook_fn))

    def spike_rate(self) -> float | None:
        if not self.collect_spike_rate:
            return None
        if self._spike_cnt == 0:
            return 0.0
        return self._spike_sum / self._spike_cnt

    def _aggregate_logits(self, logits_seq: torch.Tensor) -> torch.Tensor:
        """
        logits_seq: [T,B,C] -> logits: [B,C] depending on aggregate
        """
        if self.aggregate is None:
            return logits_seq
        if self.aggregate == "mean":
            return logits_seq.mean(dim=0)
        if self.aggregate == "sum":
            return logits_seq.sum(dim=0)
        if self.aggregate == "last":
            return logits_seq[-1]
        raise ValueError("aggregate must be None/'mean'/'sum'/'last'.")

    def forward(self, x: torch.Tensor, T: int | None = None):
        """
        Accepts:
          - x:     [B,C,H,W]
          - x_seq: [T,B,C,H,W]
        Returns:
          - if aggregate is None: logits_seq [T,B,num_classes]
          - else: logits [B,num_classes]
        """
        T = int(self.T if T is None else T)

        # Build [T,B,C,H,W] if input is
