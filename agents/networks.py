"""Actor-critic network.

Separate trunks for policy and value. Shared trunks save a little compute but
couple the value error into the policy gradient, and Notebooks 02 and 06 both
reason about V and pi as separate objects -- keeping them separate keeps that
reasoning honest.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MiniGridCNN(nn.Module):
    """The reference MiniGrid encoder, made grid-size-agnostic.

    The architecture in lcswillems/rl-starter-files, which the official MiniGrid
    docs reproduce as their StableBaselines3 feature extractor:

        Conv2d(3, 16, k=2) -> ReLU -> MaxPool2d(2)
        Conv2d(16, 32, k=2) -> ReLU
        Conv2d(32, 64, k=2) -> ReLU
        embedding = ((n-1)//2 - 2) * ((m-1)//2 - 2) * 64

    That stack only type-checks when the grid is large enough. It is built for
    the 7x7 egocentric view; on the FULLY OBSERVABLE 5x5 and 6x6 grids the
    curriculum uses, the spatial dimension hits 1x1 before the last conv and
    the module raises. So each layer is applied only when the current spatial
    size allows it, and a kernel-2 conv degrades to kernel-1 (a per-cell
    channel mixer) rather than being dropped -- which keeps the parameter
    count and the 64-wide embedding identical across grid sizes, so a
    checkpoint warm-starts cleanly from one rung of the curriculum to the next.

    Verified identical to the reference stack wherever the reference is legal:
        7x7 -> 64,  8x8 -> 64,  16x16 -> 1600   (= the formula above)
        5x5 -> 64,  6x6 -> 64                   (reference would crash)

    Observations arrive FLAT (the rollout buffer stores one row per step), so the
    module reshapes to (H, W, C) and transposes to NCHW itself. Keeping the flat
    representation everywhere else means the trajectory schema, the checkpoint
    format and the landscape files are unchanged from the vector-observation
    environments.
    """

    def __init__(self, obs_shape: Sequence[int]):
        super().__init__()
        h, w, c = obs_shape
        self.obs_shape = (int(h), int(w), int(c))
        self.net = nn.Sequential(*self._build(int(c), int(h), int(w)))
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.out_dim = int(self.net(dummy).flatten(1).shape[1])

    @staticmethod
    def _build(c: int, h: int, w: int) -> list[nn.Module]:
        layers: list[nn.Module] = []

        def conv(cin: int, cout: int) -> None:
            nonlocal h, w
            k = 2 if (h >= 2 and w >= 2) else 1
            layers.extend([nn.Conv2d(cin, cout, (k, k)), nn.ReLU()])
            h -= k - 1
            w -= k - 1

        conv(c, 16)
        if h >= 4 and w >= 4:          # the reference pool, skipped on tiny grids
            layers.append(nn.MaxPool2d((2, 2)))
            h //= 2
            w //= 2
        conv(16, 32)
        conv(32, 64)
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w, c = self.obs_shape
        x = x.view(-1, h, w, c).permute(0, 3, 1, 2)   # NHWC -> NCHW
        return self.net(x).flatten(1)


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation: str, out_std: float) -> nn.Sequential:
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(prev, h)), act()]
        prev = h
    layers += [layer_init(nn.Linear(prev, out_dim), std=out_std)]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        prob_floor: float = 0.0,
        encoder: str = "mlp",
        obs_shape: Sequence[int] | None = None,
        share_encoder: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.encoder_kind = encoder
        self.share_encoder = bool(share_encoder)
        self.obs_shape = tuple(obs_shape) if obs_shape is not None else None
        self.n_actions = n_actions
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        # Mixture weight epsilon in  pi = (1-eps)*softmax(logits) + eps/K,
        # which guarantees pi(a|s) >= eps/K for EVERY action.
        #
        # This exists because entropy regularisation cannot express that
        # constraint. Measured on Taxi-v4 after 1M steps with the entropy floor
        # active: mean policy entropy 1.231 (of log 6 = 1.792), which looks
        # healthy -- but pi(PICKUP) averaged 0.0016 and pi(DROPOFF) peaked at
        # 0.007 across all 500 states. Entropy is maximised just as happily by
        # spreading mass over the four movement actions, and a policy uniform
        # over those four has entropy log 4 = 1.386, higher still, with a
        # measured 0/2000 success rate. The failure is per-action, so the floor
        # has to be per-action.
        self.prob_floor = float(prob_floor)

        if encoder == "cnn":
            if obs_shape is None:
                raise ValueError("encoder='cnn' requires obs_shape=(H, W, C)")
            # share_encoder=False -> one encoder per head, matching the
            # separate-trunk choice made for the MLP: the value loss must not
            # shape the policy's features, which is what NB02/NB06 reason about.
            # share_encoder=True  -> one trunk for both, which is what
            # rl-starter-files does. Halves the parameters and lets the value
            # loss help shape features, which matters when reward is sparse and
            # the policy gradient carries almost no signal.
            self.actor_enc = MiniGridCNN(obs_shape)
            self.critic_enc = self.actor_enc if self.share_encoder else MiniGridCNN(obs_shape)
            emb = self.actor_enc.out_dim
        else:
            self.actor_enc = None
            self.critic_enc = None
            emb = obs_dim

        # out_std=0.01 on the policy head -> near-uniform initial policy.
        self.actor = _mlp(emb, hidden_sizes, n_actions, activation, out_std=0.01)
        self.critic = _mlp(emb, hidden_sizes, 1, activation, out_std=1.0)

    def _actor_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x if self.actor_enc is None else self.actor_enc(x))

    def _critic_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x if self.critic_enc is None else self.critic_enc(x)).squeeze(-1)

    # -- basic heads -------------------------------------------------------- #

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self._critic_value(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._actor_logits(x)

    def distribution(self, x: torch.Tensor) -> Categorical:
        return self._dist(self._actor_logits(x))

    def _dist(self, logits: torch.Tensor) -> Categorical:
        if self.prob_floor <= 0.0:
            return Categorical(logits=logits)
        p = torch.softmax(logits, dim=-1)
        eps = self.prob_floor
        return Categorical(probs=(1.0 - eps) * p + eps / self.n_actions)

    # -- rollout / update interfaces ---------------------------------------- #

    @torch.no_grad()
    def act(self, x: torch.Tensor):
        """Sample an action. Returns (action, logprob, value, probs)."""
        dist = self._dist(self._actor_logits(x))
        action = dist.sample()
        return action, dist.log_prob(action), self.value(x), dist.probs

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Returns (logprob, entropy, value) with gradients attached."""
        dist = self._dist(self._actor_logits(x))
        return dist.log_prob(actions), dist.entropy(), self.value(x)

    def evaluate_all_actions(self, x: torch.Tensor):
        """(log_probs (B, K), entropy (B,), value (B,)) with gradients attached.

        PPO-CF needs the log-probability of EVERY action, not just the sampled
        one, because its policy loss is a sum over all K actions weighted by the
        counterfactual advantage of each. Taken from the same `_dist` as
        `evaluate_actions`, so the probability floor is applied identically --
        the behaviour probabilities stored in the rollout buffer include the
        floor, and a mismatch here would silently corrupt every ratio.
        """
        dist = self._dist(self._actor_logits(x))
        return torch.log(dist.probs.clamp_min(1e-12)), dist.entropy(), self.value(x)

    # -- convenience for NB02-06 -------------------------------------------- #

    @torch.no_grad()
    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist(self._actor_logits(x)).probs

    # -- parameter groups, for separate gradient clipping ------------------- #

    def actor_parameters(self):
        """Policy parameters. With a shared encoder the trunk belongs to
        neither head alone, so it is excluded here and from
        `critic_parameters` -- see `can_clip_separately`."""
        if self.actor_enc is not None and not self.share_encoder:
            yield from self.actor_enc.parameters()
        yield from self.actor.parameters()

    def critic_parameters(self):
        if self.critic_enc is not None and not self.share_encoder:
            yield from self.critic_enc.parameters()
        yield from self.critic.parameters()

    @property
    def can_clip_separately(self) -> bool:
        """False when a shared trunk means some parameters belong to both
        heads, in which case a single global clip is the only honest option."""
        return not (self.share_encoder and self.actor_enc is not None)

    def config_dict(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "prob_floor": self.prob_floor,
            "encoder": self.encoder_kind,
            "obs_shape": list(self.obs_shape) if self.obs_shape else None,
            "share_encoder": self.share_encoder,
        }

    @classmethod
    def from_config_dict(cls, d: dict) -> "ActorCritic":
        return cls(d["obs_dim"], d["n_actions"], d["hidden_sizes"], d["activation"],
                   d.get("prob_floor", 0.0), d.get("encoder", "mlp"), d.get("obs_shape"),
                   d.get("share_encoder", False))
