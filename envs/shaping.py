"""Optional reward shaping and exploration bonuses. Both default to OFF.

READ THIS BEFORE TURNING EITHER ON. The critic learns V for whatever reward it
was trained on, and NB02's oracle is defined as

    A_CF(s, a) = r + gamma * V(s') - V_pi(s)

against that same reward. So a change to the reward is a change to the object
the oracle measures.

  * POTENTIAL-BASED SHAPING is recoverable. With F(s, s') = gamma*Phi(s') - Phi(s),
    Ng et al. (1999) gives V_shaped(s) = V_true(s) - Phi(s) exactly, and the
    optimal policy is unchanged. NB02 can undo it by adding Phi back, so this
    is the oracle-safe option.

  * THE COUNT BONUS IS NOT RECOVERABLE. It changes the MDP, and there is no
    closed form relating V_bonus to V_true. It is only safe if it has annealed
    to exactly zero well before the checkpoint the oracle is built on, which is
    what `count_bonus_anneal_frac` and the trainer's checkpoint assertion
    enforce.

Both operate on an ABSTRACT MiniGrid state -- (agent col, row, dir, carrying a
key, door open) -- rather than the raw grid encoding, because a novelty bonus
over raw grids would count every layout as novel forever.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MiniGridProbe:
    """Cheap per-step read of the DoorKey sub-goals.

    Scanning the whole grid every step for the door would cost W*H per env per
    step. DoorKey has exactly one door, so the probe locates it once per reset
    and then reads that single cell.

    The sub-goal rates this produces are the diagnostic that separates the
    three DoorKey failure modes:
        key flat, door flat     -> the policy never gets started
        key rising, door flat   -> stuck at the door (the 3M 8x8 run)
        door rising, solve flat -> stuck between the door and the goal
    """

    def __init__(self, env):
        self.env = env
        self.doors: list[tuple[tuple[int, int], str]] = []
        self.mode = "key_door"
        self.sub1 = False
        self.sub2 = False
        self.reset_probe()

    def reset_probe(self) -> None:
        self.doors, has_key_obj = self._scan()
        # DoorKey has one door and a key. RedBlueDoors has two doors and no key,
        # and its sub-goals are "red opened" then "blue opened" IN THAT ORDER.
        # Both cases return a 2-tuple from observe(), so nothing downstream
        # needs to know which environment it is looking at.
        self.mode = "key_door" if has_key_obj else "two_doors"
        self.sub1 = False
        self.sub2 = False

    def _scan(self):
        u = self.env.unwrapped
        grid = getattr(u, "grid", None)
        doors, has_key = [], False
        if grid is None:
            return doors, has_key
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell is None:
                    continue
                if cell.type == "door":
                    doors.append(((i, j), cell.color, bool(getattr(cell, "is_locked", False))))
                elif cell.type == "key":
                    has_key = True
        # deterministic order: red before blue, else grid order
        order = {"red": 0, "blue": 1}
        if has_key:
            # Key/door tasks such as KeyCorridor can contain ordinary hallway
            # doors before the locked target door. Track the locked door first.
            doors.sort(key=lambda d: (not d[2], d[0][0], d[0][1]))
        else:
            doors.sort(key=lambda d: order.get(d[1], 2 + d[0][0]))
        return [(pos, color) for pos, color, _locked in doors], has_key

    @property
    def door_pos(self):
        """Kept for callers that only care about the first door."""
        return self.doors[0][0] if self.doors else None

    def has_key(self) -> bool:
        carrying = getattr(self.env.unwrapped, "carrying", None)
        return carrying is not None and carrying.type == "key"

    def _door_open(self, idx: int) -> bool:
        if idx >= len(self.doors):
            return False
        cell = self.env.unwrapped.grid.get(*self.doors[idx][0])
        return bool(cell is not None and getattr(cell, "is_open", False))

    def door_open(self) -> bool:
        return self._door_open(0)

    def observe(self) -> tuple[bool, bool]:
        """(subgoal1, subgoal2) now, latching the per-episode flags.

        key_door  -> (carrying the key, the door is open)
        two_doors -> (door 1 is open,   door 2 is open)     [red, then blue]
        """
        if self.mode == "key_door":
            g1, g2 = self.has_key(), self._door_open(0)
        else:
            g1, g2 = self._door_open(0), self._door_open(1)
        self.sub1 |= g1
        self.sub2 |= g2
        return g1, g2

    def abstract_state(self) -> tuple[int, int, int, int, int]:
        u = self.env.unwrapped
        g1, g2 = (self.has_key(), self._door_open(0)) if self.mode == "key_door" \
            else (self._door_open(0), self._door_open(1))
        return (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir), int(g1), int(g2))


class RewardShaper:
    """Applies potential-based shaping and/or a count bonus to one env's reward.

    One instance per environment; `EnvPool` owns them. `progress` is the
    fraction of training elapsed, pushed in by the trainer so the count bonus
    can anneal.
    """

    def __init__(self, cfg, probe: MiniGridProbe | None, gamma: float):
        self.cfg = cfg
        self.probe = probe
        self.gamma = float(gamma)
        self.counts: dict[tuple, int] = {}
        self.progress = 0.0
        self._prev_phi = 0.0
        self._last_shaping = 0.0
        self._last_bonus = 0.0

    # -- potential --------------------------------------------------------- #

    def _phi(self) -> float:
        if not self.cfg.potential_shaping or self.probe is None:
            return 0.0
        g1, g2 = self.probe.observe()
        return self.cfg.potential_key * float(g1) + self.cfg.potential_door * float(g2)

    # -- count bonus ------------------------------------------------------- #

    @property
    def count_coef(self) -> float:
        c = float(self.cfg.count_bonus_coef)
        if c <= 0.0:
            return 0.0
        frac = float(self.cfg.count_bonus_anneal_frac)
        if frac <= 0.0:
            return 0.0
        return c * max(0.0, 1.0 - self.progress / frac)

    def _bonus(self) -> float:
        coef = self.count_coef
        if coef <= 0.0 or self.probe is None:
            return 0.0
        key = self.probe.abstract_state()
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        return coef / np.sqrt(n)

    # -- lifecycle --------------------------------------------------------- #

    @property
    def active(self) -> bool:
        return self.cfg.potential_shaping or self.cfg.count_bonus_coef > 0.0

    def on_reset(self) -> None:
        self._prev_phi = self._phi()

    def on_step(self, reward: float, terminated: bool) -> float:
        """Reward after shaping. Call exactly once per env step, after the step."""
        if not self.active:
            return reward
        # Phi(terminal) must be 0 for the shaping to be policy-invariant.
        phi_next = 0.0 if terminated else self._phi()
        self._last_shaping = self.gamma * phi_next - self._prev_phi
        self._prev_phi = phi_next
        self._last_bonus = self._bonus()
        return reward + self._last_shaping + self._last_bonus

    def info(self) -> dict[str, Any]:
        return {"shaping": self._last_shaping, "bonus": self._last_bonus,
                "count_coef": self.count_coef, "n_visited": len(self.counts)}
