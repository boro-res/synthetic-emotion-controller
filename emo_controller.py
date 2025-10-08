#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example Code Accompanying the Preprint:

Synthetic Emotions and Consciousness: Exploring Architectural Boundaries

Hermann Borotschnig

2025

Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2025 Hermann Borotschnig

You are free to:

- Share — copy and redistribute the material in any medium or format  
- Adapt — remix, transform, and build upon the material for any purpose, even commercially.  

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.  
  You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.  

- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

Notices:

You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.

No warranties are given. The license may not give you all of the permissions necessary for your intended use.  
For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode


Emotion-like control (A1–A8) with n-actors, episodic memory, YAML init, and plotting.

Implements:
- Categories (A1)
- Need appraisal -> drives/affect (A2)
- Memory: append-only episodes + Top-K retrieval (A3, A8)
- Affect fusion + policy hints + softmax temperatures (A4)
- Policy→action scoring templates (Seek/Avoid/Explore/Rest) (A5)
- Execute, reappraise, success tagging (A6–A7)
- n-actor 2D world, synchronous stepping, plotting

This code respects the boundedness & hygiene choices from the SI:
- All internal maps are bounded and normalized.
- Softmaxes are stabilized (subtract max).
- Retrieval is Top-K with cosine similarity on L2-normalized keys.


To Run This Code (outputs trajectories.png and metrics.png in directory results)

python emo_controller.py --config config_social.yaml --steps 600 --out_dir results

"""

from __future__ import annotations
import math, random, uuid, sys, os, argparse, json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ----------------------- Utilities (bounded ops) -----------------------


def bounded_linear(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Safe matvec with automatic pad/trim and normalized output.
    - If x is shorter than H's input dim, pad with zeros.
    - If longer, trim x.
    Returns (H @ x) / input_dim to keep outputs in [-1,1]-like scale.
    """
    H = np.asarray(H, dtype=float)
    x = np.asarray(x, dtype=float)
    c = H.shape[1]
    if x.size < c:
        x_pad = np.zeros(c, dtype=float)
        x_pad[:x.size] = x
        x = x_pad
    elif x.size > c:
        x = x[:c]
    out = H @ x
    return out / max(1.0, float(c))


def slerp_angle(theta, target, gain=0.25):
    # Interpolate on circle: theta_new = arg( (1-gain)*e^{i theta} + gain*e^{i target} )
    c1, s1 = math.cos(theta), math.sin(theta)
    c2, s2 = math.cos(target), math.sin(target)
    c = (1.0 - gain)*c1 + gain*c2
    s = (1.0 - gain)*s1 + gain*s2
    return math.atan2(s, c)



def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))

def ang_diff(a: float, b: float) -> float:
    """a - b wrapped to (-pi, pi]."""
    return wrap_angle(a - b)

def vm_bump(delta: float, kappa: float) -> float:
    """Von-Mises-like bump scaled to [0,1], safe for moderate kappa."""
    # Stable form: e^{k(cos d - 1)} in numerator, denominator normalizes to [0,1]
    k = max(0.0, min(kappa, 10.0))
    num = math.exp(k * (math.cos(delta) - 1.0)) - math.exp(-k)
    den = math.exp(k) - math.exp(-k) + 1e-12
    x = num / den
    # Bound numerically
    return float(max(0.0, min(1.0, x)))

def softmax_stable(x: np.ndarray, tau: float) -> np.ndarray:
    """Safe softmax with temperature tau>0 over last dim."""
    t = max(1e-6, float(tau))
    xm = x - np.max(x)
    e = np.exp(xm / t)
    Z = np.sum(e)
    if Z <= 0.0 or not np.isfinite(Z):
        # fallback: uniform
        return np.ones_like(x) / len(x)
    out = e / Z
    return out

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def clip01(x):
    return max(0.0, min(1.0, x))

def clip11(x):
    return max(-1.0, min(1.0, x))

# ----------------------- Episodes & Memory (A3, A8) -----------------------

@dataclass
class Episode:
    t: int
    c: np.ndarray            # category key used
    z_pre: np.ndarray        # pre-action affect [v; m; a; d] (vector)
    h: np.ndarray            # policy hints stored that drove the step
    z_post: np.ndarray       # post-action affect
    succ: float              # success in [0,1]

class EpisodicMemory:
    """Append-only, Top-K retrieval. Returns memory affect and memory policy-hint aggregates."""
    def __init__(self, K: int = 5, max_size: int = 20000, seed: int = 0):
        self.K = int(K)
        self.max_size = int(max_size)
        self.episodes: List[Episode] = []
        self.rng = random.Random(seed)

    def _similarities(self, c_t: np.ndarray) -> np.ndarray:
        if len(self.episodes) == 0:
            return np.zeros(0)
        # Cosine on L2-normalized category vectors (A.3.3)
        C = np.stack([e.c for e in self.episodes], axis=0)
        c_unit = l2_normalize(c_t)
        C_unit = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        sims = (C_unit @ c_unit)
        return sims

    def retrieve(self, c_t: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return (z_mem, r_den, h_mem). If memory empty, zeros."""
        if len(self.episodes) == 0:
            return np.zeros_like(c_t[:0]), 0.0, np.zeros(4)  # will be replaced by correct shapes by caller
        sims = self._similarities(c_t)
        K = min(self.K, len(self.episodes))
        if K <= 0:
            return np.zeros_like(c_t[:0]), 0.0, np.zeros(4)

        idx = np.argpartition(-sims, K-1)[:K]
        s = sims[idx]
        # retrieval softmax over similarity (temperature-like scale)
        # keep it peaky but safe:
        w = softmax_stable(s.astype(float), tau=0.1)

        # Aggregate affect tags and policy hints with success weighting for hints
        # We'll infer dimensions from first episode
        zdim = self.episodes[idx[0]].z_pre.shape[0]
        hdim = self.episodes[idx[0]].h.shape[0]
        z_mem = np.zeros(zdim, dtype=float)
        h_num = np.zeros(hdim, dtype=float)
        h_den = 1e-12

        for wi, j in zip(w, idx):
            ep = self.episodes[j]
            # Affect tag as convex blend of pre/post (alpha in [0,1]); choose alpha=0.5
            z_tag = 0.5 * ep.z_pre + 0.5 * ep.z_post
            z_mem += wi * z_tag
            # Utility-weighted hint aggregation (Eq. 8 in main text)
            h_num += wi * ep.succ * ep.h
            h_den += wi * ep.succ

        h_mem = h_num / h_den if h_den > 1e-9 else np.zeros(hdim, dtype=float)
        return z_mem, float(h_den), h_mem

    def append(self, ep: Episode):
        self.episodes.append(ep)
        if len(self.episodes) > self.max_size:
            # simple FIFO trim
            cut = len(self.episodes) - self.max_size
            self.episodes = self.episodes[cut:]

# ----------------------- Categories & Need appraisal (A1, A2) -----------------------

class CategoryExtractor:
    """Example: multi-attribute category from local density and nearest distance."""
    def __init__(self, cfg: Dict):
        self.C = int(cfg.get("num_category_dims", 4))
        # knobs for simple density/spacing descriptors
        self.r_att = float(cfg.get("R_att", 10.0))
        self.r_rep = float(cfg.get("R_rep", 1.5))
        self.density_norm = float(cfg.get("density_norm", 5.0))
        self.minmax_eps = 1e-9

    def categorize(self, obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        obs: dict with keys: peers: List[(r,phi)], speed, heading, etc.
        Returns: c_t ∈ [0,1]^C, y_t (optional params, here nearest vector).
        """
        peers = obs["peers"]  # list of (r, phi)
        if len(peers) == 0:
            near = 0.0
            dmin = 1e9
        else:
            dmin = min(r for (r, _) in peers)
            near = sum(clip01(1.0 - r/self.r_att) for (r, _) in peers)

        # Simple 4D category: [near-density, inv-dmin, occupancy(rep), bias]
        rep = sum(clip01(1.0 - r/self.r_rep) for (r, _) in peers)
        c = np.array([
            clip01(near / (self.density_norm + self.minmax_eps)),
            clip01(1.0 / (dmin + 1e-3) if np.isfinite(dmin) else 0.0),  # larger if very close neighbor
            clip01(rep),  # strong if invasion of personal space
            0.5  # bias for separability
        ], dtype=float)
        c = c[:self.C] if self.C <= 4 else np.pad(c, (0, self.C-4), constant_values=0.0)
        # parameters for templates: pass peers along (r,phi)
        y = np.array([dmin], dtype=float)
        return c, y

class NeedAppraisal:
    """Compute needs -> drives -> affect (dimensional)."""
    def __init__(self, cfg: Dict):
        # two needs example: affiliation (n1), independence (n2)
        targ = cfg.get("need_targets", [0.7, 0.6])
        self.n_target = np.array(targ, dtype=float)
        self.alpha = np.array(cfg.get("drive_sensitivity", [2.0, 2.0]), dtype=float)  # per-need
        self.lambda_v = np.array(cfg.get("valence_lambda", [1.0, 0.2]), dtype=float)  # map drive->valence sign
        self.base_arousal = float(cfg.get("base_arousal", 0.5))

    def assess_needs(self, c_t: np.ndarray, obs: Dict) -> np.ndarray:
        """Example mapping obs-> needs in [0,1]. Keep it simple and bounded."""
        peers = obs["peers"]
        # reuse extractor's notions: near mid-range density ~ affiliation; repulsion ~ independence hit
        # We re-derive with same constants as CategoryExtractor for clarity (or pass them through).
        R_att, R_rep = obs.get("R_att", 10.0), obs.get("R_rep", 1.5)
        near = sum(clip01(1.0 - r/R_att) for (r, _) in peers)
        rep = sum(clip01(1.0 - r/R_rep) for (r, _) in peers)

        n1 = clip01(near / 5.0)              # affiliation grows with mid-range contact
        n2 = clip01(1.0 - (2.0*near + 1.0*rep))  # independence decreases with occupancy (tuned)
        n = np.array([n1, n2], dtype=float)
        return n

    def affect_from_needs(self, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        # drives ∈ [-1,1]
        d = np.tanh(self.alpha * (self.n_target - n))
        # valence: v = - lambda ⊙ d  (so deficit -> negative valence if lambda>0)
        v = - self.lambda_v * d
        v = np.array([clip11(vi) for vi in v], dtype=float)
        # magnitude = |d|
        m = np.abs(d)
        # arousal increases with base + magnitude norm
        a = clip01(self.base_arousal + 0.5 * float(np.linalg.norm(m, ord=1)) / (len(m)+1e-9))
        return v, m, a, d

# ----------------------- Policy→Action templates (A5) -----------------------

class PolicyTemplates:
    """
    Compute per-policy action scores on a discrete heading grid plus pause.
    Policies: Seek, Avoid, Explore, Rest
    """
    def __init__(self, cfg: Dict):
        self.R_att = float(cfg.get("R_att", 10.0))
        self.R_rep = float(cfg.get("R_rep", 1.5))
        self.L_att = float(cfg.get("L_att", cfg.get("R_att", 10.0)))
        self.L_rep = float(cfg.get("L_rep", cfg.get("R_rep", 1.5)))
        self.kappa = float(cfg.get("kappa", 6.0))
        self.kappa_o = float(cfg.get("kappa_o", 4.0))
        self.M = int(cfg.get("num_headings", 36))
        # Heading grid
        self.thetas = np.linspace(-math.pi, math.pi, self.M, endpoint=False)
        # One pause action at index M
        self.actions = list(range(self.M)) + ["PAUSE"]

    def _circ_mean(self, weights: List[float], phis: List[float]) -> Tuple[Optional[float], float]:
        if len(weights) == 0:
            return None, 0.0
        Sx = Sy = A = 0.0
        for w, phi in zip(weights, phis):
            w = max(0.0, float(w))
            Sx += w * math.cos(phi)
            Sy += w * math.sin(phi)
            A += w
        if A <= 1e-9:
            return None, 0.0
        rho = math.sqrt(Sx*Sx + Sy*Sy) / (A + 1e-12)
        if rho <= 1e-9:
            return None, 0.0
        phi_star = math.atan2(Sy, Sx)
        return phi_star, float(clip01(rho))

    def score_templates(self, obs: Dict) -> Dict[str, np.ndarray]:
        peers = obs["peers"]
        # attraction/repulsion weights
        L_att = float(getattr(self, "L_att", self.R_att))  # default to R_att if not set
        L_rep = float(getattr(self, "L_rep", self.R_rep))
        a = [math.exp(-r / max(1e-6, L_att)) for (r, _) in peers]   # attraction weights ∈ (0,1]
        b = [math.exp(-r / max(1e-6, L_rep)) for (r, _) in peers]   # repulsion weights ∈ (0,1]      
        phis = [phi for (_, phi) in peers]

        # Seek
        seek_dir, seek_rho = self._circ_mean(a, phis)

        # Avoid: move opposite to repulsive resultant
        avoid_dir, avoid_rho = self._circ_mean(b, phis)
        if avoid_dir is not None:
            avoid_dir = wrap_angle(avoid_dir + math.pi)  # opposite

        # --- Explore: smooth free-space vector field (works even when alone) ---
        # kernel width for density; larger => smoother, less noisy
        sigma = float(getattr(self, "explore_sigma", 2.0))  # world units
        ux = uy = 0.0
        for (r, phi) in peers:
            # convert polar to local Cartesian unit vector
            dx, dy = math.cos(phi)*r, math.sin(phi)*r
            # Gaussian kernel density derivative points away from peers
            w = math.exp(- (r*r) / (2.0*sigma*sigma))
            if r > 1e-6:
                ux += w * (dx / r)
                uy += w * (dy / r)
        # if no peers: explore toward a slowly rotating global direction (weak bias)
        if len(peers) == 0 and (ux*ux + uy*uy) < 1e-9:
            phase = float(getattr(self, "heading_phase", 0.03))
            ux, uy = math.cos(phase), math.sin(phase)

        exp_dir = math.atan2(uy, ux)
        exp_mag = math.hypot(ux, uy)
        exp_rho = clip01(exp_mag)    # cap to [0,1]

        # Fill per-policy scores for headings and pause
        M = len(self.thetas)
        scores = {
            "Seek": np.zeros(M+1),   # +1 for pause
            "Avoid": np.zeros(M+1),
            "Explore": np.zeros(M+1),
            "Rest": np.zeros(M+1),
        }
        for i, th in enumerate(self.thetas):
            scores["Seek"][i]   = (seek_rho * vm_bump(ang_diff(th, seek_dir), self.kappa)) if seek_dir is not None else 0.0
            scores["Avoid"][i]  = (avoid_rho * vm_bump(ang_diff(th, avoid_dir), self.kappa)) if avoid_dir is not None else 0.0
            scores["Explore"][i]= (exp_rho   * vm_bump(ang_diff(th,  exp_dir), self.kappa)) if exp_dir  is not None else 0.0
            scores["Rest"][i]   = 0.0
        # Pause
        scores["Seek"]  [-1] = 0.0
        scores["Avoid"] [-1] = 0.0
        scores["Explore"][-1] = 0.0
        scores["Rest"]  [-1] = 1.0

        # Small floor for EXPLORE headings so movement can beat PAUSE when field is flat
        explore_floor = float(getattr(self, "explore_floor", 0.15))
        scores["Explore"][:-1] = np.maximum(scores["Explore"][:-1], explore_floor)


        if seek_dir is None and avoid_dir is None and exp_dir is None:
             # Uniform weak preference to move (prevents permanent pausing)
            scores = {
                      "Seek":    np.full(self.M+1, 0.0),
                      "Avoid":   np.full(self.M+1, 0.0),
                      "Explore": np.concatenate([np.full(self.M, 0.3), [0.0]]),
                      "Rest":    np.concatenate([np.zeros(self.M), [0.1]]),
                     }

   
        return scores

# ----------------------- Controller (A1–A8) -----------------------

class Controller:
    """Single-actor controller implementing A1–A8 with local memory."""
    def __init__(self, cfg: Dict, category: CategoryExtractor, needs: NeedAppraisal, templates: PolicyTemplates):
        self.cfg = cfg
        self.category = category
        self.needs = needs
        self.templates = templates
        self.memory = EpisodicMemory(K=int(cfg.get("K", 5)),
                                     max_size=int(cfg.get("memory_max", 50000)),
                                     seed=int(cfg.get("seed", 0)))
        # Affect→policy matrices (bounded and normalized)
        # 4 policies: [Seek, Avoid, Explore, Rest]
        self.policies = ["Seek", "Avoid", "Explore", "Rest"]
        # Need→policy (H_n) for 2 needs
        self.H_n = np.array(cfg.get("H_n", [
            [ 1.0, -0.3],   # Seek up with affiliation deficit, down with independence deficit
            [-0.3,  1.0],   # Avoid up with independence deficit, down with affiliation deficit
            [ 0.2,  0.2],   # Explore mild default
            [-0.6, -0.6],   # Rest when neither drive is strong
        ]), dtype=float)
        # Affect→policy from [v*m (k channels), arousal', drives] concatenation
        # Here k=2; drives=2; a' in [-1,1]
        # Affect→policy from [a', d1, d2] (three inputs)
        self.H_base = np.array([
            [ 0.8,  0.2, -0.2],   # Seek
            [-0.2,  0.3,  0.2],   # Avoid
            [ 0.1,  0.1,  0.1],   # Explore
            [-0.4, -0.2, -0.1],   # Rest
        ], dtype=float)

        self.H_vm = np.array(cfg.get("H_vm", [
            [-0.5,  0.0],   # Seek suppressed by neg v*m on lonely/crowded
            [ 0.2,  0.3],   # Avoid boosted by neg v*m on crowded
            [ 0.2,  0.2],   # Explore mild
            [-0.2, -0.2],   # Rest mild
        ]), dtype=float)

        # Fusion alphas and temperatures
        self.alpha_need = float(cfg.get("alpha_need", 0.4))
        self.alpha_mem  = float(cfg.get("alpha_mem",  0.3))
        self.alpha_aff  = float(cfg.get("alpha_aff",  0.3))

        self.tau_pi_lo  = float(cfg.get("tau_policy_low",  0.4))  # high arousal → low temp
        self.tau_pi_hi  = float(cfg.get("tau_policy_high", 1.2))  # low arousal → high temp
        self.tau_u_lo   = float(cfg.get("tau_action_low",  0.4))
        self.tau_u_hi   = float(cfg.get("tau_action_high", 1.2))

        self.last_post_affect = None  # optional input to fusion if desired

   
    # ---------- Core step ----------
    def step(self, t: int, obs: Dict) -> Dict:
        # A1: categorize
        c_t, y_t = self.category.categorize(obs)
        # A2: needs -> affect
        n_t = self.needs.assess_needs(c_t, obs)
        v_n, m_n, a_n, d_n = self.needs.affect_from_needs(n_t)
        # Pack z^need = [v; m; a; d]
        z_need = np.concatenate([v_n, m_n, np.array([a_n], dtype=float), d_n], axis=0)

        # A3: memory retrieve
        # shape bookkeeping for empty memory
        zdim = z_need.shape[0]
        hdim = len(self.policies)
        if len(self.memory.episodes) == 0:
            z_mem = np.zeros(zdim, dtype=float)
            h_mem = np.zeros(hdim, dtype=float)
        else:
            z_mem, _, h_mem = self.memory.retrieve(c_t)

        # A4: fuse affect (constant fusion here, can be gated variants M7.b/c)
        beta0 = float(self.cfg.get("affect_fusion_beta0", 0.7))
        z = beta0 * z_need + (1.0 - beta0) * z_mem

        # Build affect-derived policy hints:  combine base[a', drives] and vm[v*m]
        k = len(v_n)  # valence channels
        vm = v_n * m_n                    # (k,)
        a_prime = 2.0 * z[k*2] - 1.0      # rescale arousal from [0,1]→[-1,1]
        drives = d_n
        # X_base = [a', drives...]
        X_base = np.concatenate([[a_prime], drives], axis=0)  # shape 1+|drives|
        # Normalize matrix outputs by input-dim to keep in [-1,1] (A.3.1)
        # h_aff = 0.5 * ( (self.H_base @ X_base) / max(1.0, X_base.shape[0]) + (self.H_vm @ vm) / max(1.0, vm.shape[0]) )
        # Build affect-derived policy hints
        h_aff = 0.5 * (
                          bounded_linear(self.H_base, X_base) +
                          bounded_linear(self.H_vm,   vm)
                      )

        # Need→policy hints (normalized)
        h_need = (self.H_n @ d_n) / max(1.0, d_n.shape[0]) * 0.5

        # Fuse hints
        h = self.alpha_need * h_need + self.alpha_mem * h_mem + self.alpha_aff * h_aff
        # Policy softmax with arousal-dependent temperature
        tau_pi = self._interp_tau(z[k*2])  # tau from arousal (index of a)
        q = softmax_stable(h, tau=tau_pi)
        # A5: policy→action scores, then convex mix
        scores = self.templates.score_templates(obs)  # dict(policy)->(M+1,)
        M1 = len(scores["Seek"])  # headings + pause
        s_u = np.zeros(M1, dtype=float)
        for i, p in enumerate(self.policies):
            s_u += q[i] * scores[p]
        # A6: sample or take argmax; we do argmax by default for determinism
        tau_u = self._interp_tau(z[k*2])  # could use a different mapping
 
        # grab current heading from obs (World2D already provides it)
        cur_heading = float(obs.get("heading", 0.0))

        thetas = self.templates.thetas              # shape [M] for heading actions (no pause)
        # angular deltas to current heading
        deltas = np.array([ang_diff(th, cur_heading) for th in thetas], dtype=float)

        # hyperparameters from cfg with safe defaults
        persist_w   = float(self.cfg.get("persist_weight", 0.25))   # bonus weight
        persist_k   = float(self.cfg.get("persist_kappa", 8.0))     # narrowness of persistence bump
        turn_cost_w = float(self.cfg.get("turn_cost_weight", 0.06)) # linear turn penalty

        # +persistence bonus, -turn cost (only on heading actions, not on pause)
        persist_bonus = np.array([vm_bump(d, persist_k) for d in deltas], dtype=float) * persist_w
        turn_cost     = np.abs(deltas) * turn_cost_w

        # ---- REST/PAUSE gating: suppress pausing when arousal/drives are high ----
        drive_norm = float(np.linalg.norm(d_n, ord=1)) / (len(d_n) + 1e-12)   # ∈ [0,1]
        ar = float(z[k*2])                                                    # arousal ∈ [0,1]

        rest_ar_gate  = float(self.cfg.get("rest_arousal_gate", 0.8))         # how much arousal suppresses pause
        rest_drv_gate = float(self.cfg.get("rest_drive_gate",   0.5))         # how much drives suppress pause
        pause_bias    = float(self.cfg.get("pause_bias",       -0.05))        # constant bias against pause

        # multiplicative gate in [0.1, 1.0] (never fully zero to keep degenerate cases safe)
        rest_gate = 1.0 - rest_ar_gate*ar - rest_drv_gate*drive_norm
        rest_gate = max(0.1, min(1.0, rest_gate))

        s_u[-1] = s_u[-1] * rest_gate + pause_bias
        

        # Convert to probabilities for sampling (optional):
        p_u = softmax_stable(s_u, tau=tau_u)
        # Deterministic action:
        #a_idx = int(np.argmax(p_u))
        # Sample the action:
        # a_idx = int(np.random.choice(len(p_u), p=p_u))


        # policy confidence (low entropy => confident)
        eps = 1e-12
        H = -np.sum(q * np.log(q + eps))
        H_max = math.log(len(q))
        conf_policy = 1.0 - (H / (H_max + eps))   # in [0,1]

        # action peakiness (how much the best action beats the median)
        a_best = float(np.max(s_u))
        a_med  = float(np.median(s_u))
        peak = max(0.0, a_best - a_med)

        # thresholds (configurable)
        H_thresh   = float(self.cfg.get("entropy_threshold", 0.35))   # lower = stricter
        peak_thresh= float(self.cfg.get("peak_threshold", 0.08))      # higher = stricter

        if conf_policy >= (1.0 - H_thresh) or peak >= peak_thresh:
            # confident: take deterministic best
            a_idx = int(np.argmax(s_u))
        else:
            # uncertain: sample (explore)
            a_idx = int(np.random.choice(len(p_u), p=p_u))

        

        # Post-act “execution” handled by environment; we return selection + ingredients
        out = dict(
            c_t=c_t, z_need=z_need, z_mem=z_mem, z=z,
            h_need=h_need, h_mem=h_mem, h_aff=h_aff, h=h, q=q,
            action_index=a_idx, action_probs=p_u, action_scores=s_u
        )
        return out

    def _interp_tau(self, a: float) -> float:
        """Map arousal a∈[0,1] to temperature in [lo,hi], high arousal -> low tau."""
        a = clip01(float(a))
        return self.tau_pi_hi + (self.tau_pi_lo - self.tau_pi_hi) * a  # linear

    def write_episode(self, t: int, c_t: np.ndarray, z_pre: np.ndarray, h: np.ndarray, z_post: np.ndarray, succ: float):
        ep = Episode(t=t, c=l2_normalize(c_t), z_pre=z_pre.copy(), h=h.copy(),
                     z_post=z_post.copy(), succ=float(clip01(succ)))
        self.memory.append(ep)

# ----------------------- Simple 2D World -----------------------

@dataclass
class ActorState:
    x: float
    y: float
    theta: float  # heading
    speed: float

class World2D:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.dt = float(cfg.get("dt", 0.3))
        self.speed_move = float(cfg.get("speed", 0.6))
        self.speed_pause = 0.0
        self.pos_noise_std = float(cfg.get("pos_noise_std", 0.0))
        self.rng = random.Random(int(cfg.get("seed", 0)))
        self.N = int(cfg.get("num_actors", 8))
        # Init placement: circle or random
        layout = str(cfg.get("init_layout", "circle")).lower()  # "circle" or "random"
        radius = float(cfg.get("init_radius", 6.0))
        jitter = float(cfg.get("init_jitter", 0.0))              # stddev positional noise
        box = float(cfg.get("init_box", 10.0))                   # half-width for random box
        self.actors: List[ActorState] = []

        if layout == "random":
            # Uniform in a square [-box, box]^2
            for _ in range(self.N):
                x = self.rng.uniform(-box, box)
                y = self.rng.uniform(-box, box)
                th = self.rng.uniform(-math.pi, math.pi)
                self.actors.append(ActorState(x=x, y=y, theta=th, speed=self.speed_move))
        else:
        # Circle, but allow small jitter to break perfect symmetry
            for i in range(self.N):
                ang = 2*math.pi*i/self.N
                x = radius * math.cos(ang) + self.rng.gauss(0.0, jitter)
                y = radius * math.sin(ang) + self.rng.gauss(0.0, jitter)
                th = self.rng.uniform(-math.pi, math.pi)
                self.actors.append(ActorState(x=x, y=y, theta=th, speed=self.speed_move))
        # ---------------------------------------------------------------

        # Controllers (each private memory to stay E1–E4-friendly)
        cat = CategoryExtractor(cfg)
        needs = NeedAppraisal(cfg)
        temps = PolicyTemplates(cfg)
        self.controllers = [Controller(cfg, cat, needs, temps) for _ in range(self.N)]

        # control of heading interpretation and small exploration jitter
        self.absolute_actions = bool(cfg.get('absolute_actions', True))
        self.heading_jitter_std = float(cfg.get('heading_jitter_std', 0.0))


        # Logging
        self.traj = [[] for _ in range(self.N)]         # (x,y)
        self.arousal = [[] for _ in range(self.N)]
        self.policy_mix = [[] for _ in range(self.N)]   # store q vectors
        self.actions = [[] for _ in range(self.N)]      # action index each step

    def observe(self, i: int) -> Dict:
        """Build observation for actor i. World-frame bearings if absolute_actions, else egocentric."""
        me = self.actors[i]
        peers = []
        for j, other in enumerate(self.actors):
            if j == i: continue
            dx = other.x - me.x
            dy = other.y - me.y
            r = math.hypot(dx, dy)
            if bool(self.absolute_actions):
                # world-frame bearing
                phi = wrap_angle(math.atan2(dy, dx))
            else:
                # egocentric bearing
                phi = wrap_angle(math.atan2(dy, dx) - me.theta)
            peers.append((r, phi))
        # sort by distance (optional)
        peers.sort(key=lambda t: t[0])
        obs = dict(
            peers=peers,
            heading=me.theta,
            speed=me.speed,
            R_att=self.cfg.get("R_att", 10.0),
            R_rep=self.cfg.get("R_rep", 1.5),
        )
        return obs

    def step(self, t: int):
        """One synchronous step: pick actions from snapshot, then update positions, then reappraise/write episodes."""
        # phase 1: perceive & decide
        decisions = []
        for i in range(self.N):
            obs = self.observe(i)
            out = self.controllers[i].step(t, obs)
            decisions.append((obs, out))

        # phase 2: execute decisions
        # Convert action index into (heading, speed)
        for i in range(self.N):
            st = self.actors[i]
            _, out = decisions[i]
            a_idx = out["action_index"]
            M = len(self.controllers[i].templates.thetas)
            if a_idx < M:
                chosen = self.controllers[i].templates.thetas[a_idx]
                if bool(self.absolute_actions):
                    # interpret chosen as absolute world heading, but smooth toward it
                    gain = float(self.cfg.get("heading_smooth_gain", 0.25))  # 0..1
                    st.theta = slerp_angle(st.theta, chosen, gain=gain)
                else:
                    # relative increment, also smoothed
                    gain = float(self.cfg.get("heading_smooth_gain", 0.25))
                    st.theta = slerp_angle(st.theta, wrap_angle(st.theta + chosen), gain=gain)

                # optional small jitter to break perfect symmetries / limit cycles
                if self.heading_jitter_std > 0.0:
                    st.theta = wrap_angle(st.theta + np.random.normal(0.0, self.heading_jitter_std))
                speed_conf_gain = float(self.cfg.get("speed_conf_gain", 0.5))  # 0..1
                # reuse the policy entropy we already computed: pass it out in `out` or recompute here; easiest is to pass
                conf = 1.0 - ( -np.sum(out["q"] * np.log(out["q"] + 1e-12)) / math.log(len(out["q"])) )
                st.speed = max(0.0, self.speed_move * (1.0 - speed_conf_gain*(1.0 - conf)))

            else:
                # pause
                st.speed = self.speed_pause

        # apply kinematics
        for st in self.actors:
            st.x += st.speed * math.cos(st.theta) * self.dt
            st.y += st.speed * math.sin(st.theta) * self.dt
            # apply kinematics
            if self.pos_noise_std > 0.0:
                st.x += np.random.normal(0.0, self.pos_noise_std)
                st.y += np.random.normal(0.0, self.pos_noise_std)


        # phase 3: reappraise success & write episodes
        for i in range(self.N):
            obs_after = self.observe(i)
            c_post, _ = self.controllers[i].category.categorize(obs_after)
            n_post = self.controllers[i].needs.assess_needs(c_post, obs_after)
            v_p, m_p, a_p, d_p = self.controllers[i].needs.affect_from_needs(n_post)
            z_post = np.concatenate([v_p, m_p, np.array([a_p], dtype=float), d_p], axis=0)

            # Get pre stored pieces from decision
            c_pre = decisions[i][1]["c_t"]
            z_pre = decisions[i][1]["z"]
            h     = decisions[i][1]["h"]

            # success: mix of drive reduction + hedonic improvement (bounded)
            d_pre = z_pre[-len(d_p):]
            dpos = np.linalg.norm(d_pre, ord=1)
            dpos2= np.linalg.norm(d_p, ord=1)
            succ_drive = 0.5 + 0.5 * (dpos - dpos2) / max(dpos, dpos2, 1e-6)
            vm_pre  = decisions[i][1]["z"][:len(v_p)] * decisions[i][1]["z"][len(v_p):2*len(v_p)]
            vm_post = v_p * m_p
            num = float(np.sum(vm_post - vm_pre))
            den = 4.0 * max(np.linalg.norm(vm_post, 1), np.linalg.norm(vm_pre, 1), 1e-6)
            succ_hed  = 0.5 + 0.5 * (num / den)
            w = 0.5
            succ = clip01(w * succ_drive + (1.0 - w) * succ_hed)

            self.controllers[i].write_episode(t, c_pre, z_pre, h, z_post, succ)

            # logs
            me = self.actors[i]
            self.traj[i].append((me.x, me.y))
            self.arousal[i].append(z_post[ len(v_p)*2 ])  # the a component
            self.policy_mix[i].append(decisions[i][1]["q"])
            self.actions[i].append(decisions[i][1]["action_index"])

            if t < 5:
              q_stack = np.stack([d[1]["q"] for d in decisions], axis=0)  # [N,4]
              print(f"t={t}: mean q = {q_stack.mean(0)}  (Seek, Avoid, Explore, Rest)")

    def run(self, T: int):
        for t in range(T):
            self.step(t)

    # ----------------------- plotting -----------------------
    def plot_trajectories(self, out_png: str = "trajectories.png"):
        plt.figure(figsize=(8,8), dpi=120)
        for i in range(self.N):
            xs = [p[0] for p in self.traj[i]]
            ys = [p[1] for p in self.traj[i]]
            c = np.array(self.arousal[i]) if len(self.arousal[i]) else np.zeros(len(xs))
            # normalize color
            if len(c):
                c = (np.array(c) - np.min(c)) / (np.max(c) - np.min(c) + 1e-9)
            plt.scatter(xs, ys, s=6, c=c, cmap="viridis", alpha=0.9, label=None)
            if xs and ys:
                plt.plot(xs, ys, lw=0.8, alpha=0.6, color="gray")
        plt.axhline(0, color="k", lw=0.2); plt.axvline(0, color="k", lw=0.2)
        plt.title("Trajectories (color = arousal)")
        plt.xlabel("x"); plt.ylabel("y"); plt.gca().set_aspect("equal", "box")
        plt.tight_layout(); plt.savefig(out_png); plt.close()

    def plot_metrics(self, out_png: str = "metrics.png"):
        T = max(len(s) for s in self.arousal) if self.arousal else 0
        if T == 0: return

        mean_arousal = np.zeros(T)
        mean_nnd     = np.zeros(T)
        mean_q_seek  = np.zeros(T)
        mean_entropy = np.zeros(T)

        for t in range(T):
            # arousal
            A = [self.arousal[i][t] for i in range(self.N) if t < len(self.arousal[i])]
            if A: mean_arousal[t] = float(np.mean(A))

            # nearest-neighbor distance
            pts = [ self.traj[i][t] for i in range(self.N) if t < len(self.traj[i]) ]
            nnd = []
            for i,(xi,yi) in enumerate(pts):
                best = 1e9
                for j,(xj,yj) in enumerate(pts):
                    if i==j: continue
                    d = math.hypot(xj-xi, yj-yi)
                    best = min(best, d)
                if best < 1e9: nnd.append(best)
            mean_nnd[t] = float(np.mean(nnd)) if nnd else 0.0

            # true policy mix + entropy
            qs = [ self.policy_mix[i][t] for i in range(self.N) if t < len(self.policy_mix[i]) ]
            if qs:
                Q = np.stack(qs, axis=0)                    # [N,4] for Seek,Avoid,Explore,Rest
                mean_q_seek[t]  = float(np.mean(Q[:,0]))    # actual Seek prob
                # entropy of policy distribution (natural log)
                eps = 1e-12
                H = -np.sum(Q * np.log(Q+eps), axis=1)
                mean_entropy[t] = float(np.mean(H))

        fig, ax = plt.subplots(4,1, figsize=(10,9), dpi=120, sharex=True)
        ax[0].plot(mean_arousal, lw=1.5); ax[0].set_ylabel("Mean arousal")
        ax[1].plot(mean_nnd, lw=1.5);     ax[1].set_ylabel("Mean NND")
        ax[2].plot(mean_q_seek, lw=1.5);  ax[2].set_ylabel("Mean q(Seek)")
        ax[3].plot(mean_entropy, lw=1.5); ax[3].set_ylabel("Policy entropy"); ax[3].set_xlabel("step")
        fig.suptitle("Group metrics")
        fig.tight_layout(); plt.savefig(out_png); plt.close()
# ----------------------- CLI -----------------------

DEFAULT_YAML = """
seed: 1
num_actors: 12
dt: 0.3
speed: 0.7
init_radius: 8.0

# Category & templates
num_category_dims: 4
R_att: 10.0
R_rep: 1.5
num_headings: 36
kappa: 6.0
kappa_o: 4.0

# Needs & affect
need_targets: [0.7, 0.6]
drive_sensitivity: [2.0, 2.0]
valence_lambda: [1.0, 0.2]
base_arousal: 0.5

# Affect→policy (mix alphas)
alpha_need: 0.4
alpha_mem: 0.3
alpha_aff: 0.3
affect_fusion_beta0: 0.7

# Temperatures
tau_policy_low: 0.4
tau_policy_high: 1.1
tau_action_low: 0.5
tau_action_high: 1.2

# Memory
K: 5
memory_max: 20000

# Matrices (optional override)
H_n:
  - [ 1.0, -0.3]
  - [-0.3,  1.0]
  - [ 0.2,  0.2]
  - [-0.6, -0.6]

H_base:
  - [ 0.8,  0.2, -0.2, -0.3, 0.0]
  - [-0.2,  0.3,  0.2,  0.1, 0.0]
  - [ 0.1,  0.1,  0.1,  0.2, 0.0]
  - [-0.4, -0.2, -0.1, -0.3, 0.0]

H_vm:
  - [-0.5, 0.0]
  - [ 0.2, 0.3]
  - [ 0.2, 0.2]
  - [-0.2,-0.2]
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    if args.config is None:
        cfg = yaml.safe_load(DEFAULT_YAML)
    else:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    world = World2D(cfg)
    world.run(args.steps)

    os.makedirs(args.out_dir, exist_ok=True)
    world.plot_trajectories(os.path.join(args.out_dir, "trajectories.png"))
    world.plot_metrics(os.path.join(args.out_dir, "metrics.png"))
    print(f"Saved figures in {args.out_dir}")

if __name__ == "__main__":
    main()

