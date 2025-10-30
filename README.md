## Citation

If you use this repository, please cite the corresponding preprint:

> Borotschnig, H. (2025). *Synthetic Emotions and Consciousness: Exploring Architectural Boundaries*.  
> Preprint


To Run This Code (outputs trajectories.png and metrics.png in directory results)

python emo_controller.py --config config_social.yaml --steps 600 --out_dir results


## Note on Temporal Persistence and E3 Compliance

The implementation includes a **heading persistence mechanism** (`slerp_angle` with configurable gain) that smooths movement trajectories. Without this, agents exhibit jittering since each decision is purely reactive.

### Why This Doesn't Violate E3 (Minimal Temporal Binding)

Constraint E3 prohibits **autobiographical temporal binding** - the construction of narrative self-continuity across episodes. The heading persistence here is:

1. **Mechanical, not episodic**: Similar to physical momentum or inertia, not memory-based
2. **Single-step Markovian**: Only current heading influences next heading, no multi-step dependencies  
3. **Outside the emotion loop**: Part of the actuation/motor layer, not the affective control system
4. **No cross-episode integration**: Each step's persistence is independent, no accumulated history

This is analogous to:
- A ball's momentum when rolling (physical continuity)
- A thermostat's hysteresis (state persistence)
- An insect's movement momentum (behavioral continuity)

None of these constitute the consciousness-relevant temporal binding that E3 aims to exclude. The emotion controller still makes decisions based solely on current observations and retrieved episodes (via similarity), not on accumulated temporal narratives.

### Configuration

To experiment with different levels of persistence:
- `heading_smooth_gain`: 0.0 = no persistence (jittery), 1.0 = instant heading change
- `persist_weight`: bonus for maintaining current heading
- `turn_cost_weight`: penalty for large heading changes

Setting all these to 0 demonstrates the pure reactive behavior which leads to similar qualitative outcomes but jittering motion.
