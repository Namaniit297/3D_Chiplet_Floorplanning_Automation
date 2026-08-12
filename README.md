# 🔥 THERMAL-MARL: AI-Driven Thermal-Aware 3D Chiplet Floorplanning

> **MARL Meets AlphaChip: Thermal-Aware 3D Floorplanning for Heterogeneous Multi-Die SoCs**

[![Paper](https://img.shields.io/badge/Paper-Read%20Here-blue)](https://github.com/Namaniit297/FLAME_297)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![PPO](https://img.shields.io/badge/Algorithm-PPO-red)](https://arxiv.org/abs/1707.06347)
[![MARL](https://img.shields.io/badge/Paradigm-MARL-purple)](https://en.wikipedia.org/wiki/Multi-agent_reinforcement_learning)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Namaniit297/FLAME_297?style=social)](https://github.com/Namaniit297/FLAME_297)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Key Contributions](#-key-contributions)
- [Key Results at a Glance](#-key-results-at-a-glance)
- [System Architecture](#-system-architecture)
  - [Two-Phase Cyclic MARL](#two-phase-cyclic-marl)
  - [Phase I — Intra-Layer Floorplanning](#phase-i--intra-layer-floorplanning)
  - [Phase II — Inter-Layer Coordination](#phase-ii--inter-layer-coordination)
  - [Neural Architecture](#neural-architecture)
  - [Training Procedure](#training-procedure)
- [Thermal Modeling](#-thermal-modeling)
  - [PACT RC-Network Model](#pact-rc-network-model)
  - [Liquid Cooling Integration](#liquid-cooling-integration)
  - [Flow-Aware Placement Cost](#flow-aware-placement-cost)
- [State Space](#-state-space)
- [Action Space](#-action-space)
- [Reward Functions](#-reward-functions)
- [Hardware Accelerator](#-hardware-accelerator)
  - [Agent Processing Unit (APU)](#agent-processing-unit-apu)
  - [IERB — Intelligent Experience Relay Buffer](#ierb--intelligent-experience-relay-buffer)
  - [Hierarchical Memory System](#hierarchical-memory-system)
  - [Shared L2 Cache and NoC](#shared-l2-cache-and-noc)
- [Experimental Setup](#-experimental-setup)
- [Results](#-results)
  - [System-A](#system-a--dual-gpu--hbm--cpu-stack)
  - [System-B](#system-b--cpu--npu--dram--io--ivrs)
  - [GPU vs Custom APU](#gpu-vs-custom-apu)
- [Comparison with Prior Work](#-comparison-with-prior-work)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [References](#-references)
- [Citation](#-citation)
- [License](#-license)

---

## 🌐 Overview

**THERMAL-MARL** is the **first two-phase cyclic Multi-Agent Reinforcement Learning (MARL) framework** for **thermal-aware 3D chiplet floorplanning** in heterogeneous multi-die Systems-on-Chip (SoCs).

Modern AI accelerators — GPUs, NPUs, and custom ASICs — are increasingly built using **3D chiplet architectures** that vertically stack logic, memory, and analog dies. These stacked systems offer massive interconnect density and reduced wirelength, but introduce severe **thermal coupling between layers**, **TSV congestion**, and **cooling integration challenges** that classical Electronic Design Automation (EDA) tools simply cannot handle.

Existing reinforcement learning-based placement frameworks such as **Google's AlphaChip** operate exclusively in flat 2D environments. They have no mechanism for:
- Coordinating placement decisions across vertically stacked dies
- Reasoning about inter-layer heat propagation
- Jointly optimizing TSV insertion and liquid cooling pipe placement
- Adapting to heterogeneous technology nodes across dies

**THERMAL-MARL closes all of these gaps** in one unified, physically-grounded, hardware-accelerated framework.

### What THERMAL-MARL Does

```
Given:  A 3D stack of heterogeneous chiplet dies (logic, memory, analog, accelerators)
        with fixed floorplan outlines, power maps, technology parameters, and TSV budgets

Output: Thermally balanced macro placement across all dies
        + optimally placed TSVs
        + microchannel liquid cooling pipe layout
        + convergence 3.2× faster than GPU-based baselines
```

### Core Innovation

```
Horizontal Agents (one per die layer)
    └── Optimize local macro placement: minimize wirelength, congestion, thermal cost
    └── Receive real-time temperature maps from PACT RC thermal simulator
    └── Use GNN-encoded graph state + PPO policy updates

Vertical Coordination Agent (one global)
    └── Aggregates thermal + power maps from all layers
    └── Detects inter-layer hotspots and power density imbalances
    └── Triggers macro migration, TSV insertion, cooling pipe routing
    └── Feeds updated global state back to horizontal agents

Hardware: Each agent mapped to a dedicated APU core
    └── IERB: prioritized experience replay via TD-error + entropy scoring
    └── Shared L2 cache + NoC for inter-agent synchronization
    └── >55% faster inference, >59% lower energy vs NVIDIA A100
```

---

## 💡 Motivation

The exponential growth of large-scale AI models has pushed modern data centers to a **thermal crisis point**.

### Real-World Thermal Failures

| Incident | Consequence |
|---|---|
| NVIDIA Blackwell 72-GPU rack overheating | Delayed hyperscaler AI infrastructure timelines |
| Alibaba Cloud SIN11 heat-induced failure | Service outages across Asia-Pacific |
| Microsoft forced liquid cooling rollout | Major data center infrastructure overhaul |
| OVHcloud France fire | Catastrophic, unrecoverable data center loss |
| GPU node throttling in LLM clusters | 20–40% sustained performance degradation |

These failures share a **root architectural cause**: the 2D monolithic IC design paradigm cannot scale to the power densities demanded by modern AI chips.

### Why 3D-ICs Are Necessary — But Hard

3D Integrated Circuits (3D-ICs) offer the architectural solution:
- **TSV stacking**: Mature, high-bandwidth logic-memory integration
- **Monolithic 3D (M3D)**: Transistor-level granularity
- **Face-to-Face (F2F) bonding**: High vertical I/O density, reduced routing

But 3D integration creates new, unsolved floorplanning challenges:

| Challenge | Impact |
|---|---|
| Thermal coupling between layers | Heat trapped in middle dies, no natural escape path |
| Vertical TSV congestion | Restricts available routing channels, degrades signal integrity |
| Power asymmetry across tiers | Uneven thermal distribution, hotspot formation |
| Active cooling integration | Liquid channels must be co-designed with macro placement |
| Cross-tier dependency | Placement decision in die 1 thermally impacts die 3 |

**Existing EDA flows treat thermal, placement, and interconnect as isolated sequential stages** — producing slow convergence and thermally fragile layouts. THERMAL-MARL integrates all of these into a single, jointly-learned optimization loop.

---

## ✨ Key Contributions

### 1. 🤖 THERMAL-MARL: First 3D MARL Floorplanning Framework

The first two-phase cyclic MARL system for 3D chiplet floorplanning:
- **Intra-layer horizontal agents**: One dedicated RL agent per die layer, optimizing macro placement with thermal, congestion, and wirelength feedback
- **Inter-layer vertical agent**: Global coordination agent that resolves cross-tier thermal imbalances, triggers macro migration, and co-places TSVs and cooling pipes
- **Cyclic alternation**: Horizontal → vertical → horizontal loop continues until cost function stability threshold is reached

### 2. 🔗 Joint Macro + TSV + Liquid Cooling Co-Optimization

For the first time, **macro reassignment**, **TSV insertion**, and **microchannel liquid cooling pipe routing** are treated as a **single jointly-optimized problem** within the reinforcement learning loop — not post-processing afterthoughts.

The vertical agent's placement cost:

$$\text{Cost}(x, y, l) = \alpha \cdot \text{Congestion}(x, y, l) + \beta \cdot T(x, y, l)$$

- TSVs are placed in **cooler, less congested zones** to preserve routing channels
- Liquid cooling pipes are aligned to **high-thermal zones** to dissipate hotspots

### 3. 🌡️ Physics-Guided Thermal Modeling with Flow-Aware Placement

**PACT** (Parallel Thermal Simulator) is embedded directly into the MARL training loop. Every agent, at every timestep, receives:
- Real-time grid-based temperature maps per layer
- Fluid inlet/outlet temperature gradients
- Power dissipation vector updates

This enables agents to make **thermally informed placement decisions** rather than post-hoc thermal corrections.

Novel **flow-sensitive thermal placement metric**:

$$\text{ThermalCost}_i = \frac{P_i}{F_i} \cdot D(i,\ \text{CoolantSource})$$

| Symbol | Meaning |
|---|---|
| $P_i$ | Power dissipated by macro $i$ |
| $F_i$ | Coolant flow rate at macro $i$'s location |
| $D(i, \text{CoolantSource})$ | Distance from macro $i$ to nearest coolant inlet |

High-power macros are **pulled toward high-flow cooling regions**, directly minimizing peak temperatures.

### 4. 🧠 Technology-Aware Heterogeneous State Representation

Each agent receives **die-specific physical parameters** as part of its observation:
- Thermal conductivity of the current layer
- Propagation delay characteristics
- Power-performance trade-off profile
- Hard vs. soft macro classification (GPU cores restricted to lower tiers; stacked memory permitted in upper tiers)

This enables **heterogeneity-aware optimization** — agents do not treat all dies identically.

### 5. ⚡ Hardware-Mapped MARL Acceleration via APU

A purpose-built **Agent Processing Unit (APU)** architecture:
- Each MARL policy network is mapped to a **dedicated APU core**
- **IERB** (Intelligent Experience Relay Buffer) acts as a neural L1 cache, filtering transitions by TD-error + policy entropy
- **Shared L2 cache + NoC** enables low-latency inter-agent synchronization
- **8×8 PE array** with FP16 for parallel CNN inference
- **Sparsity-aware compute engine** skips zero-valued operations

Results: **>55% faster inference** and **>59% lower energy per rollout** compared to NVIDIA A100 GPU baseline.

---

## 📊 Key Results at a Glance

| Metric | Value |
|---|---|
| 🌡️ Peak Temperature Reduction | Up to **12.5%** vs. baselines |
| 🔗 Inter-Die Wirelength Reduction | **17.8%** vs. baselines |
| ⚡ Training Convergence Speed | **3.2× faster** than GPU baselines |
| 🖥️ APU Inference Time vs. A100 | **55%+ reduction** |
| 🔋 APU Energy per Rollout vs. A100 | **59%+ reduction** |
| 🌡️ Peak Temp Reduction vs. SA (System-A) | **5.4°C lower** |
| 📏 Wirelength Reduction vs. SA (System-A) | **11.7% shorter** |
| 🔗 TSV Count Reduction vs. SA (System-A) | **7.7% fewer** |
| ⏱️ Runtime Reduction vs. SA (System-A) | **18.4% faster** |
| 🌡️ Avg Temp Reduction vs. BO (System-B) | **3.7°C lower** |
| ⏱️ Runtime Reduction vs. BO (System-B) | **>25% faster** |

---

## 🏗️ System Architecture

### Two-Phase Cyclic MARL

```
┌─────────────────────────────────────────────────────────────────┐
│                    THERMAL-MARL FRAMEWORK                       │
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║             PHASE I: INTRA-LAYER (Horizontal)            ║  │
│  ║                                                          ║  │
│  ║  Layer 1 Agent (A1)   Layer 2 Agent (A2)  ...  AL Agent ║  │
│  ║  ┌─────────────┐      ┌─────────────┐      ┌──────────┐ ║  │
│  ║  │ GNN Encoder │      │ GNN Encoder │      │   GNN    │ ║  │
│  ║  │ Policy Head │      │ Policy Head │      │ Policy   │ ║  │
│  ║  │ Value Head  │      │ Value Head  │      │  Value   │ ║  │
│  ║  └──────┬──────┘      └──────┬──────┘      └────┬─────┘ ║  │
│  ║         │ Placement           │ Placement         │       ║  │
│  ║         ▼ Actions             ▼ Actions           ▼       ║  │
│  ╚═════════╪═══════════════════════════════════════╪════════╝  │
│            │   After k horizontal iterations       │            │
│            ▼                                       ▼            │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║          PHASE II: INTER-LAYER (Vertical Agent)          ║  │
│  ║                                                          ║  │
│  ║   Aggregates:  [T1, P1, pos1] ⊕ [T2, P2, pos2] ⊕ ...  ║  │
│  ║                        ↓                                 ║  │
│  ║            Global Stack State (S_global)                ║  │
│  ║                        ↓                                 ║  │
│  ║   Actions: Macro Migration + TSV Insertion +            ║  │
│  ║            Liquid Cooling Pipe Routing                   ║  │
│  ║                        ↓                                 ║  │
│  ║        Updated stack state → back to Phase I            ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│                                                                 │
│   Loop continues until cost function stability threshold met    │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase I — Intra-Layer Floorplanning

Each die layer $l \in \{1, 2, \ldots, L\}$ is assigned a dedicated RL agent $A_l$.

**Macro Set:** $M = \{m_i \mid i \in \mathbb{Z}^+,\ 1 \le i \le N_m\}$

Each macro $m_i$ is defined by:
- Width $w_i$, Height $h_i$, Area $a_i$, Power $p_i$
- Position $(x_i, y_i)$ in 2D plane, Die layer index $z_i$ where $1 \le z_i \le K$

**Macro Classification:**
- **Hard macros** (e.g., GPU cores): Restricted to lower tiers — power delivery and thermal constraints
- **Soft macros** (e.g., stacked memory): Permitted in upper layers — area-flexible

**Congestion Model:**

$$\text{Congestion}(i, j) = \frac{\text{Total routing demand in box } (i,j)}{\text{Routing capacity of box } (i,j)}$$

High congestion values discourage the RL agent from placing macros in those regions.

**Wirelength Estimation (HPWL):**

$$\text{HPWL} = \sum_{\text{net}_j} \left[ (x_{\max} - x_{\min}) + (y_{\max} - y_{\min}) \right]$$

---

### Phase II — Inter-Layer Coordination

After every $k$ horizontal iterations, the vertical agent $A_{\text{vert}}$ activates.

**Global Stack State:**

$$S^{\text{global}}_t = \bigoplus_{l=1}^{L} \left[ T_l(x, y, t),\ P_l(x, y),\ (x^l_i, y^l_i) \right]$$

**Macro Migration Decision:**

$$a^{\text{vert}}_t = \left\{ m_i : l \to l' \mid T_l(x^l_i, y^l_i, t) > \tau_T \ \wedge \ P_l(x^l_i, y^l_i) > \tau_P \right\}$$

Where $\tau_T$ and $\tau_P$ are thermal and power thresholds; $l'$ is the target layer selected to reduce cumulative violations.

**TSV + Cooling Pipe Placement Cost:**

$$\text{Cost}(x, y, l) = \alpha \cdot \text{Congestion}(x, y, l) + \beta \cdot T(x, y, l)$$

---

### Neural Architecture

Each agent uses a structured neural architecture:

```
Input: Matrix-based layer features
    (Thermal Map, Congestion Map, Power Map, Macro Occupancy, Cooling Proximity, Wirelength)
         ↓
Graph Construction: G = (V, E)
    Nodes V: macros
    Edges E: spatial proximity + thermal interdependency + TSV connections
         ↓
GNN Stack (3–5 layers)
    Node embedding update: neighbor aggregation modulated by edge attributes
    Output: contextual node embeddings h_v per macro
         ↓
    ┌────────────────┐        ┌───────────────┐
    │  Policy Head   │        │  Value Head   │
    │  MLP → softmax │        │  MLP → scalar │
    │  P(x,y,z|s)   │        │  V(s_t)       │
    └────────────────┘        └───────────────┘
         ↓
    Action: Place macro at (x, y, z)
```

**PPO Clipped Objective:**

$$\mathcal{L}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

**Value Function Loss:**

$$\mathcal{L}_{\text{value}} = \left( V(s_t) - R_t \right)^2$$

---

### Training Procedure

Training occurs in two phases:

**Phase A — Supervised Pretraining:**
- Dataset: $(s, \text{placement})$ pairs from known floorplans
- Loss: MSE between predicted and ground-truth rewards
- Purpose: Warm-start policy networks, accelerate convergence

**Phase B — MARL Fine-Tuning:**
- Algorithm: PPO with experience replay
- Reward signals: Wirelength + Thermal profile + TSV usage + Congestion
- Inter-agent coordination: Vertical agent feeds coordination signals back to horizontal agents after each global step

---

## 🌡️ Thermal Modeling

### PACT RC-Network Model

We embed **PACT** (Parallel Thermal Simulator) directly into the MARL training loop. PACT constructs a thermal RC netlist:
- **Resistors** → heat conduction paths
- **Capacitors** → thermal storage
- **Current sources** → power dissipation in chiplets
- **Voltage nodes** → temperature at each grid point

**Transient Thermal Equation:**

$$C \cdot \frac{dT}{dt} + G \cdot T = P$$

**Steady-State Solution:**

$$G \cdot T = P \quad \Rightarrow \quad T = G^{-1} \cdot P$$

Where:
- $C$ = Thermal capacitance matrix
- $G$ = Thermal conductance matrix (includes interlayer thermal resistances for vertical heat spread)
- $T$ = Temperature vector
- $P$ = Power dissipation vector

PACT executes via SPICE-based simulation. Output: **grid-based temperature maps** passed to each layer agent as part of the observation at every timestep.

---

### Liquid Cooling Integration

A fluidic cooling layer is inserted **between silicon tiers**. Each grid node in this layer is modeled with:

**Heat Transfer Coefficient (vertical and side walls):**

$$h_{f,\text{vertical}} = h_{f,\text{side}} = \frac{k_{\text{coolant}} \cdot \text{Nu}}{d_h}$$

Where:
- $k_{\text{coolant}}$ = Thermal conductivity of coolant
- $\text{Nu}$ = Nusselt number
- $d_h$ = Hydraulic diameter of microchannel

**Convective Fluid Behavior:**

$$J_{\text{conv}} = c_{\text{conv}} \cdot (T_{\text{in}} - T_{\text{out}})$$

Where $c_{\text{conv}}$ is the liquid convection coefficient and $T_{\text{in}}$, $T_{\text{out}}$ are adjacent fluid node temperatures.

---

### Flow-Aware Placement Cost

$$\text{ThermalCost}_i = \frac{P_i}{F_i} \cdot D(i,\ \text{CoolantSource})$$

| Symbol | Definition |
|---|---|
| $P_i$ | Power dissipated by macro $i$ |
| $F_i$ | Coolant flow rate at macro $i$'s grid location |
| $D(i, \text{CoolantSource})$ | Euclidean distance to nearest coolant inlet |

This metric **encourages high-power macros to be placed near high-flow cooling regions**, reducing peak temperature and improving thermal reliability throughout training.

---

## 📋 State Space

Each layer agent $A_l$ receives a matrix-based observation $s^l_t$ at timestep $t$:

| State Component | Description | Shape |
|---|---|---|
| **Macro Occupancy Map** | Grid with macro IDs in occupied cells, 0 elsewhere | $H \times W$ |
| **Thermal Map** | Real-time temperature distribution from PACT RC model | $H \times W$ |
| **Congestion Map** $C_l(x,y,t)$ | Routing congestion per grid cell, updated after each placement | $H \times W$ |
| **Liquid Cooling Distance** $L_l(x,y)$ | Binary grid indicating presence of liquid cooling paths | $H \times W$ |
| **Power Density Map** $P_l(x,y)$ | Power consumption per cell based on mapped macro | $H \times W$ |
| **Wirelength Map** $N_l(x,y)$ | Weighted connectivity per grid cell for wirelength gradient | $H \times W$ |
| **Fixed Boundary Constraints** | Legal placement zone grid (1 = valid, 0 = blocked) | $H \times W$ |
| **Technology Parameters** | Layer-specific: thermal conductivity, delay, power profile | Scalar vector |

The **vertical agent** receives the aggregated global stack state:

$$S^{\text{global}}_t = \bigoplus_{l=1}^{L} \left[ T_l(x,y,t),\ P_l(x,y),\ (x^l_i, y^l_i) \right]$$

---

## 🎮 Action Space

### Intra-Layer Actions (Horizontal Agents)

| Action | Description |
|---|---|
| **Swap_One_Axis** | Swap a pair of blocks along either the $X_t$ or $Y_t$ sequence to introduce localized perturbation |
| **Swap_Both_Axes** | Simultaneously swap a block pair in both $X_t$ and $Y_t$ to induce stronger local reordering |
| **Reposition_Block** | Move a selected block to a new position within sequence for flexible spatial rearrangement |
| **Rotate_Block** | Rotate a selected block by 90° to explore orientation-dependent effects on wirelength |

Actions are sampled from the policy distribution $\pi_\theta(a \mid s)$ produced by the GNN-based policy head.

### Inter-Layer Actions (Vertical Agent)

| Action | Trigger Condition |
|---|---|
| **Macro Migration** $m_i : l \to l'$ | $T_l(x^l_i, y^l_i, t) > \tau_T \wedge P_l(x^l_i, y^l_i) > \tau_P$ |
| **TSV Insertion** | Low congestion, low thermal zones — preserves routing density |
| **Liquid Cooling Pipe Routing** | High-thermal zones — maximizes convective heat removal |

---

## 🎯 Reward Functions

### Horizontal Agent Cumulative Reward

$$r_t = -\lambda_1 \cdot \max_{(x,y)} T(x,y,t) - \lambda_2 \cdot \text{HPWL}_t - \lambda_3 \cdot \sum_{(i,j)} \text{Congestion}_t(i,j) - \lambda_4 \cdot \sum_i \text{ThermalCost}^t_i$$

| Term | Role |
|---|---|
| $-\lambda_1 \cdot \max T(x,y,t)$ | Penalizes peak temperature hotspots |
| $-\lambda_2 \cdot \text{HPWL}_t$ | Penalizes excessive wirelength |
| $-\lambda_3 \cdot \sum \text{Congestion}$ | Penalizes routing congestion accumulation |
| $-\lambda_4 \cdot \sum \text{ThermalCost}_i$ | Penalizes thermally-unaware macro placement |

### Vertical Agent Scalar Reward

$$r^{\text{vert}}_t = -\max_{(x,y)} T(x,y,z) - \lambda_2 \cdot \text{VD}_t$$

Where $\text{VD}_t$ is **vertical density** — represents interconnect pressure due to TSV usage. This encourages better thermal and 3D distribution across the stack.

---

## ⚙️ Hardware Accelerator

### Agent Processing Unit (APU)

Each MARL agent is mapped to a dedicated **APU core** purpose-built for RL-driven chiplet optimization:

```
┌─────────────────────────────────────────────────┐
│                  APU Core                       │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │         8 × 8 PE Array (64 PEs)          │   │
│  │   FP16 Multiply-Accumulate Operations    │   │
│  │   Parallel CNN policy/value inference    │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────┐   ┌─────────────────────────┐  │
│  │ Compression │   │  Sparse Weight          │  │
│  │  Network    │   │  Transposer             │  │
│  │  Interface  │   │  (tile-based parallel)  │  │
│  └─────────────┘   └─────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  3-Tier Memory                            │  │
│  │  ├── PE-local buffer (operand reuse)      │  │
│  │  ├── Global memory (intra-agent sharing)  │  │
│  │  └── I/O cache buffers (minibatch stream) │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  IERB (Intelligent Experience Relay Buffer│  │
│  │  Neural L1 cache — TD-error + entropy     │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Key APU Components:**

| Component | Function |
|---|---|
| **Compression Network Interface** | Decompresses sparse models, minimizes memory bandwidth |
| **Sparse Weight Transposer** | Reorganizes weights for tile-based parallel PE broadcasting |
| **Input/Output Buffers** | Stores feature maps and activations for seamless data movement |
| **8×8 PE Array** | 64 FP16 MAC units for parallel policy/value network inference |
| **Sparsity-Aware Compute Engine** | Skips zero-valued operations — reduces energy and latency |
| **Post-Processing Unit** | Offloads activation + normalization tasks without CPU intervention |

---

### IERB — Intelligent Experience Relay Buffer

The IERB acts as a **neural L1 cache** for each APU. Rather than naive circular replay, it filters and retains transitions based on a **hybrid priority score**:

$$P_i = \lambda_1 |\delta_i| + \lambda_2 (1 - \text{PolicyAcc}_i) + \lambda_3 \cdot \text{SpatialImpact}_i$$

| Component | Meaning |
|---|---|
| $|\delta_i|$ | Temporal Difference (TD) error — surprise signal |
| $1 - \text{PolicyAcc}_i$ | Policy deviation — how much this transition challenges current policy |
| $\text{SpatialImpact}_i$ | Influence on thermal and spatial layout quality |

Critical, high-impact transitions are **prioritized for reuse**. Low-impact experiences are offloaded to the shared L2 buffer.

---

### Hierarchical Memory System

Transitions are stored across a **three-level memory hierarchy**:

| Level | Content | Purpose |
|---|---|---|
| **L1 (IERB)** | Top-K percentile high-priority samples | Frequent, fast GPU/APU access |
| **L2 (Shared Cache)** | Medium-priority experiences | Track evolving placement dynamics |
| **L3 (Archive)** | Low-priority or aged samples | Maintain diversity, prevent overfitting |

**Proportional Prioritization Sampling:**

$$P(k) = \frac{(P_k + \epsilon)^\alpha}{\sum_j (P_j + \epsilon)^\alpha}, \qquad w_k = \left( \frac{1}{N \cdot P(k)} \right)^\beta$$

Where:
- $N$ = Total memory size
- $\alpha$ = Prioritization sharpness
- $\beta$ = Importance-sampling exponent, annealed from $\beta_0 \to 1$

Final TD update is scaled by $w_k \cdot \delta_k$ for robust learning.

---

### Shared L2 Cache and NoC

- **Centralized L2 cache**: Accessible by all APUs simultaneously
- **High-bandwidth NoC**: Fast experience sharing across agents
- **Low-latency inter-agent synchronization**: Enables dynamic thermal coordination across dies during training
- **Architecture**: Emulates a decentralized training kernel — distributed, parallel, no central bottleneck

---

## 🔬 Experimental Setup

### Benchmarks

Two heterogeneous 3D-IC systems modeled after real-world AI datacenter designs:

**System-A: Dual GPU + HBM + CPU Stack**
- Inspired by NVIDIA DGX series architecture
- Components: GPU-1, GPU-2, CPU Mesh, GPU RAM, HBM, DRAM
- Challenge: Vertical data movement, thermal decoupling of stacked GPU dies, TSV congestion

**System-B: CPU + NPU + DRAM + I/O + IVRs**
- Inspired by Meta MTIA and AWS Trainium architectures
- Components: Mesh-networked CPU, Neural Processing Unit, Dedicated RAM, Stacked DRAM, Integrated Voltage Regulators, I/O Controller, AI Accelerator
- Challenge: Heterogeneous power densities, asymmetric latency requirements, multi-domain cooling

### Thermal Simulation
- **Base**: PACT compact RC thermal model
- **Extensions**: TSV thermal resistance + active microchannel liquid cooling
- Integrated into MARL agent observation-feedback loop for temperature-conscious learning

### Training Environment
- **Framework**: PyTorch
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Training Hardware**: AMD EPYC CPU (128 threads) + NVIDIA A100 GPU (Google Colab)
- **Inference Hardware**: NVIDIA A100 GPU + Custom RTL-based many-core APU simulator with L1 and shared L2 caches

### Baselines
- **Simulated Annealing (SA)**: Classical heuristic, sequence-pair based
- **Bayesian Optimization (BO)**: Surrogate-model guided placement

---

## 📈 Results

### System-A — Dual GPU + HBM + CPU Stack

| Method | Max Temp (°C) | Avg Temp (°C) | Wirelength (mm) | TSVs Used | Cooling Pipe Density | Runtime (s) |
|---|---|---|---|---|---|---|
| Simulated Annealing (SA) | 91.6 | 76.4 | 101.2 | 912 | Low | 932 |
| Bayesian Optimization (BO) | 88.9 | 74.5 | 94.8 | 980 | Medium | 1142 |
| **THERMAL-MARL (Ours)** | **86.2** | **70.8** | **89.3** | **842** | **High** | **1583** |

**System-A Floorplan Layout:**
- **Layer 1 (Bottom — Blue)**: CPU Mesh + DRAM Stack — memory and logic kept cool and accessible
- **Layer 2 (Middle — Sky Blue)**: HBM + GPU RAM — communication bridge interconnected via TSVs
- **Layer 3 (Top — Red)**: GPU-1 + GPU-2 — high-power dies with liquid cooling pipes positioned directly overhead

TSV vertical interconnects: GPU ↔ RAM, RAM ↔ HBM, HBM ↔ CPU/DRAM — optimized for data flow throughput.

---

### System-B — CPU + NPU + DRAM + I/O + IVRs

| Method | Max Temp (°C) | Avg Temp (°C) | Wirelength (mm) | TSVs Used | Cooling Pipe Density | Runtime (s) |
|---|---|---|---|---|---|---|
| Simulated Annealing (SA) | 93.8 | 78.6 | 109.4 | 1015 | Low | 932 |
| Bayesian Optimization (BO) | 90.4 | 75.8 | 103.2 | 968 | Medium | 1365 |
| **THERMAL-MARL (Ours)** | **87.2** | **72.1** | **96.7** | **905** | **High** | **1018** |

**System-B Floorplan Layout:**
- **Layer 1 (Bottom)**: CPU (mesh interconnect) + IVRs + Clock Generator + Stacked DRAM + I/O Controller — strategic placement minimizes communication latency and power delivery path length
- **Layer 2 (Top)**: NPU + Dedicated RAM — liquid cooling pipes positioned directly above NPU for direct thermal dissipation from the highest-power processing element

---

### GPU vs. Custom APU

| Platform | Training Time (Epochs) | Inference Time (s/episode) | Energy per Rollout (J) |
|---|---|---|---|
| NVIDIA A100 GPU | 842 | 0.216 | 7.84 |
| **Custom APU (Ours)** | **624** | **0.095** | **3.21** |

- **Training epochs reduced**: 842 → 624 (**25.9% fewer epochs**)
- **Inference time reduced**: 0.216s → 0.095s (**56% faster**)
- **Energy per rollout reduced**: 7.84J → 3.21J (**59% more energy-efficient**)

---

## 📊 Comparison with Prior Work

| Method | Thermal Modeling | TSV Opt. | Liquid Cooling | 3D Support | Cross-Tier Coordination |
|---|---|---|---|---|---|
| Simulated Annealing [2,3] | ❌ | ❌ | ❌ | ✅ | ❌ |
| Ant Colony + SA [4] | ❌ | ❌ | ❌ | ✅ | ❌ |
| AlphaChip [8] | ❌ | ❌ | ❌ | ❌ | ❌ |
| Q-Learning [9] | ❌ | ❌ | ❌ | ❌ | ❌ |
| GoodFloorplan [10] | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hypergraph MARL [11] | ❌ | ❌ | ❌ | ❌ | ❌ |
| **THERMAL-MARL (Ours)** | ✅ RC+PACT | ✅ Joint | ✅ Microchannel | ✅ Full 3D | ✅ Vertical Agent |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Namaniit297/FLAME_297.git
cd FLAME_297

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=1.12.0
torch-geometric>=2.1.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
networkx>=2.6.0
gym>=0.21.0
pyyaml>=6.0
tqdm>=4.62.0
```

---

## 🚀 Usage

### Quick Start — Run THERMAL-MARL on System-A

```bash
python train.py \
  --system system_a \
  --layers 3 \
  --macros 12 \
  --epochs 800 \
  --ppo_clip 0.2 \
  --thermal_lambda 1.0 \
  --wirelength_lambda 0.5 \
  --congestion_lambda 0.3 \
  --thermal_cost_lambda 0.8 \
  --cooling_enabled True \
  --tsv_budget 1000 \
  --output_dir ./results/system_a
```

### Run on System-B

```bash
python train.py \
  --system system_b \
  --layers 2 \
  --macros 8 \
  --epochs 800 \
  --thermal_lambda 1.2 \
  --output_dir ./results/system_b
```

### Evaluate a Trained Model

```bash
python evaluate.py \
  --checkpoint ./results/system_a/best_model.pt \
  --system system_a \
  --visualize True \
  --export_floorplan True
```

### Run Hardware Simulation (APU vs GPU)

```bash
python hardware_sim.py \
  --mode apu \
  --agents 3 \
  --l1_cache_size 512 \
  --l2_cache_size 2048 \
  --pe_array 8x8 \
  --benchmark system_a
```

---

## 📁 Project Structure

```
FLAME_297/
│
├── 📄 README.md                        # This file
├── 📄 requirements.txt                 # Python dependencies
├── 📄 LICENSE                          # MIT License
│
├── 📂 agents/
│   ├── horizontal_agent.py             # Intra-layer RL agent (Phase I)
│   ├── vertical_agent.py               # Inter-layer coordination agent (Phase II)
│   ├── base_agent.py                   # Shared agent base class
│   └── ppo.py                          # PPO algorithm implementation
│
├── 📂 environment/
│   ├── floorplan_env.py                # 3D chiplet floorplanning gym environment
│   ├── thermal_model.py                # PACT RC-network thermal simulator integration
│   ├── congestion.py                   # Routing congestion computation
│   ├── cooling.py                      # Microchannel liquid cooling model
│   └── wirelength.py                   # HPWL estimation
│
├── 📂 models/
│   ├── gnn_encoder.py                  # Graph Neural Network state encoder
│   ├── policy_head.py                  # Policy MLP head (outputs placement distribution)
│   ├── value_head.py                   # Value MLP head (estimates long-term reward)
│   └── vertical_coordinator.py         # Vertical agent pooling/attention module
│
├── 📂 memory/
│   ├── ierb.py                         # Intelligent Experience Relay Buffer
│   ├── prioritized_replay.py           # Hierarchical 3-level prioritized memory
│   └── shared_l2.py                    # Shared L2 cache simulation
│
├── 📂 hardware/
│   ├── apu_core.py                     # APU core simulation model
│   ├── pe_array.py                     # 8×8 Processing Element array
│   ├── noc.py                          # Network-on-Chip interconnect simulation
│   └── hardware_sim.py                 # Full hardware simulation runner
│
├── 📂 benchmarks/
│   ├── system_a/                       # Dual GPU + HBM + CPU benchmark files
│   │   ├── netlist.json
│   │   ├── power_map.npy
│   │   └── floorplan_config.yaml
│   └── system_b/                       # CPU + NPU + DRAM benchmark files
│       ├── netlist.json
│       ├── power_map.npy
│       └── floorplan_config.yaml
│
├── 📂 results/
│   ├── system_a/                       # Training logs, checkpoints, floorplan outputs
│   └── system_b/
│
├── 📂 visualization/
│   ├── plot_floorplan.py               # 3D floorplan visualization (per layer)
│   ├── plot_thermal.py                 # Thermal heatmap plots
│   └── plot_convergence.py             # Training reward convergence curves
│
├── train.py                            # Main training script
├── evaluate.py                         # Evaluation and visualization script
└── hardware_sim.py                     # Hardware benchmark runner
```

---

## 📚 References

```
[1]  Souri, Multiple Si layer ICs: Motivation, performance analysis, and design implications.
     DAC 2000, Los Angeles, CA, USA, pp. 213–220.

[2]  Frantz, I. 3D-IC floorplanning: Applying meta-optimization to improve performance.
     IEEE/IFIP VLSI System-on-Chip, 2011, pp. 404–409.

[3]  Chen, S.; Yoshimura, T. Multi-layer floorplanning for stacked ICs.
     Configuration Number, 2010, 43, 378–388.

[4]  Xu, Q. Combining the ant system algorithm and simulated annealing for 3D/2D
     fixed-outline floorplanning. Soft Computing, 2016, 40, 150–160.

[5]  Guler, A.; Jha, N.K. Hybrid monolithic 3-D IC floorplanner.
     IEEE Trans. VLSI Systems, 2018, 26, 1868–1880.

[6]  Zhu, H.Y. Floorplanning for 3D-IC with TSV co-design using simulated annealing.
     EMC/APEMC, Singapore, 2018, pp. 550–553.

[7]  Shanthi. Thermal Aware Floorplanner for Multi-Layer ICs with Fixed Outline Constraints.
     IEEE Conference on Communication, 2021, pp. 1–6.

[8]  Mirhoseini, A. et al. A graph placement methodology for fast chip design.
     Nature, 2021, 594(7862), pp. 207–212.

[9]  He, Z. et al. Learn to floorplan through acquisition of effective local search heuristics.
     IEEE ICCD 2020, Hartford, CT, USA, pp. 324–331.

[10] Xu, Q. et al. GoodFloorplan: Graph Convolutional Network and Reinforcement
     Learning-Based Floorplanning. IEEE Trans. CAD, 2021, 41, 3492–3502.

[11] Amini, M. et al. Generalizable Floorplanner through Corner Block List Representation
     and Hypergraph Embedding. ACM SIGKDD, 2022, pp. 2692–2702.

[12] Kadambarajan, J.P. et al. GPU Implementation of Thermal Aware 3D IC Floorplanning.
     IJCISIM, 2021, vol. 13.

[13] Schaul, T. et al. Prioritized Experience Replay. arXiv:1511.05952, 2015.

[14] An, S. A 8.81 TFLOPS/W Deep-Reinforcement-Learning Accelerator with Delta-Based
     Weight Sharing and Block-Mantissa Reconfigurable PE Array. IEEE ITSC, 2024,
     vol. 71, no. 5, pp. 2529–2533.

[15] Murata, H. et al. VLSI module placement based on rectangle-packing by the
     sequence-pair. IEEE Trans. CAD, 1996, 15, 1518–1524.

[16] Jang, H.B. et al. The impact of liquid cooling on 3D multi-core processors.
     IEEE ICCD, 2009.

[17] Monchiero, M. et al. Design space exploration for multicore architectures:
     a power/performance/thermal view. ICS 2006, pp. 177–186.

[18] Yuan, Z. PACT: An Extensible Parallel Thermal Simulator for Emerging Integration
     and Cooling Technologies. ICCAD, 2021.
```

---

## 📝 Citation

If you use THERMAL-MARL in your research, please cite:

```bibtex
@article{thermalmarl2024,
  title     = {MARL Meets AlphaChip: Thermal-Aware 3D Floorplanning for
               Heterogeneous Multi-Die SoCs},
  author    = {Kalra, Naman},
  journal   = {arXiv preprint},
  year      = {2024},
  url       = {https://github.com/Namaniit297/FLAME_297}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- **PACT Thermal Simulator** — Z. Yuan et al., ICCAD 2021
- **AlphaChip** — A. Mirhoseini et al., Nature 2021
- **Prioritized Experience Replay** — T. Schaul et al., 2015
- **PyTorch** and **PyTorch Geometric** communities

---

<div align="center">

**⭐ If this work helped your research, please star the repo! ⭐**

[![GitHub](https://img.shields.io/badge/GitHub-FLAME_297-black?logo=github)](https://github.com/Namaniit297/FLAME_297)

*Built with ❤️ for the future of thermally resilient heterogeneous 3D chiplet systems*

</div>
