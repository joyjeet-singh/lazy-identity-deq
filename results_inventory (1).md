# PHIL-DEQ Negative Result — Results Inventory
**Source:** `Port_Hamiltonian_Implicit_Layer_DEQ.ipynb` (133 cells)
**Rule:** every number in the paper must trace to a cell in the *Verified from executed output* column. Numbers that appear only in markdown commentary are marked ⚠ and must be re-verified or dropped.

---

## 0a. FINAL central claim (post experiment-suite, cells 132 & 135 — supersedes 0b)

> The physics-structured DEQ never performs meaningful implicit computation, across two tasks, all seeds, and all ablation arms. The dominant regime is **lazy identity collapse**: the solved equilibrium equals the solver's start point to numerical precision — whatever that start is (anchor z0, arm A; z0 with the anchor removed, arm C; z0 + injected noise, arm B training) — and substituting the start for the equilibrium changes accuracy by +0.00pp. The anchor is causally irrelevant (arm C is numerically indistinguishable from arm A), and retraining with decoupled noisy starts does not rescue the DEQ — the decoder instead learns to ignore the h\* channel (arm B: h\*-zeroed acc = full acc). The minority regime, observed in 1/5 ProofWriter seeds, is **solver blow-up** (‖h\*−z0‖ = 171.4): the non-converged output acts as a decoder-adapted noise channel whose removal *improves* accuracy (zeroed 83.23% > full 81.77%). In neither regime does iteration count track difficulty (corr(K, BFS distance) ≈ 0.009), and in neither task does the full apparatus beat a 2-layer MLP baseline (ProofWriter: 89.39 ± 4.30 vs 90.39 ± 0.33; graph: 61.80 ± 1.34 vs 61.52 ± 0.72). Mechanism: the tasks are solvable from the direct context pathway, gradient structure gives the dynamics networks no incentive to compute (in the graph pipeline the initializer is fully gradient-isolated by the custom VJP), and near-identity dynamics are a stable optimum.

## 0b. Earlier claim (pre-suite; kept for provenance — anchor-causation part now falsified by arms B/C)

> In an anchored port-Hamiltonian DEQ, the solved equilibrium is numerically identical to its anchor (mean ‖h\* − z0‖ ≈ 0 across tasks, seeds, and anchor strengths). All task-relevant signal flows through the feed-forward initializer — specifically its goal-gated p0 channel — and the implicit equilibrium computation contributes +0.00pp. The solver's iteration count is invariant to problem difficulty, ruling out adaptive test-time compute.

This is stronger and cleaner than "the model didn't learn": the *pipeline learns fine* (up to 94.6% val on ProofWriter) — the DEQ inside it is an identity map on its anchor.

**Key reconciliation the outputs provide (paper must state this):** the h\*-zeroing ablation is a *confounded* measure of DEQ contribution. Zeroing h\* also removes z0's signal, so its delta measures the initializer's importance, not the solver's. The unconfounded measure is the **z0-substitution test** (bypass the solver, feed z0 directly): gap = +0.00pp. This resolves the apparent seed-instability of the zeroing ablation (Cell 118: deltas of +44.03, +19.52, +0.00, +44.03, +39.52pp across 5 seeds) — that variance tracks how the decoder distributes weight across its input channels, not DEQ function.

---

## 1. Run eras (chronology — needed so the paper reports only post-bugfix results)

| Era | Cells | State of pipeline | Usable in paper? |
|---|---|---|---|
| E1 Early MVL / broken plumbing | ~28–75 | frozen grads, NaN cascade (cell 68: loss=nan from epoch 30), missing decoder terms | Background only — do NOT report metrics |
| E2 Anchor active, pre-c_goal-fix | 82–94 | mechanically correct, chance-level (~50–56%) | Yes — as the "collapse under correct plumbing" era |
| E3 Post c_goal decoder fix | 97–104 | learns (90%+) but h\* empty | Yes — core evidence |
| E4 Elementwise gate fix | 106–113 | initializer trains, solver destabilized (mean residual 29.06) | Yes — as instability finding |
| E5 Multi-seed + sweeps + causal tests | 118–124 | decisive experiments | Yes — headline results |
| E6 Synthetic graph task | 126–130 | second domain, same collapse | Yes — headline results |
| Side | 114–117 | standalone physics rollout (motivational positive control for PH structure) | Optional, intro/appendix |

---

## 2. Claim → evidence table

### C1. Positive control: the harness and data can support learning
| Evidence | Cell | Verified number |
|---|---|---|
| Plain 2-layer MLP on same cached DeBERTa embeddings (concat c_premise, c_goal) | **94** (output) | **Test acc 89.73%**; train n=2477, test n=620; label balance train 0.495 / test 0.477 |
| Full PHIL-DEQ pipeline after c_goal fix also learns | **101** (output) | Train acc 92.37% (ep100); **test 90.98%** (constant across tol 1e-2…1e-6); Depth-3 stratified 91.15% |

⚠ **Discrepancy D1:** Cells 102/111 print "Baseline (no DEQ at all): 81.72%" as a hardcoded reference string, and markdown cells cite 81.72%. The *executed* baseline in cell 94 is **89.73%**. The 81.72% likely comes from an earlier Colab session not preserved in outputs. → Paper uses 89.73% (or re-run baseline in the seeds session and report fresh mean±std).

### C2. The equilibrium is its anchor: z0-displacement ≡ 0
| Evidence | Cell | Verified number |
|---|---|---|
| r_anchor sweep {0.005, 0.02, 0.05, 0.1, 0.2}, 60 ep, val | **121** | z0_displacement = **0.000002 at every value**; val full acc 82.5–94.6%; zeroed val acc constant 52.42% |
| Displacement-penalty removal (does the penalty cause it?) | **124** | z0_disp = **0.00000** through 100 epochs → penalty not the cause |
| Graph task, 80 ep | **128** | z0_disp = 0.00000 every logged epoch |

### C3. The DEQ solve contributes nothing: z0-substitution gap = 0
| Evidence | Cell | Verified number |
|---|---|---|
| ProofWriter triad | **122** | Full 91.77% = z0-substituted 91.77% (**gap +0.00pp**); h\*-zeroed 47.74% |
| Graph task | **129** | Full 60.70% = z0-only 60.70% (**gap +0.00pp**) |

### C4. All signal flows through the initializer's goal-gated channel
| Evidence | Cell | Verified number |
|---|---|---|
| Initializer ablation modes (trained model, inference-time) | **123** | Full 91.77%; **p0 zeroed → 47.74%**; q0 zeroed → 91.77%; shared projection → 91.77% |
| z0-only accuracy by proof depth | **123** | D1 92.50%, D2 92.50%, D3 91.88%, D5 90.00% (range 2.50pp) — flat: task solvable without iterative computation |

### C5. h\* carries no signal beyond z0 (probe evidence)
| Evidence | Cell | Verified number |
|---|---|---|
| Linear/MLP probe on frozen h\* only (E3 model) | **103** | Loss plateaus 0.694–0.696 (≈ ln 2 entropy floor of 0.495/0.505 balance); **test 51.45%** |
| Same probe after E4 gate fix | **112** | 88.87% BUT h\* std 2.40 with range [−7.19, 8.82]; solver mean residual **29.06**, max 48.27 (cell 111) → signal rides on diverged values, not equilibria |

⚠ **Discrepancy D2:** markdown cell 104 cites probe = 51.29% and zeroing = 91.61% "matching full model 91.61%"; executed outputs show probe = **51.45%** (cell 103) and zeroing = **91.29%** vs full 91.14% reference (cell 102). Markdown numbers likely from a prior session. → Paper uses output-verified numbers only.
⚠ Markdown cell 113 cites probe learning-curve values (81.02% → 49.26% crash) not present in preserved outputs — drop or re-run.

### C6. No adaptive compute: iteration count is difficulty-invariant
| Evidence | Cell | Verified number |
|---|---|---|
| Anderson tolerance sweep, E2/E3 eras | 90, 101 | Mean K = 295.8–300.0 at every tolerance 1e-2…1e-6; accuracy constant across tolerances |
| Graph task: corr(K, true BFS distance) | **129** | **nan** — K range **[300, 300]**, zero variance across 1000 test samples |

### C7. Causal test: the anchor's pull, not the task, explains the collapse — and untrained regions are empty
| Evidence | Cell | Verified number |
|---|---|---|
| Forced decoupled start at inference (trained graph model) | **130** | Displacement 7.954 (vs 0.000) — solver *can* move; accuracy **55.60% vs 60.70%** — off-anchor regions contain no learned structure |

### C8. Seed robustness (and the confound it exposed)
| Evidence | Cell | Verified number |
|---|---|---|
| 5 seeds × 100 ep, ProofWriter | **118** | Full test acc 90.97–91.77% (tight); h\*-zeroed 47.74/72.26/90.97/47.74/52.26% → deltas +44.03/+19.52/+0.00/+44.03/+39.52pp (wild) |
| Interpretation | 122 vs 118 | Zeroing ablation is confounded (removes z0 signal too); z0-substitution is the correct test and gives 0.00pp — reframe C8 as evidence *for* the confound argument |

### C9. Gate fix instability (secondary finding)
| Evidence | Cell | Verified number |
|---|---|---|
| Elementwise gate lets initializer train | **110** | init_grad_norm 0.00265 → 0.02646 (order of magnitude ↑); full test 87.07% |
| …but destabilizes the solver | **111** | Mean final residual **29.06** (vs 0.0935 in cell 102 era), max 48.27 |
| Instrumented collapse window | **120** | Epochs 74–90: max_h 6.9 → 21.7, init_grad peak 0.05 → 1.72, acc 92% → 62% → partial recovery; per-batch CSV saved |

### C10. Side result: PH structure controls long-horizon energy drift (motivation, not the main claim)
| Evidence | Cell | Verified number |
|---|---|---|
| 5000-step rollout, learned dynamics | **117** | Baseline max normalized H-drift **6.15e25** (saturates); port-Hamiltonian **2.39e3**; at step 501: 3.48e17 vs 2.25e-2 |

### C11. Known confound to disclose: graph-task overfitting
| Evidence | Cell | Verified number |
|---|---|---|
| Train/test gap on 6000-graph dataset | **128/129** | Train 100.00% by ep 30 (loss → 0.0003) vs test 60.70% — memorization; orthogonal to the z0-identity result but must be stated as a limitation (or mitigated with the pending data-scale run) |

---

## 3. Experimental setup facts (for the Methods section)
- Encoder: frozen `microsoft/deberta-v3-base`; ProofWriter (HF dataset; 585,552 train rows scanned), stratified 800/depth over depths {1,2,3,5}; depth-5 capped at 697 → **2477 train / 620 test** (later split 2105/372/620 with untouched test, cell 119).
- Label balance: train 0.495, test 0.477 (cell 94).
- Solver: Anderson acceleration, max_iter 300, tolerance sweep 1e-2…1e-6; IFT/adjoint backward extended to z0; spectral clamp; anchor "Option A" (cells 88–89 banners).
- Graph task: 6000 train / 1000 test, 50.0% balance, BFS ground-truth reachability (cell 126).
- Optimizer: optax adamw, weight_decay 1e-4 (markdown cell 91 — ⚠ verify against code cell before quoting).
- Training cost: ~1.7–2.0 s/epoch ProofWriter; 316 s for 80-epoch graph run; 5-seed run ~125–175 s/seed — worth stating to emphasize reproducibility on free Colab.

## 4. Experiment suite results — ALL FORMERLY-PENDING SLOTS NOW FILLED

### 4.1 Graph task, 5-arm suite (VERIFIED from cell 132 output; raw logs in results_pending.zip)
| arm | n seeds | full acc | z0-sub acc | gap (pp) | h*-zeroed | mean disp | dstart acc | dstart disp | train acc |
|---|---|---|---|---|---|---|---|---|---|
| A coupled (replication, **P2**) | 5 | 61.80 ± 1.34 | 61.80 ± 1.34 | 0.00 ± 0.00 | 60.92 ± 1.01 | 6.9e-07 | 56.06 ± 1.26 | 7.896 | 100.00 ± 0.00 |
| B decoupled retraining (**P1**) | 5 | 61.82 ± 0.96 | 61.82 ± 0.96 | 0.00 ± 0.00 | **61.84 ± 0.96** | 2.8e-07 | 61.90 ± 0.90 | 7.913 | 100.00 ± 0.00 |
| C no anchor (r=0) | 5 | 61.80 ± 1.34 | 61.80 ± 1.34 | 0.00 ± 0.00 | 60.92 ± 1.01 | 7.0e-07 | 56.06 ± 1.26 | 7.896 | 100.00 ± 0.00 |
| D context-only baseline | 5 | 61.52 ± 0.72 | — | — | — | — | — | — | 100.00 ± 0.00 |
| E 30k graphs (**P3**) | 3 | 64.20 ± 0.79 | 64.20 ± 0.79 | 0.00 ± 0.00 | 64.37 ± 1.32 | 6.5e-07 | 60.97 ± 0.49 | 7.896 | 100.00 ± 0.00 |

K statistics: arm A/C range [1,300], mean 299.8, corr(K, BFS dist) = **0.0089**; arm B mean 104.3, std 142 (bimodal early-exit), corr = 0.0020; arm E corr = −0.0321.
Key reads: **P1 answered** — decoupled retraining does not rescue the DEQ; solver ends where it starts (train disp ≈ 7.88, matching E‖σε‖ = σ·E‖N(0,I₁₆)‖ ≈ 2 × 3.94 = 7.87 — the equilibrium IS the noisy start) and the decoder learns to ignore h\* (B: zeroed = full). **Anchor causally irrelevant** (C ≡ A per-seed; losses match to 4 decimals; disp differs only at 1e-7). **DEQ+initializer add nothing over context** (A ≈ D). **P3**: 5× data → +2.4pp test, train still 100%, gap still 0.00 — collapse is data-scale-independent.
⚠ Note for paper accuracy: arm A/C K-variance means the earlier "zero variance, corr = nan" phrasing (cell 129, single seed) must be replaced by "corr ≈ 0.009 (pooled, 5 seeds; 99.9% of solves at max_iter)".

### 4.2 ProofWriter, 5-seed triad + fresh baseline (**P4**) (VERIFIED from cell 135 output; logs on Drive)
| seed | full | z0-sub | gap (pp) | h*-zeroed | ‖h*−z0‖ | baseline |
|---|---|---|---|---|---|---|
| 1 | 91.77 | 91.77 | +0.00 | 47.74 | 2e-06 | 90.65 |
| 2 | 91.29 | 91.61 | −0.32 | 53.39 | 0.120 | 90.32 |
| 3 ⚠ blow-up | 81.77 | 32.58 | +49.19 | **83.23** | **171.43** | 90.00 |
| 4 | 90.32 | 90.32 | +0.00 | 47.74 | 2e-06 | 90.81 |
| 5 | 91.77 | 91.77 | +0.00 | 52.26 | 0.000 | 90.16 |
| **mean ± std** | 89.39 ± 4.30 | 79.61 ± 26.30 | 9.77 ± 22.04 | 56.87 ± 14.96 | 3.4e1 ± 7.7e1 | **90.39 ± 0.33** |

Key reads: the distribution is **bimodal — do NOT report the pooled mean as the headline** (it averages two regimes). 4/5 seeds: lazy identity collapse (disp ≤ 0.12, |gap| ≤ 0.32pp). 1/5 seeds (seed 3): solver blow-up (disp 171.4) where the non-converged h\* acts as decoder-adapted noise — z0-substitution breaks the model (32.58%, below chance) and **zeroing h\* improves over full (83.23 > 81.77)**, i.e. even the "load-bearing" regime is degenerate. This is the E4-instability phenomenon (C9, cells 111/120) resurfacing under the standard pipeline — connect the two in the paper. **D1 RESOLVED**: fresh per-seed baseline = 90.39 ± 0.33 (cell 94's 89.73 compatible; the stale 81.72 is retired). Full pipeline does not beat the baseline on either task.
Seed 1 exactly reproduces the original cell-122 triad (91.77/91.77/47.74) → replication chain intact.

## 5. Figure plan
- **Fig 1 (mechanism schematic):** initializer → z0 → anchored solver → h\* ≡ z0 → decoder; the one scalar-gated goal channel highlighted.
- **Fig 2:** z0-displacement vs training epoch across r_anchor values (from cell 121/124/128 logs) — flat zero lines.
- **Fig 3:** ablation bars, both tasks: full / z0-substituted / h\*-zeroed / p0-zeroed (cells 122, 123, 129).
- **Fig 4:** probe loss vs epoch with ln 2 entropy floor marked (cell 103), inset: E4 probe anomaly + solver residuals (111/112).
- **Fig 5 (secondary):** gate-fix instability window — max|h\*| and init_grad_norm vs epoch 40–100 (cell 120 CSV).
