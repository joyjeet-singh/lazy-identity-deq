# The Equilibrium Is the Initialization

**Lazy Identity Collapse in Physics-Structured Deep Equilibrium Reasoning**
Joyjeet Singh · [ORCID 0009-0005-1512-7439](https://orcid.org/0009-0005-1512-7439) · arXiv: *(link after posting)*

A controlled negative result: in a port-Hamiltonian deep equilibrium model
with a learned initialization, the implicit computation is a silent no-op.
Across two tasks (ProofWriter entailment, BFS-verified graph reachability),
19 training runs, an anchor-strength sweep, and three intervention arms, the
solved equilibrium equals the solver's **start point** to numerical precision,
and bypassing the solver entirely changes test accuracy by **+0.00pp** in
18 of 19 runs. Removing the anchor changes nothing; retraining from noisy
starts teaches the decoder to ignore the module; the one escaping run
*diverges* into decoder-adapted noise whose removal improves accuracy. The
full apparatus never beats a two-layer MLP. The paper distills a four-test
diagnostic protocol for auditing claimed implicit computation in DEQs,
looped, and recurrent-depth models.

## Repository contents

```
paper/
  anchored_equilibria_collapse.tex   # paper source
  refs.bib
  figures/                           # all figures (PDF + PNG)
experiments/
  graph_experiments_colab.py         # self-contained 5-arm suite (graph task)
  proofwriter_prerequisites_colab.py # verbatim pipeline definitions (paste 1st)
  proofwriter_addon_colab.py         # 5-seed triad + fresh baselines (paste 2nd)
logs/
  results_pending/                   # raw JSONL + K/dist .npz, graph suite (21 runs)
  results_proofwriter_addon/         # raw JSONL, ProofWriter (5 runs)
analysis/
  make_figures.py                    # regenerates every figure from logs/
notebook/
  Port_Hamiltonian_Implicit_Layer_DEQ.ipynb   # full research log (133+ cells)
```

## Reproduce everything

**Figures and tables from the released logs (no GPU, ~1 minute):**
```bash
pip install numpy matplotlib
python analysis/make_figures.py        # reads logs/, writes figures/
```
Every number in the paper is computed from the raw per-run logs by this
script and the `analyze()` function in the experiment suites — never from
in-memory state. Each run's JSONL begins with a manifest recording library
versions, device, config hash, and dataset SHA-256.

**Rerun the experiments (free Colab GPU):**
- *Graph task (arms A–E, ~2.5–3 h total):* open a Colab notebook, paste
  `experiments/graph_experiments_colab.py`, run. `RUN` flags at the top let
  you execute arms individually; completed runs are skipped on re-execution.
- *ProofWriter (~25 min + one-time embedding extraction):* paste
  `proofwriter_prerequisites_colab.py` as one cell, run; then paste
  `proofwriter_addon_colab.py` as the next cell, run. A preflight check
  hard-fails with instructions if definitions are missing.

## The four-test protocol (TL;DR)

| Test | Procedure | Fail signature |
|---|---|---|
| **T1 Substitution** | feed the solver's *start* to the decoder, skip the solve | accuracy unchanged → module computes nothing |
| **T2 Probe vs entropy floor** | fresh classifier on the module output alone | probe loss → ln 2 (balanced binary); verify solver residuals before trusting a *passing* probe |
| **T3 Iterations vs difficulty** | correlate solver steps with ground-truth difficulty | corr ≈ 0 or degenerate counts |
| **T4 Forced displacement** | start the solver off its learned initialization | accuracy degrades → capability absent, not suppressed |

## Citation

```bibtex
@article{singh2026lazyidentity,
  title  = {The Equilibrium Is the Initialization: Lazy Identity Collapse
            in Physics-Structured Deep Equilibrium Reasoning},
  author = {Singh, Joyjeet},
  journal= {arXiv preprint arXiv:XXXX.XXXXX},
  year   = {2026}
}
```

## License

Code: MIT. Paper text and figures: CC BY 4.0 unless noted otherwise.
