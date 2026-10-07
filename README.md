# Linear Probe Transfer

Code and result artifacts for the study **When Can a Linear Probe Be Reused? Geometry, Transfer Actions, and Risk Certification**.

**Repository maintainer:** Md Muntaqim Meherab  
**Repository name:** linear-probe-transfer

## Overview

This project studies what happens when a linear probe fitted at one layer is applied to another layer. It distinguishes direct coefficient reuse from a learned ridge alignment and compares each transferred predictor with a fitted target-layer reference. The experiments combine frozen transformer features, controlled preprocessing interventions, similarity-based risk estimation, and simultaneous intervals for clipped fitted-reference loss differences.

The notebook is the executable research workflow. The saved outputs record the completed v3 paper-profile run. The archived v2 pilot section inside the notebook is explicitly marked as legacy and is not evidence for the v3 experiments.

## Main findings in the saved run

At clipped-loss tolerance 0.02, the simultaneous betting intervals certify 315 of 396 aligned GPT-2/SST-2 pair-repeat decisions and 248 of 396 aligned BERT/SST-2 decisions. Direct identity reuse certifies 12/396 and 22/396, respectively. The 396 decisions in each setting are three descriptive repeats over a shared sampled pool, not 396 independent datasets.

Across the 18 SST-2-to-IMDB and SST-2-to-Allociné domain-transfer runs, neither transfer action has a certified decision at tolerance 0.02. Calibration uses target-domain data, so this result concerns transfer performance in those runs; it is not a source-calibrated guarantee under arbitrary distribution shift.

In the exactly enumerable bounded simulation, both interval methods cover both directional risks in all 1,000 repetitions at each calibration size. At n=2,048, mean interval width is 0.00707 for betting and 0.03430 for empirical Bernstein; the certified fractions are 0.997 and 0. These are simulation results for the stated data-generating process, not a claim about every transfer family.

All results and qualifications above are documented in the executed notebook and accompanying CSV tables. Raw-loss estimation and clipped-loss certification are separate estimands. Certification is relative to the specified fitted reference and does not establish absolute task quality.

## Experimental design

- **Frozen models:** GPT-2, GPT-2 medium, Pythia-160m, and BERT-base.
- **Tasks:** SST-2, AG News Sports-versus-rest, CoLA, French Allociné sentiment, and normalized next-token surprise; plus SST-2-to-IMDB and SST-2-to-Allociné transfer.
- **Features:** block outputs captured before terminal normalization; decoder features use the last real token, BERT uses CLS, and next-token surprise uses the preceding context position. Inputs are truncated to 64 tokens.
- **Splits:** 12,000 training / 3,000 calibration / 3,000 test examples, except CoLA at 5,000 / 1,500 / 1,500. These are internal splits sampled from public training data, not official test-set results. Repeats reuse the sampled pool.
- **Primary preprocessing:** native coordinates, shared PCA, layerwise centering with shared PCA, and layerwise centering/RMS scaling with shared PCA. The shared PCA dimension is 128; ridge strength is selected by training-only generalized cross-validation.
- **Inference:** simultaneous empirical Bernstein and predictable betting intervals are compared for fixed transfer actions. The primary decision tolerance is 0.02 on clipped fitted-reference loss differences.

For the precise definitions, settings, model/data revision hashes, checks, and uncertainty qualifications, see the notebook and its embedded run manifest.

## Repository contents

- `notebooks/main_experiment.ipynb` — executed v3 workflow, code, figures, outputs, run manifest, and the full checksummed result-table bundle.
- `figures/` — the two conceptual paper figures requested for this repository: `overview_transfer_actions.png` and `theory_invariance_oracle_regret.png`. These are explanatory diagrams, not experimental plots. Add the separately generated experimental figures here under their original filenames when ready.
- `tables/` — 17 curated CSV summaries from the supplied table export, with filenames preserved.
- `results/all_tables/` — all 123 CSV tables recovered from the notebook’s embedded output bundle, including pair-level results. Two files are explicitly labeled as v2 legacy pilot tables.
- `scripts/export_embedded_tables.py` — standard-library-only script to recover and checksum-verify all 123 tables from the notebook.
- `requirements.txt` — pinned Python analysis dependencies. Install the hardware-appropriate PyTorch build separately.
- `LICENSE` — MIT License for this repository.

No datasets, pretrained weights, native activation caches, or multi-gigabyte checkpoint archive are included. The notebook downloads data/models from their recorded Hugging Face revisions and writes generated outputs to `aistats2027_v3_outputs/`.

## Reproduce

Reference environment: Python 3.12.13, PyTorch 2.10.0+cu128, NumPy 2.0.2, SciPy 1.16.3, pandas 2.3.3, scikit-learn 1.6.1, Matplotlib 3.10.0, Transformers 5.0.0, Datasets 5.0.0, and Hugging Face Hub 1.11.0.

1. Create a Python 3.12 environment and install a PyTorch build compatible with your CPU/GPU using the official PyTorch installation selector.
2. Install the remaining dependencies with `pip install -r requirements.txt`.
3. Launch Jupyter and run all cells in `notebooks/main_experiment.ipynb` with internet access enabled. The default profile is `paper` and attempts the full experiment grid.
4. To run a smaller real-data subset, edit `JOB_FILTER` in the first code cell, for example to `{"gpt2_sst2"}`. The notebook saves feature caches and completed checkpoints under the output root so staged runs can resume.

Setting `AISTATS_RUN_MODE=smoke` exercises the smoke/synthetic path; those outputs must not be presented as real-model findings. The complete paper run is substantial and has no single-session runtime guarantee. A GPU is recommended for feature extraction.

To recover and verify all embedded result tables without rerunning the experiments:

~~~bash
python scripts/export_embedded_tables.py notebooks/main_experiment.ipynb --output-dir results/recovered_tables
~~~

## Data and model provenance

Dataset and model identifiers, resolved commit revisions, extraction conventions, split hashes, and configuration are recorded in the notebook’s embedded manifest and per-run metadata. The workflow does not distribute raw text, model weights, or activations. Users should consult the upstream dataset and model terms before reuse.

## Anonymity during review

This repository is intentionally attributed to its maintainer. AISTATS 2027 requires submissions to remain anonymized and prohibits submission links that reveal author identity. Do not link this public repository from a blinded submission during review; use an anonymized code copy if code must be submitted for review. See the [AISTATS 2027 Call for Papers](https://virtual.aistats.org/Conferences/2027/CallForPapers).

## License and citation

The repository code and included diagrams are distributed under the MIT License. Please cite the associated manuscript, **When Can a Linear Probe Be Reused? Geometry, Transfer Actions, and Risk Certification**, when using the methods or results. Add the final bibliographic record here once it is public.
