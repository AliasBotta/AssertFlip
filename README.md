# 🐛 AssertFlip Replication Package

This is the official replication package for our paper:

> **AssertFlip: Reproducing Bugs via Inversion of LLM-Generated Passing Tests**

AssertFlip is a system for automatically generating bug-reproducing tests from natural language reports

---

## 🔧 Setup Instructions

### 1. Requirements

- Python 3.10+
- Docker
- `conda` (used inside Docker containers)

Install dependencies:

```bash
pip install -e .
```
---

### 2. Add LLM API Credentials

The file scripts/.env is already created.

Just open it and insert your own credentials like this:

```bash
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=https://your_azure_endpoint
AZURE_API_VERSION=2024-05-01-preview
```
---

### 3. How to Run

Default (used in the paper)

```bash
python scripts/run_parallel.py
```

This uses:
- Agentless localization
- Pass-invert strategy
- 10 regeneration attempts
- 10 refinement attempts
- LLM validation enabled
- Planner enables

**Config is controlled in scripts/config.py.**

---

### 4. Datasets

All datasets are in the datasets folder. These are the exact files used in our experiments:

- SWT_Verified_Agentless_Test_Source_Skeleton.json (default for Verified)
- SWT_Verified_Test_Source_Skeleton.json (perfect localization dataset)
- SWT_Lite_Agentless_Test_Source_Skeleton.json (default for Lite)
- SWT_Lite_Agentless_Unique_Only.json (default for Lite 188 unique instances)

To switch datasets, change:

```bash
DATASET_PATH in scripts/config.py.
```
---

### 5. Running Ablations

Regeneration Ablation (0 or 5 attempts)

Edit this line in scripts/config.py:

```bash
max_regeneration_retries = 1  # for no regenerations 
# or
max_regeneration_retries = 5  # for the 5 regeneration ablation
```

Then run:

```bash
python scripts/run_parallel.py
```

### 6. Running No Validation Ablation

```bash
python scripts/run_parallel_without_validation_ablation.py
```

### 7. Running No Planner Ablation

```bash
python scripts/run_parallel_without_planner_ablation.py
```

### 8. Perfect Localization

Change dataset in scripts/config.py to:

```bash
DATASET_PATH = "datasets/SWT_Verified_Test_Source_Skeleton.json"
```

Then run the default script again.

```bash
python scripts/run_parallel.py
```
---

### 9. Generate Predictions

To generate preds.json from results:

```bash
python scripts/generate_preds_phases.py --results-dir results/
```

We also include our original prediction files in the preds_files folder for direct use.

---

### 10. Evaluation Instructions

The previous steps produces predictions in SWT-Bench format. You can then evaluate them using SWT-Bench instructions: https://github.com/logic-star-ai/swt-bench

We also provide:

- Full outputs preds in preds_files/
- Full results after evaluating on SWT-Bench for each reported run in evaluation_results_on_SWT_Bench/

---
## 📚 Citation

If you use this codebase, datasets, or experiments in your research, please cite our paper:

```bibtex
@article{khatib2025assertflip,
  title={AssertFlip: Reproducing Bugs via Inversion of LLM-Generated Passing Tests},
  author={Khatib, Lara and Mathews, Noble Saji and Nagappan, Meiyappan},
  journal={arXiv preprint arXiv:2507.17542},
  year={2025}
}
```
---

### Acknowledgment 

This project uses components from the opensource test generator [Coverup](https://github.com/plasma-umass/coverup), licensed under the Apache 2.0 License. 



