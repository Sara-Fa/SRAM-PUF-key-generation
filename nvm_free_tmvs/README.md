NVM-Free TMVS experiments and plotting
======================================

This package contains the NVM-free TMVS experiments (helper data and BER evaluation) plus interactive plotting utilities.

Requirements
------------
- Python 3.10+ with `pip install -r requirements.txt`
- SRAM readouts already present in `data/SRAM_readouts`
- Precomputed caches are **not** shipped in the repo (zips removed for GitHub). If you have `Enroll_comparator_data.zip` / `BER_comparator_data.zip` from elsewhere, place them in `nvm_free_tmvs/` and unzip; otherwise run the scripts below to regenerate caches.

Recompute experiments
---------------------
Run from the repository root (venv active):
- Enrollment/helper data caches (writes `.h5` files under `nvm_free_tmvs/Enroll_comparator_data`):
  - `python -m nvm_free_tmvs.experiments.helper_data_comparator`
- Regeneration BER caches (writes `.h5` files under `nvm_free_tmvs/BER_comparator_data`):
  - `python -m nvm_free_tmvs.experiments.global_ber_processor`
- Aggregate per-chip results into averaged `.h5` summaries (uses the cache files from above):
  - `python -m nvm_free_tmvs.experiments.averaging_data_processor`

Plots
-----
Run the interactive menu and choose the desired figure:
- `python -m nvm_free_tmvs.plotting.plotting_main`
Figures are written to the working directory (PDF/PNG as configured in `plotting/plotting_configuration.py`).

Notes
-----
- Default parameters are set inside each script; adjust code lengths, thresholds, and chip lists there if needed.
- Cached `.h5` files can be safely deleted/recreated; zipped archives keep a lightweight copy for sharing.

