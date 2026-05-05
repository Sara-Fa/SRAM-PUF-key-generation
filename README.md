# SRAM-PUF Key Generation

This repository implements multiple approaches for extracting reliable
cryptographic keys from SRAM Physical Unclonable Functions (PUFs):

1. **TMVS (Threshold-based Majority Voting Scheme)** — see [tmvs/](tmvs/).
2. **TS-TMVS (Two-Stage TMVS)** — see [two_stage_tmvs/](two_stage_tmvs/).
3. **NVM-Free TMVS / ODHD (On-Demand Helper Data)** — pattern-based key
   extraction in which the helper data is generated on demand rather than
   persisted in non-volatile memory. See [nvm_free_tmvs/](nvm_free_tmvs/).
4. **Bernardini Helperless Stabilizer** — bit-level key extraction using
   majority voting with acceptance masks and dual-threshold evaluation.
   See [previous_work/helperless_stabilizer_bernardini/](previous_work/helperless_stabilizer_bernardini/).

## Publications

- **TMVS** ([tmvs/](tmvs/)):
  - Sara Faour, Mališa Vučinić, Filip Maksimovic, David Burnett, Paul
    Muhlethaler, Thomas Watteyne, Kristofer Pister.
    *TMVS: Threshold-based Majority Voting Scheme for Robust SRAM PUFs.*
    IEEE Symposium on Computers and Communications (ISCC), Paris, France,
    26–29 June 2024.
  - Sara Faour, Filip Maksimovic, David Burnett, Paul Muhlethaler,
    Thomas Watteyne, Kristofer Pister, Mališa Vučinić.
    *TMVS: Threshold-based Majority Voting Scheme for Robust SRAM PUFs.*
    IEEE Transactions on Information Forensics and Security (TIFS), to
    appear in 2026.
- **TS-TMVS** ([two_stage_tmvs/](two_stage_tmvs/)):
  - Sara Faour, Mališa Vučinić, Thomas Watteyne, Kristofer Pister.
    *Two-Stage Threshold-based Majority Voting Scheme (TS-TMVS) for Robust
    SRAM PUFs.* Workshop on Crystal-Free/-Less Radio and System-based
    Research for IoT (CrystalFreeIoT), International Conference on Embedded
    Wireless Systems and Networks (EWSN), Leuven, Belgium, 22 September 2025.
- **ODHD** ([nvm_free_tmvs/](nvm_free_tmvs/)):
  - Sara Faour, Mališa Vučinić, Filip Maksimovic, Thomas Watteyne,
    Kristofer Pister.
    *ODHD: On-Demand Helper Data Generation for Reliable NVM-Free Key
    Derivation from SRAM PUF.* IEEE International Conference on Information
    Security and Cryptology (ISC), Ankara, Türkiye, 22–23 October 2025.

## Getting Started

```bash
git clone <repo-url>
cd SRAM-PUF-key-generation
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate
pip install -r requirements.txt
```

The `data/SRAM_readouts/` directory bundles the experimental SRAM readouts
used by all three approaches (9 chips, 1000 readouts each). Source: prior
research at [hal.inria.fr/hal-04589272](https://inria.hal.science/hal-04589272/),
mirrored from [scum-automated-sram-read](https://github.com/bkorecic/scum-automated-sram-read).

## TMVS

The Threshold-based Majority Voting Scheme is implemented in
[tmvs/tmvs_algo.py](tmvs/tmvs_algo.py); theoretical formulas in
[tmvs/formulas.py](tmvs/formulas.py); analysis and plotting in
[tmvs/analysis.py](tmvs/analysis.py). Run with:

```bash
python -m tmvs.main
```

> Unzip `tmvs/regenerated_keys.zip` before running.

## TS-TMVS

Two-Stage TMVS code lives in [two_stage_tmvs/](two_stage_tmvs/). See its
`main.py` and `analysis/` for usage.

## NVM-Free TMVS / ODHD

```bash
# Enrollment BER (set USE_TRIVIAL = True in __main__ for trivial codebook)
python -m nvm_free_tmvs.experiments.helper_data_comparator

# Regeneration BER
python -m nvm_free_tmvs.experiments.global_ber_processor

# Aggregate per-chip results
python -m nvm_free_tmvs.experiments.averaging_data_processor

# Trivial vs full codebook regeneration BER
python -m nvm_free_tmvs.experiments.regeneration_trivial_vs_full_codebook \
    --chip L45 --nr-read 10

# Interactive analysis plots
python -m nvm_free_tmvs.plotting.plotting_main
```

See [nvm_free_tmvs/README.md](nvm_free_tmvs/README.md) for details.

## Bernardini Helperless Stabilizer

```bash
# Lambda estimation
python -m previous_work.helperless_stabilizer_bernardini.lambda_estimation

# Dual-threshold BER evaluation
python -m previous_work.helperless_stabilizer_bernardini.evaluate_dual_threshold_ber

# Enrollment BER comparison (incremental, all chips)
python -m previous_work.helperless_stabilizer_bernardini.experiments.main

# Interactive plots
python -m previous_work.helperless_stabilizer_bernardini.plotting.plotting_main
```

See [previous_work/helperless_stabilizer_bernardini/README.md](previous_work/helperless_stabilizer_bernardini/README.md)
for the full pipeline.

## Citation

If you use this code, please cite the associated paper(s).

## License

See [LICENSE](LICENSE).
