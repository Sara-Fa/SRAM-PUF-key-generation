# NVM-Free TMVS / ODHD

NVM-free key extraction from SRAM PUFs using On-Demand Helper Data (ODHD):
the helper data is generated on demand rather than stored in non-volatile
memory.

## Requirements

- Python 3.10+ with `pip install -r ../requirements.txt`
- SRAM readouts present in `../data/SRAM_readouts/`

## Running experiments

From the repository root with the virtual environment active:

```bash
# Enrollment BER (helper data comparison)
python -m nvm_free_tmvs.experiments.helper_data_comparator

# Regeneration BER
python -m nvm_free_tmvs.experiments.global_ber_processor

# Aggregate per-chip results
python -m nvm_free_tmvs.experiments.averaging_data_processor

# Trivial vs full codebook regeneration BER
python -m nvm_free_tmvs.experiments.regeneration_trivial_vs_full_codebook \
    --chip L45 --nr-read 10
```

## Plotting

```bash
python -m nvm_free_tmvs.plotting.plotting_main
```

The interactive plots read aggregated `.h5` files. Run
`averaging_data_processor` to generate them before plotting.

## Notes

- Default parameters are set inside each script; adjust code lengths,
  thresholds, and chip lists there if needed.
- Cached `.h5` files can be safely deleted and recreated.
