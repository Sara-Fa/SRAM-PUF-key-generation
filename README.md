# TMVS-for-Robust-SRAM-PUFs

This repository contains Python scripts that implement **Threshold-based Majority Voting Scheme (TMVS)** for extracting reliable secret keys from SRAM PUF. **TMVS** has been thoroughly tested using experimental data, which is available in the `data/SRAM_readouts` directory. These scripts provide various analysis functions to evaluate **TMVS** both theoretically and experimentally, focusing on key metrics such as error decoding probability, selection probability, memory requirements, and more.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Experimental Data](#experimental-data)

## Overview 

**TMVS** is a novel flexible and efficient solution for extracting reliable secret key from SRAM PUFs. TMVS can be implemented in software without the need for pre-processing steps such as individual cells’ bit error rate (BER) qualification, nor modifying the design of SRAM cells. **TMVS** does not depend on powerful Error correction codes (ECCs) with complex decoders, but uses a simple majority voting decoder instead.

This repository provides:
- An implementation of **TMVS** in `tmvs.py`.
- Theoretical formulas in `formulas.py`.
- Tools to analyze and plot the performance of the extracted keys using various metrics in `analysis.py`.

## Getting Started

To ensure that all dependencies are correctly installed and that the environment is consistent across different setups, it is recommended to use a virtual environment.

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Sara-Fa/TMVS-for-Robust-SRAM-PUFs.git
    cd TMVS-for-Robust-SRAM-PUFs
    ```

2. **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    ```

3. **Activate the Virtual Environment:**

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

4. **Install the Required Packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main script for running **TMVS** and performing the analyses is `main.py`. To run the script, simply execute:

```bash
python main.py
```

## Experimental Data

The experimental data used for testing TMVS is stored in the `data/SRAM_readouts` directory. This data was initially generated within a [previous research work](https://inria.hal.science/hal-04589272/) and is available in [another repository](https://github.com/bkorecic/scum-automated-sram-read) and has been copied here for convenience.
