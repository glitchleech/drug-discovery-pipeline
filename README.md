# Drug Discovery Pipeline

End-to-end computational drug discovery pipeline using Python, PyTorch, RDKit, and TDC.  
Includes molecular graph generation, diffusion-based molecule generation (EGNN / discrete denoiser), ADMET prediction, and visualization of generated compounds.

## Features

- Convert SMILES molecules to graph representations and vice versa.
- Molecular generation using:
  - EGNN-style continuous denoiser (DDPM)
  - Discrete birth-death diffusion denoiser
- ADMET surrogate prediction (Lipophilicity)
- Visualization of generated vs. original molecules (QED vs LogP)
- Automatic dependency installation and environment checks
- Unit tests for sanity checks and round-trip SMILES conversion

## Requirements

- Python 3.9+
- PyTorch
- RDKit
- DeepChem
- TDC (Therapeutics Data Commons)
- scikit-learn, pandas, matplotlib, numpy

Dependencies can be installed automatically using the included helper code.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/username/drug-discovery.git
cd drug-discovery
