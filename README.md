# Generalized Adaptive Physics-Informed Neural Networks for Oppenheimer–Snyder Collapse

This repository contains a Physics-Informed Neural Network (PINN) implementation for studying relativistic gravitational collapse in the Oppenheimer–Snyder model using adaptive residual weighting and GPU-accelerated training.

The project was developed as part of my **"Estança d'Investigació"** research project.

**Author:** Óscar Delgado Fort

---

# Important Acknowledgment

This work is **not an original PINN framework**.

The entire codebase is a direct adaptation and extension of the work developed by **Antonio Ferrer** in:

> Antonio Ferrer et al.  
> *Generalized adaptive physics-informed neural networks (GA-PINNs) for solving partial differential equations*  
> Computer Methods in Applied Mechanics and Engineering (2024)  
> DOI: https://doi.org/10.1016/j.cma.2024.116940

Official article:

https://www.sciencedirect.com/science/article/pii/S0045782524001622

This project adapts the GA-PINN methodology to relativistic collapse scenarios inspired by the Oppenheimer–Snyder solution.

The research project was supervised by:

- **Anastasos Theodoropoulos**

who also made key improvements to the code.
---

# Features

- Physics-Informed Neural Networks (PINNs)
- GPU acceleration with CUDA
- Mixed precision training (AMP)
- Adaptive residual weighting
- Relativistic hydrodynamics formulation
- Oppenheimer–Snyder collapse setup
- Sobol collocation sampling
- Analytical solution comparison
- PyTorch implementation

---

# Repository Structure

```text
.
├── training_script.py
├── models.py
├── utils.py
├── config.json
├── data/
├── images/
└── models/
