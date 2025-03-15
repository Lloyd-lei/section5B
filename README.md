# Physics 129AL - Computational Physics Week 5B

This repository contains solutions for the Week 5B section worksheet focusing on quantum statistical mechanics.

## Project Structure

- `Task1_FermiDirac/`: Derivation of the partition function for Fermi-Dirac statistics
- `Task2_BEC/`: Implementation of Bose-Einstein Condensate simulations
  - `a_microstates/`: Analysis of microstates in a 2-level boson system
  - `b_classical_partition/`: Classical partition function under canonical ensemble
  - `c_classical_average/`: Classical average particle number calculations
  - `d_quantum_partition/`: Quantum partition function under canonical ensemble
  - `e_quantum_average/`: Quantum average particle number calculations
  - `f_grand_canonical/`: Quantum partition function under grand canonical ensemble
  - `g_h_particle_number/`: Particle number calculations using grand potential
  - `i_near_degenerate/`: Simulation of near-degenerate Bose systems
  - `j_without_BEC/`: Design of a Bose system without BEC behavior
  - `k_with_BEC/`: Design of a Bose system with BEC behavior

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## How to Run

Each task directory contains Python scripts that can be executed independently:

```bash
python Task1_FermiDirac/solution.py
python Task2_BEC/k_with_BEC/bec_simulation.py
# etc.
```

## Author

[Your Name]

## License

MIT 