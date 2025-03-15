"""
Task 1: Fermi-Dirac Statistics Solution

This script derives the partition function for fermions under the grand canonical ensemble.
Fermions are particles with half-integer spin that obey the Pauli exclusion principle.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, exp, Sum, oo, latex

def main():
    # Define symbolic variables
    mu, epsilon, beta, M = symbols('mu epsilon beta M', real=True)
    i = symbols('i', integer=True)
    
    # Grand canonical partition function for fermions
    # For fermions, each energy level can be either occupied (1) or unoccupied (0)
    # The grand canonical partition function is:
    # Ω_G = ∏_i (1 + exp(-β(ε_i - μ)))
    
    # For a system with M energy levels ε, 2ε, 3ε, ..., Mε
    # The energy of level i is i*ε
    single_level_term = 1 + exp(-beta * (i * epsilon - mu))
    
    # The product over all energy levels
    partition_function = sp.product(single_level_term, (i, 1, M))
    
    # Simplify and expand the expression
    partition_function_expanded = sp.expand(partition_function)
    
    # Taking the logarithm to get the grand potential
    grand_potential = -1/beta * sp.log(partition_function)
    
    # Print the results
    print("Fermi-Dirac Statistics - Grand Canonical Ensemble")
    print("=" * 50)
    print("\nGrand Canonical Partition Function:")
    print(f"Ω_G = {partition_function}")
    
    print("\nGrand Potential:")
    print(f"Φ = {grand_potential}")
    
    # Create a LaTeX representation for visualization
    latex_partition = latex(partition_function)
    
    # Visualize the partition function using matplotlib
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"$\\Omega_G = {latex_partition}$", 
             fontsize=14, ha='center', va='center')
    plt.axis('off')
    plt.title("Fermi-Dirac Grand Canonical Partition Function", fontsize=16)
    plt.tight_layout()
    plt.savefig("Task1_FermiDirac/fermi_dirac_partition.png", dpi=300, bbox_inches='tight')
    
    # Create a more visually appealing figure showing the Fermi-Dirac distribution
    plot_fermi_dirac_distribution()

def plot_fermi_dirac_distribution():
    """
    Create a visually appealing plot of the Fermi-Dirac distribution function.
    """
    # Energy values
    energy = np.linspace(-5, 5, 1000)
    
    # Fermi-Dirac distribution for different temperatures
    # f(E) = 1 / (exp((E-μ)/kT) + 1)
    # Setting μ = 0 for simplicity
    
    temperatures = [0.1, 0.5, 1.0, 2.0]
    plt.figure(figsize=(10, 6))
    
    for T in temperatures:
        fd_dist = 1 / (np.exp(energy/T) + 1)
        plt.plot(energy, fd_dist, linewidth=2.5, label=f'T = {T}')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Energy (E - μ)', fontsize=14)
    plt.ylabel('Occupation Probability f(E)', fontsize=14)
    plt.title('Fermi-Dirac Distribution Function', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add a dark theme
    plt.style.use('dark_background')
    plt.tight_layout()
    plt.savefig("Task1_FermiDirac/fermi_dirac_distribution.png", dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

if __name__ == "__main__":
    main() 