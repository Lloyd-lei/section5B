"""
Task 2f: Quantum Partition Function under the Grand Canonical Ensemble

This script calculates and visualizes the quantum grand canonical partition function
for a system of bosons in a 2-level system, and determines the conditions for
normalizability of the system.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, exp, Sum, oo, latex
from matplotlib import cm

def grand_canonical_partition_function(beta, mu, epsilon, N_max=100):
    """
    Calculate the grand canonical partition function for bosons in a 2-level system.
    
    Args:
        beta: Inverse temperature (1/kT)
        mu: Chemical potential
        epsilon: Energy of the excited state
        N_max: Maximum number of particles to consider
        
    Returns:
        Grand canonical partition function value
    """
    # For the grand canonical ensemble, we sum over all possible particle numbers
    # Ω_G = sum_{N=0}^∞ exp(βμN) * Z_N
    # where Z_N is the canonical partition function for N particles
    
    Omega_G = 0
    
    for N in range(N_max + 1):
        # Calculate the canonical partition function for N particles
        Z_N = 0
        for n0 in range(N + 1):
            n1 = N - n0
            Z_N += np.exp(-beta * n1 * epsilon)
        
        # Add the contribution to the grand canonical partition function
        Omega_G += np.exp(beta * mu * N) * Z_N
    
    return Omega_G

def symbolic_grand_canonical_partition_function():
    """
    Derive the symbolic expression for the grand canonical partition function
    and determine the condition for normalizability.
    """
    beta, mu, epsilon = symbols('beta mu epsilon', real=True)
    N = symbols('N', integer=True)
    n0, n1 = symbols('n0 n1', integer=True)
    
    # Canonical partition function for N particles
    Z_N = Sum(exp(-beta * n1 * epsilon), (n0, 0, N)).subs(n1, N - n0)
    
    # Grand canonical partition function
    Omega_G = Sum(exp(beta * mu * N) * Z_N, (N, 0, oo))
    
    # For a 2-level system, we can simplify this
    # The canonical partition function Z_N can be written as:
    # Z_N = sum_{n0=0}^N exp(-beta * (N-n0) * epsilon)
    # = exp(-beta*N*epsilon) * sum_{n0=0}^N exp(beta*n0*epsilon)
    # = exp(-beta*N*epsilon) * (1-exp(beta*(N+1)*epsilon))/(1-exp(beta*epsilon))
    
    # The grand canonical partition function becomes:
    # Omega_G = sum_{N=0}^∞ exp(beta*mu*N) * Z_N
    
    # For this to be normalizable, we need the sum to converge
    # This requires: mu < epsilon
    
    # Let's try to evaluate the sum explicitly
    # First, rewrite Z_N in a more convenient form
    Z_N_simplified = Sum(exp(-beta * (N - n0) * epsilon), (n0, 0, N))
    
    # Substitute this into the grand canonical partition function
    Omega_G_simplified = Sum(exp(beta * mu * N) * Z_N_simplified, (N, 0, oo))
    
    # For convergence, we need: mu < epsilon
    convergence_condition = "μ < ε"
    
    return Omega_G, Omega_G_simplified, convergence_condition

def visualize_grand_canonical(epsilon=1.0):
    """
    Visualize the grand canonical partition function for different temperatures
    and chemical potentials.
    
    Args:
        epsilon: Energy of the excited state
    """
    # Temperature and chemical potential ranges
    T_values = np.linspace(0.1, 2.0, 30)
    mu_values = np.linspace(-5.0, 0.9, 30)  # Keep mu < epsilon for convergence
    
    beta_values = 1.0 / T_values
    
    plt.style.use('dark_background')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 3D plot of grand canonical partition function
    ax1 = fig.add_subplot(221, projection='3d')
    
    X, Y = np.meshgrid(T_values, mu_values)
    Z = np.zeros((len(mu_values), len(T_values)))
    
    for i, mu in enumerate(mu_values):
        for j, beta in enumerate(beta_values):
            # Skip calculations where mu is too close to epsilon (would diverge)
            if mu < epsilon - 0.1:
                Z[i, j] = grand_canonical_partition_function(beta, mu, epsilon)
            else:
                Z[i, j] = np.nan
    
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=True)
    
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Chemical Potential (μ)', fontsize=12)
    ax1.set_zlabel('Grand Canonical Partition Function (Ω)', fontsize=12)
    ax1.set_title('Grand Canonical Partition Function vs. T and μ', fontsize=14)
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 2D plot of grand canonical partition function vs chemical potential
    ax2 = fig.add_subplot(222)
    
    T_selected = [0.2, 0.5, 1.0, 2.0]
    beta_selected = [1.0/T for T in T_selected]
    
    for T, beta in zip(T_selected, beta_selected):
        Omega_values = []
        valid_mu = []
        
        for mu in mu_values:
            if mu < epsilon - 0.1:
                Omega_values.append(grand_canonical_partition_function(beta, mu, epsilon))
                valid_mu.append(mu)
        
        ax2.plot(valid_mu, Omega_values, linewidth=2.5, label=f'T = {T}')
    
    ax2.set_xlabel('Chemical Potential (μ)', fontsize=12)
    ax2.set_ylabel('Grand Canonical Partition Function (Ω)', fontsize=12)
    ax2.set_title('Grand Canonical Partition Function vs. Chemical Potential', fontsize=14)
    ax2.axvline(x=epsilon, color='red', linestyle='--', label=f'μ = ε = {epsilon}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Visualize the convergence condition
    ax3 = fig.add_subplot(223)
    
    # Create a region plot showing where the system is normalizable
    mu_grid = np.linspace(-5, 5, 100)
    T_grid = np.linspace(0.1, 5, 100)
    
    X, Y = np.meshgrid(T_grid, mu_grid)
    Z = np.zeros_like(X)
    
    # Color the region where mu < epsilon (normalizable)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = 1 if mu_grid[i] < epsilon else 0
    
    ax3.contourf(X, Y, Z, cmap='viridis', alpha=0.7)
    ax3.axhline(y=epsilon, color='red', linestyle='--', linewidth=2)
    
    ax3.set_xlabel('Temperature (T)', fontsize=12)
    ax3.set_ylabel('Chemical Potential (μ)', fontsize=12)
    ax3.set_title('Normalizability Condition: μ < ε', fontsize=14)
    ax3.text(2.5, epsilon/2, 'Normalizable Region\n(μ < ε)', 
             fontsize=12, ha='center', va='center', color='white')
    ax3.text(2.5, epsilon*1.5, 'Non-Normalizable Region\n(μ ≥ ε)', 
             fontsize=12, ha='center', va='center', color='white')
    
    # Display the symbolic expressions
    ax4 = fig.add_subplot(224)
    
    Omega_G, Omega_G_simplified, convergence_condition = symbolic_grand_canonical_partition_function()
    
    ax4.text(0.5, 0.8, f"$\\Omega_G = {latex(Omega_G)}$", fontsize=12, ha='center')
    ax4.text(0.5, 0.5, f"$\\Omega_G = {latex(Omega_G_simplified)}$", fontsize=12, ha='center')
    ax4.text(0.5, 0.2, f"Convergence condition: ${convergence_condition}$", 
             fontsize=14, ha='center', color='#ff9900')
    ax4.axis('off')
    ax4.set_title('Symbolic Expression for Grand Canonical Partition Function', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/f_grand_canonical/grand_canonical_partition.png', dpi=300, bbox_inches='tight')
    
    # Create a second figure to visualize the behavior near the critical point
    fig2 = plt.figure(figsize=(10, 8))
    
    # Plot the grand canonical partition function as mu approaches epsilon
    ax = fig2.add_subplot(111)
    
    # Fixed temperature
    T_fixed = 0.5
    beta_fixed = 1.0 / T_fixed
    
    # Chemical potentials approaching epsilon
    mu_critical = np.linspace(-5.0, epsilon - 0.01, 100)
    
    Omega_values = [grand_canonical_partition_function(beta_fixed, mu, epsilon) for mu in mu_critical]
    
    ax.plot(mu_critical, Omega_values, linewidth=3, color='#ff9900')
    ax.axvline(x=epsilon, color='red', linestyle='--', linewidth=2, label=f'μ = ε = {epsilon}')
    
    ax.set_xlabel('Chemical Potential (μ)', fontsize=14)
    ax.set_ylabel('Grand Canonical Partition Function (Ω)', fontsize=14)
    ax.set_title(f'Grand Canonical Partition Function as μ Approaches ε (T = {T_fixed})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Use a logarithmic scale for the y-axis to better visualize the divergence
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/f_grand_canonical/grand_canonical_critical.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

if __name__ == "__main__":
    visualize_grand_canonical() 