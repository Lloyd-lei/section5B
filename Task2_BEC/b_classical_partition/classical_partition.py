"""
Task 2b: Classical Partition Function under the Canonical Ensemble

This script calculates and visualizes the classical partition function for a system
of N indistinguishable bosons in a 2-level system under the canonical ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, exp, binomial, factorial, latex
from matplotlib import cm

def classical_partition_function(N, beta, epsilon):
    """
    Calculate the classical partition function for N indistinguishable bosons
    in a 2-level system under the canonical ensemble.
    
    Args:
        N: Number of bosons
        beta: Inverse temperature (1/kT)
        epsilon: Energy of the excited state
        
    Returns:
        Partition function value
    """
    Z = 0
    for n0 in range(N + 1):
        n1 = N - n0
        # Classical partition function includes binomial coefficient
        # Z_C = sum_{n0=0}^N binomial(N, n0) * exp(-beta * n1 * epsilon)
        Z += binomial(N, n0) * np.exp(-beta * n1 * epsilon)
    
    return Z

def probability_of_microstate(n0, N, beta, epsilon):
    """
    Calculate the probability of finding a particular microstate with n0 particles
    in the ground state.
    
    Args:
        n0: Number of particles in ground state
        N: Total number of particles
        beta: Inverse temperature (1/kT)
        epsilon: Energy of the excited state
        
    Returns:
        Probability of the microstate
    """
    n1 = N - n0
    Z = classical_partition_function(N, beta, epsilon)
    return binomial(N, n0) * np.exp(-beta * n1 * epsilon) / Z

def symbolic_partition_function():
    """
    Derive the symbolic expression for the classical partition function.
    """
    N, beta, epsilon = symbols('N beta epsilon', real=True)
    n0 = symbols('n0', integer=True)
    n1 = N - n0
    
    # Classical partition function term
    term = binomial(N, n0) * exp(-beta * n1 * epsilon)
    
    # Sum over all possible values of n0
    Z_C = sp.Sum(term, (n0, 0, N))
    
    # Evaluate the sum if possible
    Z_C_evaluated = Z_C.doit()
    
    return Z_C, Z_C_evaluated

def visualize_partition_function(N_values, epsilon=1.0):
    """
    Visualize the classical partition function for different temperatures and particle numbers.
    
    Args:
        N_values: List of particle numbers to visualize
        epsilon: Energy of the excited state
    """
    # Temperature range (in units where k_B = 1)
    T_values = np.linspace(0.1, 5.0, 100)
    beta_values = 1.0 / T_values
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    
    # 3D plot of partition function
    ax1 = fig.add_subplot(221, projection='3d')
    
    X, Y = np.meshgrid(T_values, N_values)
    Z = np.zeros((len(N_values), len(T_values)))
    
    for i, N in enumerate(N_values):
        for j, beta in enumerate(beta_values):
            Z[i, j] = classical_partition_function(N, beta, epsilon)
    
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=True)
    
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Number of Particles (N)', fontsize=12)
    ax1.set_zlabel('Partition Function (Z)', fontsize=12)
    ax1.set_title('Classical Partition Function vs. Temperature and Particle Number', fontsize=14)
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 2D plot of partition function vs temperature for different N
    ax2 = fig.add_subplot(222)
    
    for N in N_values:
        Z_values = [classical_partition_function(N, beta, epsilon) for beta in beta_values]
        ax2.plot(T_values, Z_values, linewidth=2.5, label=f'N = {N}')
    
    ax2.set_xlabel('Temperature (T)', fontsize=12)
    ax2.set_ylabel('Partition Function (Z)', fontsize=12)
    ax2.set_title('Classical Partition Function vs. Temperature', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot of microstate probabilities for a specific N
    N_specific = N_values[-1]  # Use the largest N value
    ax3 = fig.add_subplot(223)
    
    # Calculate probabilities for different temperatures
    T_selected = [0.5, 1.0, 2.0, 5.0]
    
    for T in T_selected:
        beta = 1.0 / T
        probs = [probability_of_microstate(n0, N_specific, beta, epsilon) 
                 for n0 in range(N_specific + 1)]
        ax3.plot(range(N_specific + 1), probs, 'o-', linewidth=2, label=f'T = {T}')
    
    ax3.set_xlabel('Number of Particles in Ground State (n₀)', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title(f'Microstate Probabilities (N = {N_specific})', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Display the symbolic expression
    ax4 = fig.add_subplot(224)
    Z_C, Z_C_evaluated = symbolic_partition_function()
    
    ax4.text(0.5, 0.7, f"$Z_C = {latex(Z_C)}$", fontsize=14, ha='center')
    ax4.text(0.5, 0.3, f"$Z_C = {latex(Z_C_evaluated)}$", fontsize=14, ha='center')
    ax4.axis('off')
    ax4.set_title('Symbolic Expression for Classical Partition Function', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/b_classical_partition/classical_partition_function.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

if __name__ == "__main__":
    # Visualize the partition function for different particle numbers
    N_values = [5, 10, 15, 20]
    visualize_partition_function(N_values) 