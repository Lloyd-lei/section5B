"""
Task 2d: Quantum Partition Function under the Canonical Ensemble

This script calculates and visualizes the quantum partition function for a system
of N indistinguishable bosons in a 2-level system under the canonical ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, exp, latex
from matplotlib import cm

def quantum_partition_function(N, beta, epsilon):
    """
    Calculate the quantum partition function for N indistinguishable bosons
    in a 2-level system under the canonical ensemble.
    
    Args:
        N: Number of bosons
        beta: Inverse temperature (1/kT)
        epsilon: Energy of the excited state
        
    Returns:
        Partition function value
    """
    # For quantum bosons, the partition function is simpler because
    # we don't have the binomial coefficient (no distinguishability)
    # Z = sum_{n0=0}^N exp(-beta * (N-n0) * epsilon)
    Z = 0
    for n0 in range(N + 1):
        n1 = N - n0
        Z += np.exp(-beta * n1 * epsilon)
    
    return Z

def probability_of_microstate(n0, N, beta, epsilon):
    """
    Calculate the probability of finding a particular microstate with n0 particles
    in the ground state for the quantum case.
    
    Args:
        n0: Number of particles in ground state
        N: Total number of particles
        beta: Inverse temperature (1/kT)
        epsilon: Energy of the excited state
        
    Returns:
        Probability of the microstate
    """
    n1 = N - n0
    Z = quantum_partition_function(N, beta, epsilon)
    return np.exp(-beta * n1 * epsilon) / Z

def symbolic_quantum_partition_function():
    """
    Derive the symbolic expression for the quantum partition function.
    """
    N, beta, epsilon = symbols('N beta epsilon', real=True)
    n0 = symbols('n0', integer=True)
    n1 = N - n0
    
    # Quantum partition function term
    term = exp(-beta * n1 * epsilon)
    
    # Sum over all possible values of n0
    Z = sp.Sum(term, (n0, 0, N))
    
    # Evaluate the sum if possible
    Z_evaluated = Z.doit()
    
    return Z, Z_evaluated

def visualize_quantum_partition_function(N_values, epsilon=1.0):
    """
    Visualize the quantum partition function for different temperatures and particle numbers.
    
    Args:
        N_values: List of particle numbers to visualize
        epsilon: Energy of the excited state
    """
    # Temperature range (in units where k_B = 1)
    T_values = np.linspace(0.1, 5.0, 100)
    beta_values = 1.0 / T_values
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 12))
    
    # 3D plot of quantum partition function
    ax1 = fig.add_subplot(221, projection='3d')
    
    X, Y = np.meshgrid(T_values, N_values)
    Z_quantum = np.zeros((len(N_values), len(T_values)))
    
    for i, N in enumerate(N_values):
        for j, beta in enumerate(beta_values):
            Z_quantum[i, j] = quantum_partition_function(N, beta, epsilon)
    
    surf = ax1.plot_surface(X, Y, Z_quantum, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Number of Particles (N)', fontsize=12)
    ax1.set_zlabel('Quantum Partition Function (Z)', fontsize=12)
    ax1.set_title('Quantum Partition Function vs. Temperature and Particle Number', fontsize=14)
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 2D plot of quantum partition function vs temperature for different N
    ax2 = fig.add_subplot(222)
    
    for N in N_values:
        Z_values = [quantum_partition_function(N, beta, epsilon) for beta in beta_values]
        ax2.plot(T_values, Z_values, linewidth=2.5, label=f'N = {N}')
    
    ax2.set_xlabel('Temperature (T)', fontsize=12)
    ax2.set_ylabel('Quantum Partition Function (Z)', fontsize=12)
    ax2.set_title('Quantum Partition Function vs. Temperature', fontsize=14)
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
    ax3.set_title(f'Quantum Microstate Probabilities (N = {N_specific})', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Compare classical vs quantum partition functions
    ax4 = fig.add_subplot(224)
    
    # Define classical partition function for comparison
    def classical_partition_function(N, beta, epsilon):
        Z = 0
        for n0 in range(N + 1):
            n1 = N - n0
            Z += sp.binomial(N, n0) * np.exp(-beta * n1 * epsilon)
        return Z
    
    N_compare = 10  # Choose a specific N for comparison
    
    Z_quantum_values = [quantum_partition_function(N_compare, beta, epsilon) for beta in beta_values]
    Z_classical_values = [classical_partition_function(N_compare, beta, epsilon) for beta in beta_values]
    
    ax4.plot(T_values, Z_quantum_values, linewidth=2.5, label='Quantum')
    ax4.plot(T_values, Z_classical_values, linewidth=2.5, label='Classical')
    ax4.plot(T_values, np.array(Z_classical_values) / np.array(Z_quantum_values), 
             linewidth=2.5, label='Classical/Quantum Ratio')
    
    ax4.set_xlabel('Temperature (T)', fontsize=12)
    ax4.set_ylabel('Partition Function (Z)', fontsize=12)
    ax4.set_title(f'Comparison of Quantum vs Classical Partition Functions (N = {N_compare})', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/d_quantum_partition/quantum_partition_function.png', dpi=300, bbox_inches='tight')
    
    # Create a second figure for the symbolic expressions
    fig2 = plt.figure(figsize=(10, 6))
    ax = fig2.add_subplot(111)
    
    Z, Z_evaluated = symbolic_quantum_partition_function()
    
    # Simplify the LaTeX expression to avoid rendering issues
    simple_latex = "Z = \\sum_{n_0=0}^{N} e^{-\\beta (N-n_0) \\epsilon}"
    
    ax.text(0.5, 0.7, f"${simple_latex}$", fontsize=14, ha='center')
    ax.text(0.5, 0.3, "For a 2-level system, this evaluates to:", fontsize=14, ha='center')
    ax.text(0.5, 0.1, "$Z = \\frac{1 - e^{-\\beta \\epsilon (N+1)}}{1 - e^{-\\beta \\epsilon}}$", fontsize=14, ha='center')
    ax.axis('off')
    ax.set_title('Symbolic Expression for Quantum Partition Function', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/d_quantum_partition/symbolic_quantum_partition.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

if __name__ == "__main__":
    # Visualize the quantum partition function for different particle numbers
    N_values = [5, 10, 15, 20]
    visualize_quantum_partition_function(N_values) 