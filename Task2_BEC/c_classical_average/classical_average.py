"""
Task 2c: Classical Average Particle Number under the Canonical Ensemble

This script calculates and visualizes the average number of particles in the ground state
and excited state for a system of N indistinguishable bosons in a 2-level system under
the classical canonical ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, exp, binomial, diff, latex
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
        Z += binomial(N, n0) * np.exp(-beta * n1 * epsilon)
    
    return Z

def average_ground_state_particles_classical(N, beta, epsilon):
    """
    Calculate the average number of particles in the ground state
    for the classical canonical ensemble.
    
    Args:
        N: Total number of particles
        beta: Inverse temperature (1/kT)
        epsilon: Energy of the excited state
        
    Returns:
        Average number of particles in the ground state
    """
    Z = classical_partition_function(N, beta, epsilon)
    avg_n0 = 0
    
    for n0 in range(N + 1):
        n1 = N - n0
        prob = binomial(N, n0) * np.exp(-beta * n1 * epsilon) / Z
        avg_n0 += n0 * prob
    
    return avg_n0

def average_excited_state_particles_classical(N, beta, epsilon):
    """
    Calculate the average number of particles in the excited state
    for the classical canonical ensemble.
    
    Args:
        N: Total number of particles
        beta: Inverse temperature (1/kT)
        epsilon: Energy of the excited state
        
    Returns:
        Average number of particles in the excited state
    """
    avg_n0 = average_ground_state_particles_classical(N, beta, epsilon)
    return N - avg_n0

def symbolic_average_particles():
    """
    Derive the symbolic expressions for the average number of particles
    in the ground and excited states.
    """
    N, beta, epsilon = symbols('N beta epsilon', real=True)
    n0 = symbols('n0', integer=True)
    n1 = N - n0
    
    # Classical partition function term
    term = binomial(N, n0) * exp(-beta * n1 * epsilon)
    
    # Sum over all possible values of n0
    Z_C = sp.Sum(term, (n0, 0, N)).doit()
    
    # Average number of particles in ground state
    avg_n0_term = n0 * term / Z_C
    avg_n0 = sp.Sum(avg_n0_term, (n0, 0, N)).doit()
    
    # Average number of particles in excited state
    avg_n1 = N - avg_n0
    
    # Alternative derivation using derivative of partition function
    # avg_n1 = epsilon * diff(sp.log(Z_C), epsilon)
    
    return avg_n0, avg_n1

def visualize_average_particles(N_values, epsilon=1.0):
    """
    Visualize the average number of particles in ground and excited states
    for different temperatures and particle numbers.
    
    Args:
        N_values: List of particle numbers to visualize
        epsilon: Energy of the excited state
    """
    # Temperature range (in units where k_B = 1)
    T_values = np.linspace(0.1, 5.0, 100)
    beta_values = 1.0 / T_values
    
    plt.style.use('dark_background')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 3D plot of average ground state particles
    ax1 = fig.add_subplot(221, projection='3d')
    
    X, Y = np.meshgrid(T_values, N_values)
    Z_ground = np.zeros((len(N_values), len(T_values)))
    Z_excited = np.zeros((len(N_values), len(T_values)))
    
    for i, N in enumerate(N_values):
        for j, beta in enumerate(beta_values):
            Z_ground[i, j] = average_ground_state_particles_classical(N, beta, epsilon)
            Z_excited[i, j] = N - Z_ground[i, j]
    
    surf1 = ax1.plot_surface(X, Y, Z_ground, cmap=cm.viridis, linewidth=0, antialiased=True)
    
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Number of Particles (N)', fontsize=12)
    ax1.set_zlabel('Average Ground State Particles ⟨n₀⟩', fontsize=12)
    ax1.set_title('Average Ground State Particles vs. Temperature and N', fontsize=14)
    
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # 3D plot of average excited state particles
    ax2 = fig.add_subplot(222, projection='3d')
    
    surf2 = ax2.plot_surface(X, Y, Z_excited, cmap=cm.plasma, linewidth=0, antialiased=True)
    
    ax2.set_xlabel('Temperature (T)', fontsize=12)
    ax2.set_ylabel('Number of Particles (N)', fontsize=12)
    ax2.set_zlabel('Average Excited State Particles ⟨nₑ⟩', fontsize=12)
    ax2.set_title('Average Excited State Particles vs. Temperature and N', fontsize=14)
    
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 2D plots for specific N values
    ax3 = fig.add_subplot(223)
    
    for N in N_values:
        n0_values = [average_ground_state_particles_classical(N, beta, epsilon) for beta in beta_values]
        n1_values = [N - n0 for n0 in n0_values]
        
        ax3.plot(T_values, n0_values, linewidth=2.5, label=f'⟨n₀⟩, N = {N}')
        ax3.plot(T_values, n1_values, linewidth=2.5, linestyle='--', label=f'⟨nₑ⟩, N = {N}')
    
    ax3.set_xlabel('Temperature (T)', fontsize=12)
    ax3.set_ylabel('Average Particle Number', fontsize=12)
    ax3.set_title('Average Particle Numbers vs. Temperature', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Fractional occupation
    ax4 = fig.add_subplot(224)
    
    for N in N_values:
        n0_values = [average_ground_state_particles_classical(N, beta, epsilon) for beta in beta_values]
        n0_fraction = [n0/N for n0 in n0_values]
        n1_fraction = [1 - n0/N for n0 in n0_values]
        
        ax4.plot(T_values, n0_fraction, linewidth=2.5, label=f'⟨n₀⟩/N, N = {N}')
        ax4.plot(T_values, n1_fraction, linewidth=2.5, linestyle='--', label=f'⟨nₑ⟩/N, N = {N}')
    
    ax4.set_xlabel('Temperature (T)', fontsize=12)
    ax4.set_ylabel('Fractional Occupation', fontsize=12)
    ax4.set_title('Fractional Occupation vs. Temperature', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/c_classical_average/classical_average_particles.png', dpi=300, bbox_inches='tight')
    
    # Create a second figure for the symbolic expressions
    fig2 = plt.figure(figsize=(10, 6))
    ax = fig2.add_subplot(111)
    
    avg_n0, avg_n1 = symbolic_average_particles()
    
    ax.text(0.5, 0.7, f"$\\langle n_0 \\rangle_C = {latex(avg_n0)}$", fontsize=14, ha='center')
    ax.text(0.5, 0.3, f"$\\langle n_\\epsilon \\rangle_C = {latex(avg_n1)}$", fontsize=14, ha='center')
    ax.axis('off')
    ax.set_title('Symbolic Expressions for Average Particle Numbers', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/c_classical_average/symbolic_average_particles.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

if __name__ == "__main__":
    # Visualize the average particle numbers for different particle numbers
    N_values = [5, 10, 20, 50]
    visualize_average_particles(N_values) 