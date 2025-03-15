"""
Task 2g & 2h: Particle Number under the Grand Canonical Ensemble

This script calculates and visualizes the average particle number in a system of bosons
using the grand canonical ensemble. It uses the grand potential to calculate the average
particle number and explores the behavior for large systems (N ~ 10^5).
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, exp, Sum, oo, diff, latex
from matplotlib import cm
from scipy.optimize import fsolve

def grand_canonical_partition_function(beta, mu, epsilon, N_max=1000):
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

def average_particle_number(beta, mu, epsilon, N_max=1000):
    """
    Calculate the average particle number using the grand potential.
    
    Args:
        beta: Inverse temperature (1/kT)
        mu: Chemical potential
        epsilon: Energy of the excited state
        N_max: Maximum number of particles to consider
        
    Returns:
        Average particle number
    """
    # Calculate the grand canonical partition function
    Omega_G = grand_canonical_partition_function(beta, mu, epsilon, N_max)
    
    # Calculate the average particle number using numerical differentiation
    # <N> = kB*T * d/dμ ln(Ω_G)
    delta_mu = 1e-6
    Omega_G_plus = grand_canonical_partition_function(beta, mu + delta_mu, epsilon, N_max)
    
    # Numerical derivative
    d_ln_Omega = (np.log(Omega_G_plus) - np.log(Omega_G)) / delta_mu
    
    # Average particle number
    avg_N = beta**(-1) * d_ln_Omega
    
    return avg_N

def symbolic_average_particle_number():
    """
    Derive the symbolic expression for the average particle number
    using the grand potential.
    """
    beta, mu, epsilon = symbols('beta mu epsilon', real=True)
    N = symbols('N', integer=True)
    n0, n1 = symbols('n0 n1', integer=True)
    
    # Canonical partition function for N particles
    Z_N = Sum(exp(-beta * n1 * epsilon), (n0, 0, N)).subs(n1, N - n0)
    
    # Grand canonical partition function
    Omega_G = Sum(exp(beta * mu * N) * Z_N, (N, 0, oo))
    
    # Grand potential
    Phi = -1/beta * sp.log(Omega_G)
    
    # Average particle number
    avg_N = -diff(Phi, mu)
    
    # For a 2-level system with mu < epsilon, we can derive a closed form
    # The average particle number is:
    # <N> = 1/(exp(-beta*mu) - 1) + 1/(exp(beta*(epsilon-mu)) - 1)
    
    # Ground state occupation
    n0_avg = 1/(sp.exp(-beta*mu) - 1)
    
    # Excited state occupation
    n1_avg = 1/(sp.exp(beta*(epsilon-mu)) - 1)
    
    # Total average particle number
    avg_N_closed = n0_avg + n1_avg
    
    return avg_N, avg_N_closed, n0_avg, n1_avg

def find_mu_for_target_N(target_N, beta, epsilon):
    """
    Find the chemical potential that gives a target average particle number.
    
    Args:
        target_N: Target average particle number
        beta: Inverse temperature (1/kT)
        epsilon: Energy of the excited state
        
    Returns:
        Chemical potential value
    """
    def objective(mu):
        # Calculate average N for a given mu
        avg_N = average_particle_number(beta, mu, epsilon)
        return avg_N - target_N
    
    # Initial guess for mu (must be less than epsilon)
    initial_mu = epsilon - 1.0
    
    # Find the root of the objective function
    mu_solution = fsolve(objective, initial_mu)[0]
    
    return mu_solution

def analytical_average_particle_number(beta, mu, epsilon):
    """
    Calculate the average particle number using the analytical formula.
    
    Args:
        beta: Inverse temperature (1/kT)
        mu: Chemical potential
        epsilon: Energy of the excited state
        
    Returns:
        Average particle number, ground state occupation, excited state occupation
    """
    # Ground state occupation
    n0_avg = 1/(np.exp(-beta*mu) - 1)
    
    # Excited state occupation
    n1_avg = 1/(np.exp(beta*(epsilon-mu)) - 1)
    
    # Total average particle number
    avg_N = n0_avg + n1_avg
    
    return avg_N, n0_avg, n1_avg

def visualize_particle_number(epsilon=1.0):
    """
    Visualize the average particle number for different temperatures
    and chemical potentials.
    
    Args:
        epsilon: Energy of the excited state
    """
    # Temperature and chemical potential ranges
    T_values = np.linspace(0.1, 5.0, 30)
    mu_values = np.linspace(-5.0, 0.9, 30)  # Keep mu < epsilon for convergence
    
    beta_values = 1.0 / T_values
    
    plt.style.use('dark_background')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 3D plot of average particle number
    ax1 = fig.add_subplot(221, projection='3d')
    
    X, Y = np.meshgrid(T_values, mu_values)
    Z = np.zeros((len(mu_values), len(T_values)))
    
    for i, mu in enumerate(mu_values):
        for j, beta in enumerate(beta_values):
            # Skip calculations where mu is too close to epsilon (would diverge)
            if mu < epsilon - 0.1:
                # Use analytical formula for efficiency
                avg_N, _, _ = analytical_average_particle_number(beta, mu, epsilon)
                Z[i, j] = avg_N
            else:
                Z[i, j] = np.nan
    
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Chemical Potential (μ)', fontsize=12)
    ax1.set_zlabel('Average Particle Number ⟨N⟩', fontsize=12)
    ax1.set_title('Average Particle Number vs. T and μ', fontsize=14)
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 2D plot of average particle number vs chemical potential
    ax2 = fig.add_subplot(222)
    
    T_selected = [0.2, 0.5, 1.0, 2.0]
    beta_selected = [1.0/T for T in T_selected]
    
    for T, beta in zip(T_selected, beta_selected):
        avg_N_values = []
        valid_mu = []
        
        for mu in mu_values:
            if mu < epsilon - 0.1:
                avg_N, _, _ = analytical_average_particle_number(beta, mu, epsilon)
                avg_N_values.append(avg_N)
                valid_mu.append(mu)
        
        ax2.plot(valid_mu, avg_N_values, linewidth=2.5, label=f'T = {T}')
    
    ax2.set_xlabel('Chemical Potential (μ)', fontsize=12)
    ax2.set_ylabel('Average Particle Number ⟨N⟩', fontsize=12)
    ax2.set_title('Average Particle Number vs. Chemical Potential', fontsize=14)
    ax2.axvline(x=epsilon, color='red', linestyle='--', label=f'μ = ε = {epsilon}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot ground state and excited state occupations
    ax3 = fig.add_subplot(223)
    
    # Fixed temperature
    T_fixed = 0.5
    beta_fixed = 1.0 / T_fixed
    
    n0_values = []
    n1_values = []
    valid_mu = []
    
    for mu in mu_values:
        if mu < epsilon - 0.1:
            _, n0, n1 = analytical_average_particle_number(beta_fixed, mu, epsilon)
            n0_values.append(n0)
            n1_values.append(n1)
            valid_mu.append(mu)
    
    ax3.plot(valid_mu, n0_values, linewidth=2.5, label='Ground State ⟨n₀⟩')
    ax3.plot(valid_mu, n1_values, linewidth=2.5, label='Excited State ⟨n₁⟩')
    ax3.plot(valid_mu, np.array(n0_values) + np.array(n1_values), 
             linewidth=2.5, linestyle='--', label='Total ⟨N⟩')
    
    ax3.set_xlabel('Chemical Potential (μ)', fontsize=12)
    ax3.set_ylabel('Occupation Number', fontsize=12)
    ax3.set_title(f'Ground and Excited State Occupations (T = {T_fixed})', fontsize=14)
    ax3.axvline(x=epsilon, color='red', linestyle='--', label=f'μ = ε = {epsilon}')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Display the symbolic expressions
    ax4 = fig.add_subplot(224)
    
    avg_N, avg_N_closed, n0_avg, n1_avg = symbolic_average_particle_number()
    
    ax4.text(0.5, 0.8, f"$\\langle N \\rangle = {latex(avg_N_closed)}$", fontsize=12, ha='center')
    ax4.text(0.5, 0.6, f"$\\langle n_0 \\rangle = {latex(n0_avg)}$", fontsize=12, ha='center')
    ax4.text(0.5, 0.4, f"$\\langle n_1 \\rangle = {latex(n1_avg)}$", fontsize=12, ha='center')
    ax4.text(0.5, 0.2, "Condition: $\\mu < \\epsilon$", fontsize=14, ha='center', color='#ff9900')
    ax4.axis('off')
    ax4.set_title('Symbolic Expressions for Average Particle Numbers', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/g_h_particle_number/particle_number.png', dpi=300, bbox_inches='tight')
    
    # Create a second figure for large N systems
    fig2 = plt.figure(figsize=(12, 10))
    
    # Target large N values
    N_large = [1e3, 1e4, 1e5]
    T_range = np.linspace(0.1, 5.0, 50)
    beta_range = 1.0 / T_range
    
    # Plot chemical potential vs temperature for fixed large N
    ax1 = fig2.add_subplot(211)
    
    for target_N in N_large:
        mu_values = []
        
        for beta in beta_range:
            # Find the chemical potential that gives the target N
            try:
                mu = find_mu_for_target_N(target_N, beta, epsilon)
                mu_values.append(mu)
            except:
                # If the solver fails, use a value close to epsilon
                mu_values.append(epsilon - 0.01)
        
        ax1.plot(T_range, mu_values, linewidth=2.5, label=f'N = {target_N:.0e}')
    
    ax1.axhline(y=epsilon, color='red', linestyle='--', label=f'μ = ε = {epsilon}')
    
    ax1.set_xlabel('Temperature (T)', fontsize=14)
    ax1.set_ylabel('Chemical Potential (μ)', fontsize=14)
    ax1.set_title('Chemical Potential vs. Temperature for Large N Systems', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot ground state occupation fraction vs temperature for fixed large N
    ax2 = fig2.add_subplot(212)
    
    for target_N in N_large:
        n0_fraction = []
        
        for beta in beta_range:
            try:
                # Find the chemical potential that gives the target N
                mu = find_mu_for_target_N(target_N, beta, epsilon)
                
                # Calculate ground state occupation
                _, n0, _ = analytical_average_particle_number(beta, mu, epsilon)
                
                # Calculate fraction
                n0_fraction.append(n0 / target_N)
            except:
                n0_fraction.append(np.nan)
        
        ax2.plot(T_range, n0_fraction, linewidth=2.5, label=f'N = {target_N:.0e}')
    
    ax2.set_xlabel('Temperature (T)', fontsize=14)
    ax2.set_ylabel('Ground State Occupation Fraction ⟨n₀⟩/N', fontsize=14)
    ax2.set_title('Ground State Occupation Fraction vs. Temperature for Large N Systems', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/g_h_particle_number/large_N_systems.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

if __name__ == "__main__":
    visualize_particle_number() 