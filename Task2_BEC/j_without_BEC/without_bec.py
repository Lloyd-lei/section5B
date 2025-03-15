"""
Task 2j: Bose System without BEC

This script simulates a Bose system that does not experience Bose-Einstein Condensation (BEC).
It designs a system with specific degeneracy conditions that prevent the singular behaviors
associated with BEC in thermodynamic quantities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys
import os
import time

# Add the parent directory to the path to import the BoseSystem class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i_near_degenerate.near_degenerate import BoseSystem

def create_system_without_bec():
    """
    Create a Bose system that does not experience BEC.
    
    The key to preventing BEC is to have a high degeneracy for the ground state
    or a continuous density of states near the ground state. This prevents the
    macroscopic occupation of any single state.
    
    Returns:
        BoseSystem object
    """
    # Energy levels with increasing gaps
    energy_levels = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5]
    
    # High degeneracy for low energy levels prevents BEC
    # The degeneracy increases rapidly with energy level
    degeneracies = [100, 200, 400, 800, 1600, 3200]
    
    # Create the system with N ~ 10^5 bosons
    return BoseSystem(energy_levels, degeneracies, N_total=1e5)

def analyze_system_without_bec():
    """
    Analyze the thermodynamic properties of a Bose system without BEC.
    """
    # Create the system
    system = create_system_without_bec()
    
    # Simulate temperature dependence
    print("Simulating temperature dependence for system without BEC...")
    results = system.simulate_temperature_dependence(T_min=0.01, T_max=2.0, num_points=100)
    
    # Visualize the results
    system.visualize_results(results, title_prefix="Bose System without BEC: ")
    
    # Analyze the results to confirm absence of BEC
    analyze_bec_indicators(results, system)

def analyze_bec_indicators(results, system):
    """
    Analyze indicators of BEC in the simulation results.
    
    Args:
        results: Dictionary containing the simulation results
        system: BoseSystem object
    """
    # Check for signs of BEC
    T = results['T']
    mu = results['mu']
    n0 = results['n0']
    gradient = results['gradient']
    cv = results['cv']
    
    # In a system with BEC, we would expect:
    # 1. Chemical potential approaching the ground state energy at a critical temperature
    # 2. Sharp increase in ground state occupation below the critical temperature
    # 3. Peak in the occupation gradient at the critical temperature
    # 4. Lambda-like peak in the specific heat at the critical temperature
    
    # Check if chemical potential approaches ground state energy
    min_gap = min(mu - system.energy_levels[0])
    
    # Check for sharp changes in thermodynamic quantities
    max_gradient = max(gradient)
    max_cv = max(cv)
    
    # Print analysis
    print("\nAnalysis of BEC indicators:")
    print(f"Minimum gap between μ and ε₀: {min_gap:.6f}")
    print(f"Maximum occupation gradient: {max_gradient:.2e}")
    print(f"Maximum specific heat: {max_cv:.2e}")
    
    # Create a figure to visualize the analysis
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    
    # Plot the gap between chemical potential and ground state energy
    ax1 = fig.add_subplot(211)
    gap = mu - system.energy_levels[0]
    ax1.plot(T, gap, linewidth=2.5, color='#3498db')
    
    ax1.set_xlabel('Temperature (T)', fontsize=14)
    ax1.set_ylabel('μ - ε₀', fontsize=14)
    ax1.set_title('Gap between Chemical Potential and Ground State Energy', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Plot the ground state occupation fraction
    ax2 = fig.add_subplot(212)
    ax2.plot(T, n0 / system.N_total, linewidth=2.5, color='#2ecc71')
    
    ax2.set_xlabel('Temperature (T)', fontsize=14)
    ax2.set_ylabel('Ground State Occupation Fraction ⟨n₀⟩/N', fontsize=14)
    ax2.set_title('Ground State Occupation Fraction vs. Temperature', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/j_without_BEC/bec_indicators_analysis.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')
    
    # Create a figure to explain the physical mechanism
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    
    # Plot the density of states
    ax = fig.add_subplot(111)
    
    # Create a bar plot of the degeneracies
    ax.bar(system.energy_levels, system.degeneracies, width=0.05, alpha=0.7, color='#e74c3c')
    
    ax.set_xlabel('Energy (ε)', fontsize=14)
    ax.set_ylabel('Degeneracy (g)', fontsize=14)
    ax.set_title('Density of States in a System without BEC', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add annotations explaining the physics
    ax.text(0.5, 0.9, 'High degeneracy at low energy levels prevents BEC', 
            transform=ax.transAxes, fontsize=14, ha='center', color='white')
    
    ax.text(0.5, 0.85, 'Particles can occupy many states with similar energy', 
            transform=ax.transAxes, fontsize=14, ha='center', color='white')
    
    ax.text(0.5, 0.8, 'No macroscopic occupation of a single state occurs', 
            transform=ax.transAxes, fontsize=14, ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/j_without_BEC/physical_explanation.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

if __name__ == "__main__":
    # Analyze a Bose system without BEC
    analyze_system_without_bec() 