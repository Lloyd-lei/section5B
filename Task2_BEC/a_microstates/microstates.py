"""
Task 2a: Microstates of a 2-level Boson System

This script analyzes the microstates of a system with N indistinguishable bosons
in a 2-level system with ground state (energy 0) and first excited state (energy ε).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

def generate_microstates(N):
    """
    Generate all possible microstates for N indistinguishable bosons in a 2-level system.
    
    Args:
        N: Number of bosons
        
    Returns:
        List of tuples (n0, n1) where n0 is the number of bosons in ground state
        and n1 is the number in excited state
    """
    microstates = []
    for n0 in range(N + 1):
        n1 = N - n0
        microstates.append((n0, n1))
    
    return microstates

def calculate_energy(microstate, epsilon=1.0):
    """
    Calculate the energy of a microstate.
    
    Args:
        microstate: Tuple (n0, n1) representing the microstate
        epsilon: Energy of the excited state
        
    Returns:
        Total energy of the microstate
    """
    n0, n1 = microstate
    # Ground state has energy 0, excited state has energy epsilon
    return n1 * epsilon

def visualize_microstates(N, epsilon=1.0):
    """
    Create a visual representation of all microstates for N bosons.
    
    Args:
        N: Number of bosons
        epsilon: Energy of the excited state
    """
    microstates = generate_microstates(N)
    energies = [calculate_energy(state, epsilon) for state in microstates]
    
    # Create a figure with a dark background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Plot the microstates
    ax1 = plt.subplot(gs[0])
    ax1.set_xlim(-0.5, N + 0.5)
    ax1.set_ylim(-0.5, len(microstates) - 0.5)
    
    # Custom colors for visualization
    ground_color = '#3498db'  # Blue
    excited_color = '#e74c3c'  # Red
    
    # Draw each microstate
    for i, (n0, n1) in enumerate(microstates):
        # Draw ground state particles
        for j in range(n0):
            circle = plt.Circle((j, i), 0.3, color=ground_color, alpha=0.8)
            ax1.add_patch(circle)
        
        # Draw excited state particles
        for j in range(n1):
            circle = plt.Circle((n0 + j, i), 0.3, color=excited_color, alpha=0.8)
            ax1.add_patch(circle)
    
    # Add energy level indicators
    for i in range(len(microstates)):
        ax1.axhline(y=i, color='white', linestyle='--', alpha=0.2)
    
    # Add labels
    ax1.set_title(f'Microstates for {N} Indistinguishable Bosons in a 2-Level System', fontsize=16)
    ax1.set_ylabel('Microstate Index', fontsize=14)
    ax1.set_xlabel('Particle Index', fontsize=14)
    
    # Add a legend
    ground_patch = Rectangle((0, 0), 1, 1, color=ground_color)
    excited_patch = Rectangle((0, 0), 1, 1, color=excited_color)
    ax1.legend([ground_patch, excited_patch], ['Ground State (E=0)', f'Excited State (E={epsilon})'], 
               loc='upper right', fontsize=12)
    
    # Plot the energy distribution
    ax2 = plt.subplot(gs[1])
    ax2.bar(range(len(microstates)), energies, color='#f39c12', alpha=0.8)
    ax2.set_xlabel('Microstate Index', fontsize=14)
    ax2.set_ylabel('Energy', fontsize=14)
    ax2.set_title('Energy of Each Microstate', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/a_microstates/boson_microstates.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')
    
    # Print information about the microstates
    print(f"Microstates for {N} indistinguishable bosons in a 2-level system:")
    print("=" * 60)
    print(f"{'Microstate':^15} | {'Ground State':^15} | {'Excited State':^15} | {'Energy':^10}")
    print("-" * 60)
    
    for i, (n0, n1) in enumerate(microstates):
        print(f"{i:^15} | {n0:^15} | {n1:^15} | {energies[i]:^10.1f}")

if __name__ == "__main__":
    # Analyze microstates for 5 bosons
    N = 5
    visualize_microstates(N) 