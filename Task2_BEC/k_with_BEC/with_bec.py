"""
Task 2k: Bose System with BEC

This script simulates a Bose system that experiences Bose-Einstein Condensation (BEC).
It designs a system with specific degeneracy conditions that lead to singular behaviors
in thermodynamic quantities, identifies the critical temperature, and analyzes the
scaling laws of heat capacity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
import sys
import os
import time

# Add the parent directory to the path to import the BoseSystem class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i_near_degenerate.near_degenerate import BoseSystem

def create_system_with_bec():
    """
    Create a Bose system that experiences BEC.
    
    The key to enabling BEC is to have a low degeneracy for the ground state
    and a large energy gap between the ground state and excited states.
    This promotes the macroscopic occupation of the ground state below a critical temperature.
    
    Returns:
        BoseSystem object
    """
    # Energy levels with a large gap between ground state and excited states
    energy_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Low degeneracy for ground state, higher for excited states
    degeneracies = [1, 5, 10, 15, 20, 25]
    
    # Create the system with N ~ 10^5 bosons
    return BoseSystem(energy_levels, degeneracies, N_total=1e5)

def analyze_system_with_bec():
    """
    Analyze the thermodynamic properties of a Bose system with BEC.
    """
    # Create the system
    system = create_system_with_bec()
    
    # Simulate temperature dependence with more points around the expected critical region
    print("Simulating temperature dependence for system with BEC...")
    
    # First, do a coarse simulation to identify the approximate critical region
    coarse_results = system.simulate_temperature_dependence(T_min=0.01, T_max=2.0, num_points=50)
    
    # Identify the approximate critical temperature from the coarse simulation
    T_c_approx = identify_critical_temperature(coarse_results)
    print(f"Approximate critical temperature: T_c ≈ {T_c_approx:.4f}")
    
    # Now do a finer simulation around the critical region
    T_min = max(0.01, T_c_approx - 0.3)
    T_max = min(2.0, T_c_approx + 0.3)
    fine_results = system.simulate_temperature_dependence(T_min=T_min, T_max=T_max, num_points=100)
    
    # Combine the results
    results = coarse_results
    
    # Visualize the results
    system.visualize_results(results, title_prefix="Bose System with BEC: ")
    
    # Analyze the results to confirm presence of BEC and identify critical properties
    analyze_bec_indicators(results, system)
    
    # Analyze the scaling laws near the critical temperature
    analyze_scaling_laws(fine_results, system, T_c_approx)

def identify_critical_temperature(results):
    """
    Identify the critical temperature for BEC from simulation results.
    
    Args:
        results: Dictionary containing the simulation results
        
    Returns:
        Estimated critical temperature
    """
    # The critical temperature can be identified by:
    # 1. The temperature where the chemical potential approaches the ground state energy
    # 2. The temperature where the occupation gradient peaks
    # 3. The temperature where the specific heat has a lambda-like peak
    
    T = results['T']
    mu = results['mu']
    gradient = results['gradient']
    cv = results['cv']
    
    # Method 1: Find where chemical potential approaches ground state energy
    # Calculate the rate of change of (mu - epsilon_0)
    epsilon_0 = 0.0  # Ground state energy
    mu_gap = mu - epsilon_0
    d_mu_gap = np.gradient(mu_gap, T)
    T_c_mu = T[np.argmin(d_mu_gap)]
    
    # Method 2: Find where occupation gradient peaks
    T_c_gradient = T[np.argmax(gradient)]
    
    # Method 3: Find where specific heat peaks
    T_c_cv = T[np.argmax(cv)]
    
    # Average the estimates
    T_c = (T_c_mu + T_c_gradient + T_c_cv) / 3
    
    print(f"Critical temperature estimates:")
    print(f"  From chemical potential: T_c ≈ {T_c_mu:.4f}")
    print(f"  From occupation gradient: T_c ≈ {T_c_gradient:.4f}")
    print(f"  From specific heat: T_c ≈ {T_c_cv:.4f}")
    print(f"  Average estimate: T_c ≈ {T_c:.4f}")
    
    return T_c

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
    
    # In a system with BEC, we expect:
    # 1. Chemical potential approaching the ground state energy at a critical temperature
    # 2. Sharp increase in ground state occupation below the critical temperature
    # 3. Peak in the occupation gradient at the critical temperature
    # 4. Lambda-like peak in the specific heat at the critical temperature
    
    # Identify the critical temperature
    T_c = identify_critical_temperature(results)
    
    # Check if chemical potential approaches ground state energy
    min_gap = min(mu - system.energy_levels[0])
    
    # Check for sharp changes in thermodynamic quantities
    max_gradient = max(gradient)
    max_cv = max(cv)
    
    # Print analysis
    print("\nAnalysis of BEC indicators:")
    print(f"Critical temperature: T_c ≈ {T_c:.4f}")
    print(f"Minimum gap between μ and ε₀: {min_gap:.6f}")
    print(f"Maximum occupation gradient: {max_gradient:.2e}")
    print(f"Maximum specific heat: {max_cv:.2e}")
    
    # Create a figure to visualize the analysis
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 12))
    
    # Plot the gap between chemical potential and ground state energy
    ax1 = fig.add_subplot(221)
    gap = mu - system.energy_levels[0]
    ax1.plot(T, gap, linewidth=2.5, color='#3498db')
    ax1.axvline(x=T_c, color='red', linestyle='--', label=f'T_c ≈ {T_c:.4f}')
    
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('μ - ε₀', fontsize=12)
    ax1.set_title('Gap between Chemical Potential and Ground State Energy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot the ground state occupation fraction
    ax2 = fig.add_subplot(222)
    ax2.plot(T, n0 / system.N_total, linewidth=2.5, color='#2ecc71')
    ax2.axvline(x=T_c, color='red', linestyle='--', label=f'T_c ≈ {T_c:.4f}')
    
    ax2.set_xlabel('Temperature (T)', fontsize=12)
    ax2.set_ylabel('Ground State Occupation Fraction ⟨n₀⟩/N', fontsize=12)
    ax2.set_title('Ground State Occupation Fraction vs. Temperature', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot the occupation gradient
    ax3 = fig.add_subplot(223)
    ax3.plot(T, gradient, linewidth=2.5, color='#e74c3c')
    ax3.axvline(x=T_c, color='red', linestyle='--', label=f'T_c ≈ {T_c:.4f}')
    
    ax3.set_xlabel('Temperature (T)', fontsize=12)
    ax3.set_ylabel('-∂⟨n₀⟩/∂T', fontsize=12)
    ax3.set_title('Negative Gradient of Ground State Occupation', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot the specific heat
    ax4 = fig.add_subplot(224)
    ax4.plot(T, cv, linewidth=2.5, color='#9b59b6')
    ax4.axvline(x=T_c, color='red', linestyle='--', label=f'T_c ≈ {T_c:.4f}')
    
    ax4.set_xlabel('Temperature (T)', fontsize=12)
    ax4.set_ylabel('Specific Heat (Cᵥ)', fontsize=12)
    ax4.set_title('Specific Heat vs. Temperature', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/k_with_BEC/bec_indicators_analysis.png', dpi=300, bbox_inches='tight')
    
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
    ax.set_title('Density of States in a System with BEC', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add annotations explaining the physics
    ax.text(0.5, 0.9, 'Low degeneracy at ground state enables BEC', 
            transform=ax.transAxes, fontsize=14, ha='center', color='white')
    
    ax.text(0.5, 0.85, 'Large energy gap promotes ground state occupation', 
            transform=ax.transAxes, fontsize=14, ha='center', color='white')
    
    ax.text(0.5, 0.8, f'Below T_c ≈ {T_c:.4f}, macroscopic occupation of ground state occurs', 
            transform=ax.transAxes, fontsize=14, ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/k_with_BEC/physical_explanation.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')

def analyze_scaling_laws(results, system, T_c):
    """
    Analyze the scaling laws of thermodynamic quantities near the critical temperature.
    
    Args:
        results: Dictionary containing the simulation results
        system: BoseSystem object
        T_c: Critical temperature
    """
    T = results['T']
    n0 = results['n0']
    cv = results['cv']
    
    # Create a figure for scaling law analysis
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Ground state occupation scaling: n0/N ~ (1 - T/T_c)^α for T < T_c
    ax1 = fig.add_subplot(221)
    
    # Filter data for T < T_c
    mask_below = T < T_c
    T_below = T[mask_below]
    n0_fraction_below = (n0 / system.N_total)[mask_below]
    
    # Calculate reduced temperature
    t_below = 1 - T_below / T_c
    
    # Fit power law: n0/N ~ t^α
    def power_law(t, alpha, A):
        return A * t**alpha
    
    # Avoid log(0) by filtering out very small values
    mask_nonzero = t_below > 1e-6
    t_fit = t_below[mask_nonzero]
    n0_fit = n0_fraction_below[mask_nonzero]
    
    if len(t_fit) > 2:  # Need at least 3 points for a meaningful fit
        params, _ = curve_fit(power_law, t_fit, n0_fit)
        alpha, A = params
        
        # Plot the data and fit
        ax1.scatter(t_below, n0_fraction_below, color='#3498db', label='Simulation data')
        t_smooth = np.linspace(min(t_fit), max(t_fit), 100)
        ax1.plot(t_smooth, power_law(t_smooth, alpha, A), 'r-', 
                linewidth=2, label=f'Fit: n₀/N ~ t^{alpha:.3f}')
        
        ax1.set_xlabel('Reduced Temperature t = 1 - T/T_c', fontsize=12)
        ax1.set_ylabel('Ground State Occupation Fraction ⟨n₀⟩/N', fontsize=12)
        ax1.set_title('Ground State Occupation Scaling Law (T < T_c)', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Use log-log scale to visualize power law
        ax2 = fig.add_subplot(222)
        ax2.loglog(t_below, n0_fraction_below, 'o', color='#3498db', label='Simulation data')
        ax2.loglog(t_smooth, power_law(t_smooth, alpha, A), 'r-', 
                  linewidth=2, label=f'Fit: n₀/N ~ t^{alpha:.3f}')
        
        ax2.set_xlabel('log(t)', fontsize=12)
        ax2.set_ylabel('log(⟨n₀⟩/N)', fontsize=12)
        ax2.set_title('Log-Log Plot of Ground State Occupation Scaling', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # 2. Specific heat scaling: Cv ~ |T - T_c|^(-α) near T_c
    ax3 = fig.add_subplot(223)
    
    # Filter data near T_c
    mask_near = np.abs(T - T_c) / T_c < 0.2  # Within 20% of T_c
    T_near = T[mask_near]
    cv_near = cv[mask_near]
    
    # Calculate |T - T_c|
    t_near = np.abs(T_near - T_c)
    
    # Separate data for T < T_c and T > T_c
    mask_below_near = T_near < T_c
    mask_above_near = T_near > T_c
    
    t_below_near = t_near[mask_below_near]
    cv_below_near = cv_near[mask_below_near]
    
    t_above_near = t_near[mask_above_near]
    cv_above_near = cv_near[mask_above_near]
    
    # Fit power law: Cv ~ |T - T_c|^(-α)
    def inverse_power_law(t, alpha, A):
        return A * t**(-alpha)
    
    # Avoid division by zero by filtering out very small values
    mask_nonzero_below = t_below_near > 1e-6
    t_fit_below = t_below_near[mask_nonzero_below]
    cv_fit_below = cv_below_near[mask_nonzero_below]
    
    mask_nonzero_above = t_above_near > 1e-6
    t_fit_above = t_above_near[mask_nonzero_above]
    cv_fit_above = cv_above_near[mask_nonzero_above]
    
    # Fit for T < T_c
    if len(t_fit_below) > 2:
        try:
            params_below, _ = curve_fit(inverse_power_law, t_fit_below, cv_fit_below)
            alpha_below, A_below = params_below
            
            # Plot the data and fit
            ax3.scatter(t_below_near, cv_below_near, color='#e74c3c', label='T < T_c')
            t_smooth_below = np.linspace(min(t_fit_below), max(t_fit_below), 100)
            ax3.plot(t_smooth_below, inverse_power_law(t_smooth_below, alpha_below, A_below), 
                    'r-', linewidth=2, label=f'Fit (T < T_c): Cv ~ |t|^(-{alpha_below:.3f})')
        except:
            print("Could not fit power law for T < T_c")
    
    # Fit for T > T_c
    if len(t_fit_above) > 2:
        try:
            params_above, _ = curve_fit(inverse_power_law, t_fit_above, cv_fit_above)
            alpha_above, A_above = params_above
            
            # Plot the data and fit
            ax3.scatter(t_above_near, cv_above_near, color='#2ecc71', label='T > T_c')
            t_smooth_above = np.linspace(min(t_fit_above), max(t_fit_above), 100)
            ax3.plot(t_smooth_above, inverse_power_law(t_smooth_above, alpha_above, A_above), 
                    'g-', linewidth=2, label=f'Fit (T > T_c): Cv ~ |t|^(-{alpha_above:.3f})')
        except:
            print("Could not fit power law for T > T_c")
    
    ax3.set_xlabel('|T - T_c|', fontsize=12)
    ax3.set_ylabel('Specific Heat (Cᵥ)', fontsize=12)
    ax3.set_title('Specific Heat Scaling Near Critical Temperature', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Log-log plot of specific heat scaling
    ax4 = fig.add_subplot(224)
    
    if len(t_fit_below) > 2:
        try:
            ax4.loglog(t_below_near, cv_below_near, 'o', color='#e74c3c', label='T < T_c')
            ax4.loglog(t_smooth_below, inverse_power_law(t_smooth_below, alpha_below, A_below), 
                      'r-', linewidth=2, label=f'Fit (T < T_c): Cv ~ |t|^(-{alpha_below:.3f})')
        except:
            pass
    
    if len(t_fit_above) > 2:
        try:
            ax4.loglog(t_above_near, cv_above_near, 'o', color='#2ecc71', label='T > T_c')
            ax4.loglog(t_smooth_above, inverse_power_law(t_smooth_above, alpha_above, A_above), 
                      'g-', linewidth=2, label=f'Fit (T > T_c): Cv ~ |t|^(-{alpha_above:.3f})')
        except:
            pass
    
    ax4.set_xlabel('log(|T - T_c|)', fontsize=12)
    ax4.set_ylabel('log(Cᵥ)', fontsize=12)
    ax4.set_title('Log-Log Plot of Specific Heat Scaling', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2_BEC/k_with_BEC/scaling_laws.png', dpi=300, bbox_inches='tight')
    
    # Reset style for future plots
    plt.style.use('default')
    
    # Print scaling law results
    print("\nScaling Law Analysis:")
    try:
        print(f"Ground state occupation scaling: n₀/N ~ (1 - T/T_c)^{alpha:.3f} for T < T_c")
    except:
        print("Could not determine ground state occupation scaling")
    
    try:
        print(f"Specific heat scaling for T < T_c: Cv ~ |T - T_c|^(-{alpha_below:.3f})")
    except:
        print("Could not determine specific heat scaling for T < T_c")
    
    try:
        print(f"Specific heat scaling for T > T_c: Cv ~ |T - T_c|^(-{alpha_above:.3f})")
    except:
        print("Could not determine specific heat scaling for T > T_c")

if __name__ == "__main__":
    # Analyze a Bose system with BEC
    analyze_system_with_bec() 