"""
Task 2i: Near-degenerate Bose Systems

This script simulates a near-degenerate Bose system with N~10^5 bosons and calculates
various thermodynamic quantities including chemical potential, ground state occupation,
occupation gradient, and specific heat.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import cm
import time

class BoseSystem:
    """
    A class to simulate a Bose system with multiple energy levels.
    """
    
    def __init__(self, energy_levels, degeneracies, N_total=1e5):
        """
        Initialize the Bose system.
        
        Args:
            energy_levels: Array of energy levels
            degeneracies: Array of degeneracy factors for each energy level
            N_total: Total number of bosons in the system
        """
        self.energy_levels = np.array(energy_levels)
        self.degeneracies = np.array(degeneracies)
        self.N_total = N_total
        
        # Ensure arrays have the same length
        assert len(energy_levels) == len(degeneracies), "Energy levels and degeneracies must have the same length"
        
        # Sort energy levels and corresponding degeneracies
        idx = np.argsort(self.energy_levels)
        self.energy_levels = self.energy_levels[idx]
        self.degeneracies = self.degeneracies[idx]
        
        # Number of energy levels
        self.num_levels = len(energy_levels)
        
        print(f"Initialized Bose system with {self.num_levels} energy levels and {N_total:.1e} bosons")
        print(f"Energy levels: {self.energy_levels}")
        print(f"Degeneracies: {self.degeneracies}")
    
    def occupation_number(self, mu, T, level_idx):
        """
        Calculate the occupation number for a given energy level.
        
        Args:
            mu: Chemical potential
            T: Temperature
            level_idx: Index of the energy level
            
        Returns:
            Occupation number
        """
        # Bose-Einstein distribution
        # n_i = g_i / (exp((ε_i - μ)/kT) - 1)
        # where g_i is the degeneracy of level i
        
        epsilon = self.energy_levels[level_idx]
        g_i = self.degeneracies[level_idx]
        
        # Avoid division by zero or negative values
        if mu >= epsilon:
            return float('inf')  # Diverges when μ ≥ ε
        
        exponent = (epsilon - mu) / T
        
        # Handle large exponents to avoid numerical issues
        if exponent > 700:  # np.exp(700) is close to the limit of float64
            return g_i * np.exp(-exponent)
        
        return g_i / (np.exp(exponent) - 1)
    
    def total_occupation(self, mu, T):
        """
        Calculate the total occupation number across all energy levels.
        
        Args:
            mu: Chemical potential
            T: Temperature
            
        Returns:
            Total occupation number
        """
        return sum(self.occupation_number(mu, T, i) for i in range(self.num_levels))
    
    def find_chemical_potential(self, T):
        """
        Find the chemical potential that gives the correct total number of bosons.
        
        Args:
            T: Temperature
            
        Returns:
            Chemical potential
        """
        # Define the objective function: total occupation - N_total
        def objective(mu):
            return self.total_occupation(mu, T) - self.N_total
        
        # Initial guess for mu (slightly below the lowest energy level)
        initial_mu = self.energy_levels[0] - 0.1
        
        # Find the root of the objective function
        try:
            mu_solution = fsolve(objective, initial_mu)[0]
            return mu_solution
        except:
            # If solver fails, return a value slightly below the lowest energy level
            return self.energy_levels[0] - 1e-6
    
    def ground_state_occupation(self, T):
        """
        Calculate the ground state occupation at a given temperature.
        
        Args:
            T: Temperature
            
        Returns:
            Ground state occupation
        """
        mu = self.find_chemical_potential(T)
        return self.occupation_number(mu, T, 0)
    
    def occupation_gradient(self, T, delta_T=1e-4):
        """
        Calculate the negative gradient of ground state occupation with respect to temperature.
        
        Args:
            T: Temperature
            delta_T: Small temperature increment for numerical differentiation
            
        Returns:
            Negative gradient of ground state occupation
        """
        n0_T = self.ground_state_occupation(T)
        n0_T_plus = self.ground_state_occupation(T + delta_T)
        
        # Negative gradient
        return -(n0_T_plus - n0_T) / delta_T
    
    def specific_heat(self, T, delta_T=1e-4):
        """
        Calculate the specific heat at a given temperature.
        
        Args:
            T: Temperature
            delta_T: Small temperature increment for numerical differentiation
            
        Returns:
            Specific heat
        """
        # Calculate the average energy at T and T+delta_T
        mu_T = self.find_chemical_potential(T)
        mu_T_plus = self.find_chemical_potential(T + delta_T)
        
        E_T = sum(self.energy_levels[i] * self.occupation_number(mu_T, T, i) 
                 for i in range(self.num_levels))
        
        E_T_plus = sum(self.energy_levels[i] * self.occupation_number(mu_T_plus, T + delta_T, i) 
                      for i in range(self.num_levels))
        
        # Specific heat is the derivative of energy with respect to temperature
        return (E_T_plus - E_T) / delta_T
    
    def simulate_temperature_dependence(self, T_min=0.01, T_max=5.0, num_points=100):
        """
        Simulate the temperature dependence of various thermodynamic quantities.
        
        Args:
            T_min: Minimum temperature
            T_max: Maximum temperature
            num_points: Number of temperature points
            
        Returns:
            Dictionary containing the simulation results
        """
        # Temperature range
        T_values = np.linspace(T_min, T_max, num_points)
        
        # Initialize arrays for results
        mu_values = np.zeros(num_points)
        n0_values = np.zeros(num_points)
        log_n0_values = np.zeros(num_points)
        gradient_values = np.zeros(num_points)
        cv_values = np.zeros(num_points)
        
        # Simulate for each temperature
        start_time = time.time()
        for i, T in enumerate(T_values):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Simulating T = {T:.3f} ({i+1}/{num_points}), elapsed time: {elapsed:.2f}s")
            
            # Calculate thermodynamic quantities
            mu = self.find_chemical_potential(T)
            n0 = self.occupation_number(mu, T, 0)
            gradient = self.occupation_gradient(T)
            cv = self.specific_heat(T)
            
            # Store results
            mu_values[i] = mu
            n0_values[i] = n0
            log_n0_values[i] = np.log(n0) if n0 > 0 else float('nan')
            gradient_values[i] = gradient
            cv_values[i] = cv
        
        # Return results as a dictionary
        return {
            'T': T_values,
            'mu': mu_values,
            'n0': n0_values,
            'log_n0': log_n0_values,
            'gradient': gradient_values,
            'cv': cv_values
        }
    
    def visualize_results(self, results, title_prefix=""):
        """
        Visualize the simulation results.
        
        Args:
            results: Dictionary containing the simulation results
            title_prefix: Prefix for the plot titles
        """
        plt.style.use('dark_background')
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Plot chemical potential
        ax1 = fig.add_subplot(321)
        ax1.plot(results['T'], results['mu'], linewidth=2.5, color='#3498db')
        ax1.axhline(y=self.energy_levels[0], color='red', linestyle='--', 
                   label=f'ε₀ = {self.energy_levels[0]}')
        
        ax1.set_xlabel('Temperature (T)', fontsize=12)
        ax1.set_ylabel('Chemical Potential (μ)', fontsize=12)
        ax1.set_title(f'{title_prefix}Chemical Potential vs. Temperature', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot ground state occupation
        ax2 = fig.add_subplot(322)
        ax2.plot(results['T'], results['n0'], linewidth=2.5, color='#2ecc71')
        
        ax2.set_xlabel('Temperature (T)', fontsize=12)
        ax2.set_ylabel('Ground State Occupation ⟨n₀⟩', fontsize=12)
        ax2.set_title(f'{title_prefix}Ground State Occupation vs. Temperature', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Plot log of ground state occupation
        ax3 = fig.add_subplot(323)
        ax3.plot(results['T'], results['log_n0'], linewidth=2.5, color='#e74c3c')
        
        ax3.set_xlabel('Temperature (T)', fontsize=12)
        ax3.set_ylabel('log(⟨n₀⟩)', fontsize=12)
        ax3.set_title(f'{title_prefix}Log of Ground State Occupation vs. Temperature', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Plot negative gradient of ground state occupation
        ax4 = fig.add_subplot(324)
        ax4.plot(results['T'], results['gradient'], linewidth=2.5, color='#f39c12')
        
        ax4.set_xlabel('Temperature (T)', fontsize=12)
        ax4.set_ylabel('-∂⟨n₀⟩/∂T', fontsize=12)
        ax4.set_title(f'{title_prefix}Negative Gradient of Ground State Occupation', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Plot specific heat
        ax5 = fig.add_subplot(325)
        ax5.plot(results['T'], results['cv'], linewidth=2.5, color='#9b59b6')
        
        ax5.set_xlabel('Temperature (T)', fontsize=12)
        ax5.set_ylabel('Specific Heat (Cᵥ)', fontsize=12)
        ax5.set_title(f'{title_prefix}Specific Heat vs. Temperature', fontsize=14)
        ax5.grid(True, alpha=0.3)
        
        # Plot ground state occupation fraction
        ax6 = fig.add_subplot(326)
        ax6.plot(results['T'], results['n0'] / self.N_total, linewidth=2.5, color='#1abc9c')
        
        ax6.set_xlabel('Temperature (T)', fontsize=12)
        ax6.set_ylabel('Ground State Occupation Fraction ⟨n₀⟩/N', fontsize=12)
        ax6.set_title(f'{title_prefix}Ground State Occupation Fraction vs. Temperature', fontsize=14)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"Task2_BEC/i_near_degenerate/{'_'.join(title_prefix.lower().split())}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Reset style for future plots
        plt.style.use('default')

def create_near_degenerate_system():
    """
    Create a near-degenerate Bose system.
    
    Returns:
        BoseSystem object
    """
    # Energy levels with small gaps
    energy_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    
    # Degeneracies increase with energy level
    degeneracies = [1, 2, 3, 4, 5, 6]
    
    # Create the system with N ~ 10^5 bosons
    return BoseSystem(energy_levels, degeneracies, N_total=1e5)

if __name__ == "__main__":
    # Create a near-degenerate Bose system
    system = create_near_degenerate_system()
    
    # Simulate temperature dependence
    results = system.simulate_temperature_dependence(T_min=0.01, T_max=2.0, num_points=100)
    
    # Visualize the results
    system.visualize_results(results, title_prefix="Near-degenerate Bose System: ") 