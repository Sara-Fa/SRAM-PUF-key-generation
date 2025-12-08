#!/usr/bin/env python3
"""
Plot theoretical local stability ε(q) as function of reliability q.

Based on Bernardini Figure 8 upper part:
- (0.99, 0.005)-stable stabilizer
- SRAM with q distributed according to (30a) with λ = 0.05
- Local stability ε(q) ≈ Φ(√K |q - 1/2| / √q(1-q)) (equation 33)
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

# Add current directory to Python path for local imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def plot_epsilon_vs_q(K: int = 17, lambda_val: float = 0.05, stability_target: float = 0.99, instability_target: float = 0.005, figsize: tuple = (7, 4), output_dir: pathlib.Path = None):
    """
    Plot theoretical local stability ε(q) as function of reliability q.
    
    Args:
        K: Number of reads (default 17)
        lambda_val: Lambda parameter for distribution (default 0.05)
        stability_target: Target stability threshold (default 0.99)
        instability_target: Target instability threshold (default 0.005)
        figsize: Figure size
        output_dir: Directory to save plots (defaults to script_dir/results/plots)
    """
    
    # Set up default output directory if not provided
    if output_dir is None:
        # Save under central plotting results directory
        output_dir = pathlib.Path(__file__).parent / "plotting" / ".." / "results" / "plots"
        output_dir = output_dir.resolve()
    
    # Create q values from 0 to 1
    q_values = np.linspace(0.001, 0.999, 1000)
    
    # Compute local stability ε(q) ≈ Φ(√K |q - 1/2| / √q(1-q))
    # This is equation (33) from Bernardini's paper
    epsilon_values = norm.cdf(np.sqrt(K) * np.abs(q_values - 0.5) / np.sqrt(q_values * (1 - q_values)))
    
    # Create plot
    _, ax = plt.subplots(figsize=figsize)
    
    # Plot ε(q) vs q
    ax.plot(q_values, epsilon_values, 'b-', linewidth=2,
            label=f'ε(q) ≈ Φ(√{K} |q-1/2| / √q(1-q))')
    
    # Add horizontal line at target stability
    ax.axhline(y=stability_target, color='red', linestyle='--', linewidth=2,
               label=f'Target Stability = {stability_target}')
    
    # Find intersection points with target stability line
    # Solve: ε(q) = stability_target
    diff = epsilon_values - stability_target
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) >= 2:
        # Get the two intersection points
        q1_idx = sign_changes[0]
        q2_idx = sign_changes[1]
        q1 = q_values[q1_idx]
        q2 = q_values[q2_idx]
        
        # Add vertical lines at intersection points
        ax.axvline(q1, color='green', linestyle=':', alpha=0.7,
                   label=f'q₁ = {q1:.3f}')
        ax.axvline(q2, color='green', linestyle=':', alpha=0.7,
                   label=f'q₂ = {q2:.3f}')
        
        # Shade the region where ε(q) ≥ stability_target
        ax.fill_between(q_values, epsilon_values, stability_target, 
                       where=(epsilon_values >= stability_target), 
                       alpha=0.3, color='green',
                       label=f'ε(q) ≥ {stability_target} region')
        
        print(f"Reliability bounds for {stability_target} stability:")
        print(f"q₁ = {q1:.4f}")
        print(f"q₂ = {q2:.4f}")
        print(f"Acceptable range: [{q1:.4f}, {q2:.4f}]")
    
    # Set axis properties
    ax.set_xlabel('Reliability q', fontsize=14)
    ax.set_ylabel('Local Stability ε(q)', fontsize=14)
    ax.set_title(f'Local Stability ε(q) vs Reliability q\n'
                f'({stability_target}, {instability_target})-stable stabilizer, K={K}, λ={lambda_val}', 
                fontsize=14)
    
    # Set axis limits - focus on the interesting region
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.0)  # Focus on the upper part where most values are
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, which='both')
    
    # Add legend
    ax.legend(fontsize=12, frameon=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    output_file = output_dir / f'epsilon_vs_q_K{K}_lambda{lambda_val}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    # Also save as PDF
    output_file_pdf = output_dir / f'epsilon_vs_q_K{K}_lambda{lambda_val}.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved plot (PDF): {output_file_pdf}")
    
    plt.show()
    
    # Print key values
    print("\nKey Values:")
    print(f"K = {K}")
    print(f"λ = {lambda_val}")
    print(f"Target stability = {stability_target}")
    print(f"ε(0.5) = {norm.cdf(np.sqrt(K) * np.abs(0.5 - 0.5) / np.sqrt(0.5 * 0.5)):.6f}")
    print(f"ε(0.1) = {norm.cdf(np.sqrt(K) * np.abs(0.1 - 0.5) / np.sqrt(0.1 * 0.9)):.6f}")
    print(f"ε(0.9) = {norm.cdf(np.sqrt(K) * np.abs(0.9 - 0.5) / np.sqrt(0.9 * 0.1)):.6f}")
    print(f"ε(0.01) = {norm.cdf(np.sqrt(K) * np.abs(0.01 - 0.5) / np.sqrt(0.01 * 0.99)):.6f}")
    print(f"ε(0.99) = {norm.cdf(np.sqrt(K) * np.abs(0.99 - 0.5) / np.sqrt(0.99 * 0.01)):.6f}")


def main():
    """Main execution function."""
    
    # Set up paths
    output_dir = pathlib.Path(__file__).parent / "plotting" / ".." / "results" / "plots"
    output_dir = output_dir.resolve()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # (0.99, 0.005)-stable stabilizer parameters
    # These can be easily changed here
    K = 68                    # Number of reads
    lambda_val = 0.1        # Lambda parameter for SRAM distribution
    stability_target = 0.999999  # Target stability (0.99 for 99% stability)
    instability_target = 5*10**(-2) # Target instability (0.005 for 0.5% instability)
    
    print("Plotting theoretical ε(q) vs q")
    print(f"Parameters: K={K}, λ={lambda_val}")
    print(f"({stability_target}, {instability_target})-stable stabilizer")
    
    # Generate plot
    plot_epsilon_vs_q(K=K, lambda_val=lambda_val, stability_target=stability_target, instability_target=instability_target, figsize=(7, 4), output_dir=output_dir)


if __name__ == "__main__":
    main()