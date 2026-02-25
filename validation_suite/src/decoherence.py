"""
Gravitational Decoherence: Diósi-Penrose Rates with Synchronization

This module computes gravitational decoherence rates and compares
standard Diósi-Penrose predictions with Φ-modified rates.

References:
- Section 7 (Quantum matter and gravitational decoherence)
- Section 10.4 (Decoherence vs Diósi--Penrose parameter sweeps)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# Physical constants (SI units)
G = 6.67430e-11      # Gravitational constant (m³ kg⁻¹ s⁻²)
HBAR = 1.054571817e-34  # Reduced Planck constant (J s)
C = 299792458.0      # Speed of light (m/s)


def diosi_penrose_rate(M: float, R: float, Delta_x: float) -> float:
    """
    Compute standard Diósi-Penrose gravitational decoherence rate.
    
    Γ_DP = G M² / (ℏ R) × f_DP(Δx/R)
    
    where f_DP is a geometric factor depending on object shape.
    For a localized object with Δx << R, f_DP ≈ Δx/R.
    
    Args:
        M: Object mass (kg)
        R: Characteristic radius (m)
        Delta_x: Superposition separation (m)
    
    Returns:
        Decoherence rate Γ_DP (s⁻¹)
    """
    # Geometric factor for spherical object
    ratio = Delta_x / R
    if ratio < 1:
        f_DP = ratio  # Linear regime
    else:
        f_DP = 1.0    # Saturation
    
    Gamma = (G * M**2) / (HBAR * R) * f_DP
    return Gamma


def modified_decoherence_rate(M: float, R: float, Delta_x: float,
                               Phi: float, p: float = 1.0) -> float:
    """
    Compute Φ-modified gravitational decoherence rate.
    
    Γ_grav = Φ^p × Γ_DP
    
    Args:
        M: Object mass (kg)
        R: Characteristic radius (m)
        Delta_x: Superposition separation (m)
        Phi: Synchronization parameter
        p: Exponent (p ≥ 1)
    
    Returns:
        Modified decoherence rate (s⁻¹)
    """
    Gamma_DP = diosi_penrose_rate(M, R, Delta_x)
    return (Phi ** p) * Gamma_DP


def ppn_constraint(Phi: float, test: str = 'light_deflection') -> Tuple[float, float]:
    """
    Compute post-Newtonian test constraints on synchronization.
    
    In weak-field regime, Φ → 1 to recover GR.
    Deviations from GR set bounds on 1 - Φ.
    
    Args:
        Phi: Synchronization parameter
        test: Type of PPN test
            - 'light_deflection': VLBI measurements
            - 'shapiro_delay': Cassini tracking
            - 'perihelion': Mercury precession
    
    Returns:
        Tuple of (predicted_deviation, observed_bound)
    """
    # PPN parameter γ bounds (2σ)
    constraints = {
        'light_deflection': {
            'gamma_obs': 1.0,
            'sigma': 3e-4,  # VLBI precision
        },
        'shapiro_delay': {
            'gamma_obs': 1.0,
            'sigma': 2.3e-5,  # Cassini
        },
        'perihelion': {
            'gamma_obs': 1.0,
            'sigma': 3e-3,  # Mercury
        }
    }
    
    if test not in constraints:
        raise ValueError(f"Unknown test: {test}")
    
    data = constraints[test]
    
    # In our model, Φ < 1 induces γ deviation
    # Simplified: δγ ~ (1 - Φ)
    predicted_deviation = 1 - Phi
    observed_bound = 2 * data['sigma']
    
    return predicted_deviation, observed_bound


def ppn_lower_bound(test: str = 'light_deflection') -> float:
    """
    Compute lower bound on Φ from PPN test.
    
    Requires Φ > 1 - observed_bound
    
    Returns:
        Lower bound on Φ
    """
    _, bound = ppn_constraint(1.0, test)
    return 1.0 - bound


class ExperimentalConfiguration:
    """
    Represents an experimental setup for decoherence tests.
    """
    
    def __init__(self, name: str, M: float, R: float, Delta_x: float,
                 Gamma_max: Optional[float] = None):
        """
        Args:
            name: Experiment name
            M: Mass (kg)
            R: Radius (m)
            Delta_x: Superposition separation (m)
            Gamma_max: Upper bound on observed decoherence rate (s⁻¹)
        """
        self.name = name
        self.M = M
        self.R = R
        self.Delta_x = Delta_x
        self.Gamma_max = Gamma_max
    
    def dp_rate(self) -> float:
        """Compute standard DP rate."""
        return diosi_penrose_rate(self.M, self.R, self.Delta_x)
    
    def modified_rate(self, Phi: float, p: float = 1.0) -> float:
        """Compute Φ-modified rate."""
        return modified_decoherence_rate(self.M, self.R, self.Delta_x, Phi, p)
    
    def consistent_with_bound(self, Phi: float, p: float = 1.0) -> bool:
        """Check if modified rate is below experimental bound."""
        if self.Gamma_max is None:
            return True
        return self.modified_rate(Phi, p) < self.Gamma_max


# Standard experimental configurations
EXPERIMENTS = [
    ExperimentalConfiguration(
        name="Matter-wave interferometry",
        M=1e-23,     # ~10000 amu
        R=5e-9,      # 5 nm
        Delta_x=1e-6, # 1 μm
        Gamma_max=1e3  # Observed coherence time ~ms
    ),
    ExperimentalConfiguration(
        name="LIGO mirrors",
        M=40.0,      # 40 kg
        R=0.17,      # 17 cm
        Delta_x=1e-18, # Quantum noise level
        Gamma_max=1e-3  # Sensitive to decoherence
    ),
    ExperimentalConfiguration(
        name="Optomechanical resonator",
        M=1e-12,     # 1 pg
        R=1e-6,      # 1 μm
        Delta_x=1e-12, # pm-scale
        Gamma_max=1e2  # kHz linewidth
    ),
    ExperimentalConfiguration(
        name="NV center spin",
        M=2e-26,     # Carbon atom
        R=1e-10,     # Atomic scale
        Delta_x=1e-10, # Atomic separation
        Gamma_max=1e6  # MHz T2 times
    ),
    ExperimentalConfiguration(
        name="LISA Pathfinder",
        M=2.0,       # 2 kg
        R=0.023,     # 2.3 cm cube
        Delta_x=1e-15, # fm-scale
        Gamma_max=1e-4  # Sub-Hz requirements
    ),
]


def parameter_sweep(M_range: np.ndarray, 
                    R: float = 1e-6,
                    Delta_x: float = 1e-6,
                    Phi_values: List[float] = [0.5, 0.9, 1.0]) -> Dict:
    """
    Perform parameter sweep over mass range.
    
    Args:
        M_range: Array of masses (kg)
        R: Fixed radius (m)
        Delta_x: Fixed separation (m)
        Phi_values: List of Φ values to compute
    
    Returns:
        Dictionary with results
    """
    results = {'M': M_range}
    
    for Phi in Phi_values:
        rates = []
        for M in M_range:
            rate = modified_decoherence_rate(M, R, Delta_x, Phi)
            rates.append(rate)
        results[f'Gamma_Phi={Phi}'] = np.array(rates)
    
    return results


if __name__ == "__main__":
    print("Testing gravitational decoherence calculations...")
    
    # Test standard DP rate
    M = 1e-20  # 10 fg
    R = 1e-7   # 100 nm
    Delta_x = 1e-6  # 1 μm
    
    Gamma_DP = diosi_penrose_rate(M, R, Delta_x)
    print(f"\nStandard DP rate for M={M:.0e} kg, R={R:.0e} m, Δx={Delta_x:.0e} m:")
    print(f"  Γ_DP = {Gamma_DP:.2e} s⁻¹")
    print(f"  τ_DP = {1/Gamma_DP:.2e} s")
    
    # Test Φ-modified rate
    for Phi in [0.5, 0.9, 1.0]:
        Gamma_mod = modified_decoherence_rate(M, R, Delta_x, Phi)
        print(f"  Γ(Φ={Phi}) = {Gamma_mod:.2e} s⁻¹")
    
    # PPN constraints
    print("\nPPN lower bounds on Φ:")
    for test in ['light_deflection', 'shapiro_delay', 'perihelion']:
        bound = ppn_lower_bound(test)
        print(f"  {test}: Φ > {bound:.6f}")
    
    # Experimental configurations
    print("\nExperimental DP rates:")
    for exp in EXPERIMENTS:
        rate = exp.dp_rate()
        consistent = exp.consistent_with_bound(1.0)
        status = "✓" if consistent else "✗"
        print(f"  {exp.name}: Γ_DP = {rate:.2e} s⁻¹ {status}")
