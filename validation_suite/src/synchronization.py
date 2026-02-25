"""
Synchronization Parameter Evolution

This module implements the synchronization order parameter Φ and its
Landau-Ginzburg dynamics.

Φ(t_r, t_i) = exp[-α S(ρ(t_i)) - β Var(R_μν(t_r))]

References:
- Section 4.1-4.2 (Dilaton modulus vs effective synchronization parameter)
- Section 5 (Phase Transition Structure: Landau-Ginzburg Theory)
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Callable, Optional


class SynchronizationParameter:
    """
    Implements the synchronization order parameter Φ and its dynamics.
    
    The Landau-Ginzburg free energy is:
    F[Φ] = ∫ d³x [κ(∇Φ)²/2 + V_eff(Φ)]
    
    with effective potential:
    V_eff(Φ) = -μ²Φ²/2 + λΦ⁴/4
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 mu_squared: float = 1.0,
                 lambda_quartic: float = 1.0,
                 tau_Phi: float = 1.0):
        """
        Initialize synchronization parameter.
        
        Args:
            alpha: Entropy coupling coefficient
            beta: Curvature variance coupling coefficient
            mu_squared: Control parameter (>0 for synchronized phase)
            lambda_quartic: Quartic self-coupling (>0 for stability)
            tau_Phi: Relaxation timescale
        """
        self.alpha = alpha
        self.beta = beta
        self.mu_squared = mu_squared
        self.lambda_quartic = lambda_quartic
        self.tau_Phi = tau_Phi
        
        # Compute equilibrium value
        self.Phi_star = self.equilibrium_value()
    
    def equilibrium_value(self) -> float:
        """
        Compute equilibrium synchronization Φ_*.
        
        For μ² > 0: Φ_* = sqrt(μ²/λ)
        For μ² < 0: Φ_* = 0 (desynchronized phase)
        """
        if self.mu_squared > 0:
            return np.sqrt(self.mu_squared / self.lambda_quartic)
        else:
            return 0.0
    
    def effective_potential(self, Phi: float) -> float:
        """
        Compute V_eff(Φ) = -μ²Φ²/2 + λΦ⁴/4
        """
        return -0.5 * self.mu_squared * Phi**2 + 0.25 * self.lambda_quartic * Phi**4
    
    def potential_derivative(self, Phi: float) -> float:
        """
        Compute dV_eff/dΦ = -μ²Φ + λΦ³
        """
        return -self.mu_squared * Phi + self.lambda_quartic * Phi**3
    
    def second_derivative(self, Phi: float) -> float:
        """
        Compute d²V_eff/dΦ² = -μ² + 3λΦ²
        """
        return -self.mu_squared + 3 * self.lambda_quartic * Phi**2
    
    def relaxation_timescale(self, Phi: Optional[float] = None) -> float:
        """
        Compute relaxation timescale τ_Φ at given Φ.
        
        τ_Φ = γ / V''(Φ)
        
        where γ is the kinetic coefficient.
        """
        if Phi is None:
            Phi = self.Phi_star
        
        V_pp = self.second_derivative(Phi)
        if abs(V_pp) < 1e-10:
            return float('inf')
        
        return self.tau_Phi / V_pp
    
    def from_entropy_curvature(self, 
                                entropy: float, 
                                curvature_variance: float) -> float:
        """
        Compute Φ from von Neumann entropy and curvature variance.
        
        Φ = exp(-α S - β Var(R))
        
        Args:
            entropy: Von Neumann entropy S(ρ)
            curvature_variance: Covariant curvature variance Var(R_μν)
        
        Returns:
            Synchronization parameter Φ ∈ (0, 1]
        """
        exponent = -self.alpha * entropy - self.beta * curvature_variance
        return np.exp(exponent)
    
    def lambda_sync(self, Phi: float, Lambda_0: float = 1e-52, 
                    Delta_Lambda: float = 1e-52) -> float:
        """
        Compute synchronization vacuum term.
        
        Λ_sync(Φ) = Λ_0 + ΔΛ(1-Φ)²
        
        Args:
            Phi: Synchronization parameter
            Lambda_0: Base cosmological constant (m⁻²)
            Delta_Lambda: Synchronization correction amplitude
        
        Returns:
            Effective cosmological function Λ_sync
        """
        return Lambda_0 + Delta_Lambda * (1 - Phi)**2
    
    def relaxation_dynamics(self, Phi: float, t: float,
                            Phi_target: Callable[[float], float]) -> float:
        """
        Relaxation ODE: dΦ/dt = -(Φ - Φ_*)/τ_Φ
        
        Args:
            Phi: Current synchronization value
            t: Time
            Phi_target: Function returning target Φ_*(t)
        
        Returns:
            dΦ/dt
        """
        return -(Phi - Phi_target(t)) / self.tau_Phi
    
    def evolve(self, Phi_0: float, t_span: np.ndarray,
               Phi_target: Optional[Callable[[float], float]] = None) -> np.ndarray:
        """
        Evolve synchronization parameter over time.
        
        Args:
            Phi_0: Initial value
            t_span: Time array
            Phi_target: Target function Φ_*(t), default is constant Φ_*
        
        Returns:
            Array of Φ(t) values
        """
        if Phi_target is None:
            Phi_target = lambda t: self.Phi_star
        
        def ode(Phi, t):
            return self.relaxation_dynamics(Phi, t, Phi_target)
        
        Phi_evolution = odeint(ode, Phi_0, t_span)
        return Phi_evolution.flatten()
    
    def critical_exponents(self) -> dict:
        """
        Return mean-field Landau-Ginzburg critical exponents.
        """
        return {
            'beta': 0.5,    # Order parameter: Φ ~ |μ²|^β
            'gamma': 1.0,   # Susceptibility: χ ~ |μ²|^(-γ)
            'nu': 0.5,      # Correlation length: ξ ~ |μ²|^(-ν)
            'delta': 3.0,   # Critical isotherm: h ~ Φ^δ at T_c
        }


def phase_diagram(mu_range: np.ndarray, lambda_val: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase diagram of synchronization.
    
    Args:
        mu_range: Array of μ² values
        lambda_val: Fixed quartic coupling
    
    Returns:
        Tuple of (mu_values, Phi_star_values)
    """
    Phi_values = []
    
    for mu_sq in mu_range:
        if mu_sq > 0:
            Phi = np.sqrt(mu_sq / lambda_val)
        else:
            Phi = 0.0
        Phi_values.append(Phi)
    
    return mu_range, np.array(Phi_values)


if __name__ == "__main__":
    print("Testing synchronization parameter dynamics...")
    
    # Create parameter instance
    sync = SynchronizationParameter(mu_squared=1.0, lambda_quartic=1.0)
    
    print(f"Equilibrium Φ_*: {sync.Phi_star:.4f}")
    print(f"Critical exponents: {sync.critical_exponents()}")
    
    # Test evolution from desynchronized state
    t = np.linspace(0, 5, 100)
    Phi_0 = 0.1  # Start far from equilibrium
    Phi_t = sync.evolve(Phi_0, t)
    
    print(f"\nEvolution from Φ_0 = {Phi_0}:")
    print(f"  t=0: Φ = {Phi_t[0]:.4f}")
    print(f"  t=5: Φ = {Phi_t[-1]:.4f}")
    
    # Test Lambda_sync
    for Phi in [0.0, 0.5, 1.0]:
        Lambda = sync.lambda_sync(Phi)
        print(f"\nΛ_sync(Φ={Phi}): {Lambda:.2e} m⁻²")
