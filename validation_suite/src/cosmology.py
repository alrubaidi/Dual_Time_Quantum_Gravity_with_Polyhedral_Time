"""
Cosmological Applications: FRW Universe with Synchronization

This module implements the modified Friedmann equation with
synchronization-dependent cosmological term Λ_sync(Φ).

References:
- Section 9.2 (Cosmological emergence and epoch transitions)
- Section 6.5 (Minimal reductions: FRW cosmology)
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from typing import Tuple, Callable, Optional, Dict


# Physical constants
C = 299792458.0              # Speed of light (m/s)
G = 6.67430e-11             # Gravitational constant (m³ kg⁻¹ s⁻²)
HBAR = 1.054571817e-34      # Reduced Planck constant (J s)
H0 = 67.4e3 / 3.086e22      # Hubble constant (s⁻¹), 67.4 km/s/Mpc
OMEGA_M0 = 0.315          # Present matter density parameter
OMEGA_L0 = 0.685          # Present dark energy density parameter
OMEGA_R0 = 9.2e-5         # Present radiation density parameter


class FRWSynchronization:
    """
    Solve modified Friedmann equation with synchronization.
    
    H²(t_r) = (8πG/3)ρ(t_r, t_i) + Λ_sync(Φ)/3
    
    where Λ_sync(Φ) = Λ_0 + ΔΛ(1-Φ)²
    """
    
    def __init__(self,
                 Lambda_0: float = 1.1e-52,  # m⁻² (observed ~10⁻⁵² m⁻²)
                 Delta_Lambda: float = 1e-52,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 tau_Phi: float = 1e17):  # ~Hubble time in seconds
        """
        Initialize cosmological model.
        
        Args:
            Lambda_0: Base cosmological constant (m⁻²)
            Delta_Lambda: Synchronization correction amplitude
            alpha: Entropy coupling
            beta: Curvature variance coupling
            tau_Phi: Synchronization relaxation timescale (s)
        """
        self.Lambda_0 = Lambda_0
        self.Delta_Lambda = Delta_Lambda
        self.alpha = alpha
        self.beta = beta
        self.tau_Phi = tau_Phi
        
        # Convert Lambda to energy density
        self.rho_Lambda_0 = (C**4 / (8 * np.pi * G)) * Lambda_0
    
    def lambda_sync(self, Phi: float) -> float:
        """Compute Λ_sync(Φ) = Λ_0 + ΔΛ(1-Φ)²"""
        return self.Lambda_0 + self.Delta_Lambda * (1 - Phi)**2
    
    def rho_sync(self, Phi: float) -> float:
        """Convert Λ_sync to energy density."""
        return (C**4 / (8 * np.pi * G)) * self.lambda_sync(Phi)
    
    def w_sync(self, a: float, Phi: float, dPhi_dlna: float) -> float:
        """
        Compute effective equation of state for synchronization sector.
        
        w_sync = -1 - (1/3) d ln(ρ_sync) / d ln(a)
        
        For slowly varying Φ, w_sync ≈ -1 (vacuum-like).
        """
        if Phi < 1e-10:
            return -1.0
        
        # d ln(ρ_sync)/d ln(a) = d ln(Λ_sync)/d ln(a)
        # = (2ΔΛ(1-Φ)/Λ_sync) × (-dΦ/d ln(a))
        numerator = 2 * self.Delta_Lambda * (1 - Phi)
        denominator = self.lambda_sync(Phi)
        
        if abs(denominator) < 1e-60:
            return -1.0
        
        d_ln_rho_d_ln_a = -(numerator / denominator) * dPhi_dlna
        
        return -1.0 - d_ln_rho_d_ln_a / 3.0
    
    def hubble_rate(self, a: float, Phi: float) -> float:
        """
        Compute Hubble rate H(a, Φ).
        
        H² = H₀² [Ω_r0/a⁴ + Ω_m0/a³ + ρ_sync(Φ)/ρ_c0]
        """
        rho_c0 = 3 * H0**2 / (8 * np.pi * G) * C**2
        
        # Component densities
        rho_r = OMEGA_R0 * rho_c0 / a**4  # Radiation
        rho_m = OMEGA_M0 * rho_c0 / a**3  # Matter
        rho_sync = self.rho_sync(Phi)
        
        rho_total = rho_r + rho_m + rho_sync
        
        H_squared = (8 * np.pi * G / 3) * rho_total / C**2
        return np.sqrt(max(0, H_squared))
    
    def synchronization_target(self, a: float, entropy_model: str = 'constant') -> float:
        """
        Compute target Φ_* as function of scale factor.
        
        Args:
            a: Scale factor
            entropy_model: Model for entropy evolution
                - 'constant': Φ_* = 1 (pure synchronized)
                - 'early_high': High entropy at early times
                - 'late_deviation': Small late-time desynchronization
        """
        if entropy_model == 'constant':
            return 1.0
        
        elif entropy_model == 'early_high':
            # High entropy at early times (a << 1)
            return 1.0 - 0.1 * np.exp(-a)
        
        elif entropy_model == 'late_deviation':
            # Small deviation at late times (a > 1)
            if a > 1:
                return 1.0 - 0.01 * (a - 1)
            return 1.0
        
        else:
            return 1.0
    
    def equations(self, y: np.ndarray, t: float, 
                  entropy_model: str = 'constant') -> np.ndarray:
        """
        System of ODEs for (a, Φ) evolution.
        
        da/dt = a × H(a, Φ)
        dΦ/dt = -(Φ - Φ_*) / τ_Φ
        """
        a, Phi = y
        Phi = np.clip(Phi, 0.0, 1.0)
        
        H = self.hubble_rate(a, Phi)
        Phi_star = self.synchronization_target(a, entropy_model)
        
        da_dt = a * H
        dPhi_dt = -(Phi - Phi_star) / self.tau_Phi
        
        return np.array([da_dt, dPhi_dt])
    
    def evolve(self, a_0: float, Phi_0: float, t_span: np.ndarray,
               entropy_model: str = 'constant') -> Dict[str, np.ndarray]:
        """
        Evolve the cosmological model.
        
        Args:
            a_0: Initial scale factor
            Phi_0: Initial synchronization
            t_span: Time array (cosmic time in seconds)
            entropy_model: Model for target synchronization
        
        Returns:
            Dictionary with 't', 'a', 'Phi', 'H', 'Lambda_sync'
        """
        y0 = np.array([a_0, Phi_0])
        
        solution = odeint(self.equations, y0, t_span, args=(entropy_model,))
        
        a_arr = solution[:, 0]
        Phi_arr = solution[:, 1]
        
        # Compute derived quantities
        H_arr = np.array([self.hubble_rate(a, Phi) 
                         for a, Phi in zip(a_arr, Phi_arr)])
        Lambda_arr = np.array([self.lambda_sync(Phi) for Phi in Phi_arr])
        
        return {
            't': t_span,
            'a': a_arr,
            'Phi': Phi_arr,
            'H': H_arr,
            'Lambda_sync': Lambda_arr
        }
    
    def redshift_to_scale(self, z: float) -> float:
        """Convert redshift to scale factor: a = 1/(1+z)"""
        return 1.0 / (1.0 + z)
    
    def scale_to_redshift(self, a: float) -> float:
        """Convert scale factor to redshift: z = 1/a - 1"""
        return 1.0 / a - 1.0


class BlackHoleSynchronization:
    """
    Synchronization profile near black hole horizon.
    
    Models Φ(r) as a domain wall structure across the horizon.
    """
    
    def __init__(self, M_solar: float = 10.0, alpha: float = 0.5, beta: float = 0.5):
        """
        Args:
            M_solar: Black hole mass in solar masses
            alpha, beta: Synchronization coupling parameters
        """
        M_sun = 1.989e30  # kg
        self.M = M_solar * M_sun
        self.alpha = alpha
        self.beta = beta
        
        # Schwarzschild radius
        self.r_s = 2 * G * self.M / C**2
        
        # Surface gravity
        self.kappa = C**4 / (4 * G * self.M)
    
    def phi_profile(self, r: float) -> float:
        """
        Synchronization as function of radial coordinate.
        
        Φ(r) = exp[-α S(r) - β Var(κ(r))]
        
        Near horizon: entropy ~πr_s²/ℓ_P², curvature variance ~ r_s/r
        """
        if r < self.r_s:
            return 0.0  # Inside horizon: complete desynchronization
        
        # Entropy contribution (increases near horizon)
        entropy_factor = 1.0 - np.exp(-(r - self.r_s) / self.r_s)
        
        # Curvature variance (increases near horizon)
        curvature_factor = (r - self.r_s) / r
        
        exponent = -self.alpha * (1 - entropy_factor) - self.beta * (1 - curvature_factor)
        
        return np.exp(exponent)
    
    def lambda_sync_profile(self, r: float, Lambda_0: float = 1e-52,
                            Delta_Lambda: float = 1e-52) -> float:
        """Λ_sync(r) from Φ(r)"""
        Phi = self.phi_profile(r)
        return Lambda_0 + Delta_Lambda * (1 - Phi)**2
    
    def hawking_power_modified(self, Phi: Optional[float] = None) -> float:
        """
        Compute Φ-modified Hawking power.
        
        P_Hawking(Φ) = P_standard × f(Φ)
        
        where P_standard ∝ 1/M²
        """
        # Standard Hawking power (in Planck units scaled)
        # P = ℏc⁶ / (15360 π G² M²)
        P_standard = (HBAR * C**6) / (15360 * np.pi * G**2 * self.M**2)
        
        if Phi is None:
            Phi = self.phi_profile(3 * self.r_s)  # Evaluate at 3r_s
        
        # Modification factor
        f_Phi = Phi  # Linear suppression at low Φ
        
        return P_standard * f_Phi
    
    def radial_profile(self, r_min: float = 1.01, r_max: float = 10.0,
                       n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute radial profiles of Φ and Λ_sync.
        
        Args:
            r_min, r_max: Radial range in units of r_s
            n_points: Number of points
        
        Returns:
            Dictionary with 'r', 'Phi', 'Lambda_sync'
        """
        r_arr = np.linspace(r_min * self.r_s, r_max * self.r_s, n_points)
        
        Phi_arr = np.array([self.phi_profile(r) for r in r_arr])
        Lambda_arr = np.array([self.lambda_sync_profile(r) for r in r_arr])
        
        return {
            'r': r_arr,
            'r_over_rs': r_arr / self.r_s,
            'Phi': Phi_arr,
            'Lambda_sync': Lambda_arr
        }


if __name__ == "__main__":
    print("Testing cosmological synchronization model...")
    
    # Initialize model
    model = FRWSynchronization()
    
    # Evolve from early time
    t_span = np.linspace(1e15, 4.4e17, 1000)  # ~1 Gyr to present
    a_0 = 0.1
    Phi_0 = 0.95
    
    results = model.evolve(a_0, Phi_0, t_span, entropy_model='constant')
    
    print(f"\nCosmological evolution:")
    print(f"  Initial: a = {a_0}, Φ = {Phi_0}")
    print(f"  Final:   a = {results['a'][-1]:.3f}, Φ = {results['Phi'][-1]:.4f}")
    print(f"  Final H: {results['H'][-1]:.2e} s⁻¹")
    print(f"  Final Λ_sync: {results['Lambda_sync'][-1]:.2e} m⁻²")
    
    # Black hole profile
    print("\n\nBlack hole synchronization profile (M = 10 M_☉):")
    bh = BlackHoleSynchronization(M_solar=10.0)
    profile = bh.radial_profile()
    
    print(f"  r = 1.01 r_s: Φ = {profile['Phi'][0]:.4f}")
    print(f"  r = 2 r_s:    Φ = {bh.phi_profile(2*bh.r_s):.4f}")
    print(f"  r = 10 r_s:   Φ = {profile['Phi'][-1]:.4f}")
    
    print(f"\n  Schwarzschild radius: {bh.r_s:.2e} m")
    print(f"  Standard Hawking power: {bh.hawking_power_modified(1.0):.2e} W")
