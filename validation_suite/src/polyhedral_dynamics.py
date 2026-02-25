"""
Polyhedral Dynamics: Dodecahedron-Icosahedron Dual Complex

This module simulates the coupled Kairos-Chronos dynamics on the
dual polyhedral time complex.

- Dodecahedron (12 faces): Chronos sector, discrete quantum updates
- Icosahedron (20 faces): Kairos sector, continuous geometric evolution

References:
- Section 4.3 (Dual polyhedral time complex)
- Section 6 (Fourfold complementarity and polyhedral bookkeeping)
"""

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
from typing import Tuple, List, Optional


# Polyhedral constants
N_DODECA_FACES = 12
N_DODECA_VERTICES = 20
N_ICOSA_FACES = 20
N_ICOSA_VERTICES = 12
N_EDGES = 30  # Both have 30 edges

# Total modes: 32 vertices + 32 faces = 64
N_TOTAL_MODES = 64


def dodecahedron_adjacency() -> np.ndarray:
    """
    Returns the 12x12 face-adjacency matrix of the dodecahedron.
    
    Each pentagonal face shares an edge with 5 neighbors.
    """
    # Standard face adjacency for regular dodecahedron
    # Using icosahedral symmetry labeling
    adj = np.zeros((12, 12), dtype=int)
    
    # Each face has 5 neighbors
    neighbors = [
        [1, 2, 3, 4, 5],      # Face 0 (top)
        [0, 2, 5, 6, 7],      # Face 1
        [0, 1, 3, 7, 8],      # Face 2
        [0, 2, 4, 8, 9],      # Face 3
        [0, 3, 5, 9, 10],     # Face 4
        [0, 1, 4, 6, 10],     # Face 5
        [1, 5, 7, 10, 11],    # Face 6
        [1, 2, 6, 8, 11],     # Face 7
        [2, 3, 7, 9, 11],     # Face 8
        [3, 4, 8, 10, 11],    # Face 9
        [4, 5, 6, 9, 11],     # Face 10
        [6, 7, 8, 9, 10],     # Face 11 (bottom)
    ]
    
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            adj[i, j] = 1
    
    return adj


def icosahedron_adjacency() -> np.ndarray:
    """
    Returns the 20x20 face-adjacency matrix of the icosahedron.
    
    Each triangular face shares an edge with 3 neighbors.
    """
    adj = np.zeros((20, 20), dtype=int)
    
    # Triangular face adjacency pattern
    neighbors = [
        [1, 2, 5],    # 0
        [0, 2, 3],    # 1
        [0, 1, 4],    # 2
        [1, 4, 6],    # 3
        [2, 3, 5],    # 4
        [0, 4, 7],    # 5
        [3, 8, 9],    # 6
        [5, 10, 11],  # 7
        [6, 9, 12],   # 8
        [6, 8, 13],   # 9
        [7, 11, 14],  # 10
        [7, 10, 15],  # 11
        [8, 13, 16],  # 12
        [9, 12, 17],  # 13
        [10, 15, 18], # 14
        [11, 14, 19], # 15
        [12, 17, 18], # 16
        [13, 16, 19], # 17
        [14, 16, 19], # 18
        [15, 17, 18], # 19
    ]
    
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            adj[i, j] = 1
    
    return adj


def incidence_matrix() -> np.ndarray:
    """
    Returns the 20x12 incidence matrix B_ai linking icosahedral faces
    to dodecahedral faces through the duality.
    
    B_ai = 1 if icosahedral face a is dual to dodecahedral face i.
    """
    # In the dual relationship:
    # - 12 dodeca faces ↔ 12 icosa vertices
    # - 20 dodeca vertices ↔ 20 icosa faces
    
    # Simplified coupling matrix based on dual vertex-face correspondence
    B = np.zeros((N_ICOSA_FACES, N_DODECA_FACES))
    
    # Each icosahedral face is coupled to neighboring dodecahedral faces
    # through the dual vertex structure
    for a in range(N_ICOSA_FACES):
        # Connect to 3 dodecahedral faces (triangular neighborhood)
        coupled_faces = [a % 12, (a + 1) % 12, (a + 2) % 12]
        for i in coupled_faces:
            B[a, i] = 1.0 / 3.0  # Normalized coupling
    
    return B


class PolyhedralComplex:
    """
    Simulates the dual dodecahedron-icosahedron time complex.
    
    Attributes:
        rho_chronos: 12x12 density matrix for Chronos sector
        g_kairos: 20-component amplitude vector for Kairos sector
        Phi: Synchronization parameter
    """
    
    def __init__(self, initial_purity: float = 1.0):
        """
        Initialize the polyhedral complex.
        
        Args:
            initial_purity: Initial purity of Chronos state (default: pure state)
        """
        # Chronos: Initialize as |0⟩⟨0| (pure state on first face)
        self.rho_chronos = np.zeros((N_DODECA_FACES, N_DODECA_FACES), dtype=complex)
        self.rho_chronos[0, 0] = 1.0
        
        # If not pure, mix with maximally mixed state
        if initial_purity < 1.0:
            mixed = np.eye(N_DODECA_FACES) / N_DODECA_FACES
            self.rho_chronos = initial_purity * self.rho_chronos + (1 - initial_purity) * mixed
        
        # Kairos: Initialize with uniform amplitudes on icosahedral faces
        self.g_kairos = np.ones(N_ICOSA_FACES) / np.sqrt(N_ICOSA_FACES)
        
        # Adjacency matrices
        self.A_dodeca = dodecahedron_adjacency()
        self.A_icosa = icosahedron_adjacency()
        self.B = incidence_matrix()
        
        # Kairos generator (graph Laplacian)
        self.Omega = self._kairos_generator()
        
        # Synchronization parameter
        self.Phi = self.compute_synchronization()
    
    def _kairos_generator(self) -> np.ndarray:
        """
        Construct the Kairos evolution generator from icosahedral graph.
        
        Returns graph Laplacian: Ω = D - A where D is degree matrix.
        """
        degree = np.diag(self.A_icosa.sum(axis=1))
        return degree - self.A_icosa
    
    def chronos_unitary(self, dt: float = 0.1) -> np.ndarray:
        """
        Generate a unitary step for Chronos evolution.
        
        Uses the dodecahedral adjacency as a Hamiltonian.
        """
        H_chronos = self.A_dodeca.astype(float)
        return expm(-1j * H_chronos * dt)
    
    def cptp_measurement_channel(self, strength: float = 0.1) -> np.ndarray:
        """
        Apply a CPTP measurement channel (partial decoherence).
        
        This implements the non-invertible semigroup dynamics:
        record formation introduces an information barrier.
        
        Args:
            strength: Decoherence strength in [0, 1]
        
        Returns:
            Updated density matrix
        """
        # Kraus operators for dephasing channel
        # K_0 = sqrt(1-p) I, K_i = sqrt(p/12) |i⟩⟨i|
        
        p = strength
        rho_new = (1 - p) * self.rho_chronos
        
        # Add diagonal projector terms
        for i in range(N_DODECA_FACES):
            proj = np.zeros((N_DODECA_FACES, N_DODECA_FACES))
            proj[i, i] = 1.0
            rho_new += (p / N_DODECA_FACES) * proj @ self.rho_chronos @ proj
        
        return rho_new
    
    def kairos_step(self, dt: float = 0.1, coupling: float = 0.1) -> np.ndarray:
        """
        Evolve Kairos sector for time step dt.
        
        dg_a/dt_r = -Ω_ab g_b + λ Tr(ρ O_i) B_ai
        """
        # Compute coupling from Chronos observables
        O_expectation = np.diag(self.rho_chronos.real)
        coupling_term = coupling * self.B @ O_expectation
        
        # Linear evolution
        g_new = self.g_kairos - dt * (self.Omega @ self.g_kairos) + dt * coupling_term
        
        # Normalize
        return g_new / np.linalg.norm(g_new)
    
    def compute_synchronization(self) -> float:
        """
        Compute the synchronization parameter Φ.
        
        Φ = exp(-α S(ρ) - β Var(g))
        
        where S(ρ) is von Neumann entropy and Var(g) is amplitude variance.
        """
        alpha = 0.5
        beta = 0.5
        
        # Von Neumann entropy of Chronos state
        eigenvalues = np.linalg.eigvalsh(self.rho_chronos)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Avoid log(0)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        
        # Variance of Kairos amplitudes
        variance = np.var(np.abs(self.g_kairos)**2)
        
        # Synchronization parameter
        Phi = np.exp(-alpha * entropy - beta * variance)
        
        return float(np.clip(Phi, 0.0, 1.0))
    
    def purity(self) -> float:
        """Compute purity Tr(ρ²) of Chronos state."""
        return float(np.real(np.trace(self.rho_chronos @ self.rho_chronos)))
    
    def step(self, dt: float = 0.1, measure: bool = False,
             measure_strength: float = 0.1) -> Tuple[float, float]:
        """
        Perform one combined Chronos-Kairos evolution step.
        
        Args:
            dt: Time step
            measure: Whether to apply CPTP measurement channel
            measure_strength: Strength of measurement decoherence
        
        Returns:
            Tuple of (purity, synchronization)
        """
        # Chronos: Unitary evolution
        U = self.chronos_unitary(dt)
        self.rho_chronos = U @ self.rho_chronos @ U.conj().T
        
        # Chronos: Optional measurement (CPTP channel)
        if measure:
            self.rho_chronos = self.cptp_measurement_channel(measure_strength)
        
        # Kairos: Continuous evolution with coupling
        self.g_kairos = self.kairos_step(dt)
        
        # Update synchronization
        self.Phi = self.compute_synchronization()
        
        return self.purity(), self.Phi
    
    def run_simulation(self, n_steps: int = 500, dt: float = 0.1,
                       measure_step: Optional[int] = None,
                       measure_strength: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a full simulation of the polyhedral complex.
        
        Args:
            n_steps: Number of time steps
            dt: Time step size
            measure_step: Step at which to apply measurement (None = no measurement)
            measure_strength: Strength of measurement decoherence
        
        Returns:
            Arrays of (purity_history, Phi_history)
        """
        purity_history = []
        Phi_history = []
        
        for n in range(n_steps):
            measure = (measure_step is not None and n == measure_step)
            purity, Phi = self.step(dt, measure=measure, 
                                    measure_strength=measure_strength)
            purity_history.append(purity)
            Phi_history.append(Phi)
        
        return np.array(purity_history), np.array(Phi_history)


if __name__ == "__main__":
    print("Running polyhedral dynamics simulation...")
    
    # Create complex
    complex = PolyhedralComplex(initial_purity=1.0)
    
    print(f"Initial purity: {complex.purity():.4f}")
    print(f"Initial Φ: {complex.Phi:.4f}")
    
    # Run simulation with measurement at step 250
    purity_hist, Phi_hist = complex.run_simulation(
        n_steps=500, dt=0.1, 
        measure_step=250, measure_strength=0.3
    )
    
    print(f"\nFinal purity: {purity_hist[-1]:.4f}")
    print(f"Final Φ: {Phi_hist[-1]:.4f}")
    print(f"Purity drop at measurement: {purity_hist[249]:.4f} → {purity_hist[251]:.4f}")
