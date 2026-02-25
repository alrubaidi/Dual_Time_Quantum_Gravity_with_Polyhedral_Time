"""
Unit tests for polyhedral dynamics module.
"""

import pytest
import numpy as np
from src.polyhedral_dynamics import (
    N_DODECA_FACES,
    N_ICOSA_FACES,
    dodecahedron_adjacency,
    icosahedron_adjacency,
    incidence_matrix,
    PolyhedralComplex
)


class TestPolyhedralConstants:
    """Tests for polyhedral structure constants."""
    
    def test_dodecahedron_faces(self):
        assert N_DODECA_FACES == 12
    
    def test_icosahedron_faces(self):
        assert N_ICOSA_FACES == 20
    
    def test_total_modes(self):
        # 32 vertices + 32 faces = 64 modes (E8 correspondence)
        total = (20 + 12) + (12 + 20)  # vertices + faces
        assert total == 64


class TestDodecahedronAdjacency:
    """Tests for dodecahedron adjacency matrix."""
    
    def test_shape(self):
        adj = dodecahedron_adjacency()
        assert adj.shape == (12, 12)
    
    def test_symmetric(self):
        adj = dodecahedron_adjacency()
        assert np.allclose(adj, adj.T)
    
    def test_degree_five(self):
        # Each pentagonal face has 5 neighbors
        adj = dodecahedron_adjacency()
        degrees = adj.sum(axis=1)
        assert np.all(degrees == 5)


class TestIcosahedronAdjacency:
    """Tests for icosahedron adjacency matrix."""
    
    def test_shape(self):
        adj = icosahedron_adjacency()
        assert adj.shape == (20, 20)
    
    def test_symmetric(self):
        adj = icosahedron_adjacency()
        assert np.allclose(adj, adj.T)
    
    def test_degree_three(self):
        # Each triangular face has 3 neighbors
        adj = icosahedron_adjacency()
        degrees = adj.sum(axis=1)
        assert np.all(degrees == 3)


class TestIncidenceMatrix:
    """Tests for dodecahedron-icosahedron incidence."""
    
    def test_shape(self):
        B = incidence_matrix()
        assert B.shape == (20, 12)  # Icosa faces × Dodeca faces
    
    def test_normalized(self):
        B = incidence_matrix()
        row_sums = B.sum(axis=1)
        assert np.allclose(row_sums, 1.0)


class TestPolyhedralComplex:
    """Tests for the full polyhedral complex simulation."""
    
    @pytest.fixture
    def complex(self):
        return PolyhedralComplex(initial_purity=1.0)
    
    def test_initial_purity(self, complex):
        assert np.isclose(complex.purity(), 1.0)
    
    def test_initial_synchronization(self, complex):
        # Pure state should have high Φ
        assert complex.Phi > 0.5
    
    def test_rho_normalized(self, complex):
        # Density matrix should have trace 1
        trace = np.real(np.trace(complex.rho_chronos))
        assert np.isclose(trace, 1.0)
    
    def test_kairos_normalized(self, complex):
        # Kairos amplitudes should be normalized
        norm = np.linalg.norm(complex.g_kairos)
        assert np.isclose(norm, 1.0)
    
    def test_step_preserves_trace(self, complex):
        complex.step(dt=0.1, measure=False)
        trace = np.real(np.trace(complex.rho_chronos))
        assert np.isclose(trace, 1.0, atol=1e-10)
    
    def test_measurement_reduces_purity(self, complex):
        initial_purity = complex.purity()
        complex.step(dt=0.1, measure=True, measure_strength=0.5)
        final_purity = complex.purity()
        assert final_purity < initial_purity
    
    def test_cptp_is_trace_preserving(self, complex):
        complex.cptp_measurement_channel(strength=0.3)
        trace = np.real(np.trace(complex.rho_chronos))
        # CPTP should preserve trace
        assert np.isclose(trace, 1.0, atol=1e-10)


class TestSimulation:
    """Tests for full simulation runs."""
    
    def test_simulation_runs(self):
        complex = PolyhedralComplex()
        purity, Phi = complex.run_simulation(n_steps=100, dt=0.1)
        assert len(purity) == 100
        assert len(Phi) == 100
    
    def test_purity_bounded(self):
        complex = PolyhedralComplex()
        purity, _ = complex.run_simulation(n_steps=100)
        assert np.all(purity >= -1e-10)
        assert np.all(purity <= 1 + 1e-10)  # Allow floating-point tolerance
    
    def test_synchronization_bounded(self):
        complex = PolyhedralComplex()
        _, Phi = complex.run_simulation(n_steps=100)
        assert np.all(Phi >= 0)
        assert np.all(Phi <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
