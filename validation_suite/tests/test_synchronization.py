"""
Unit tests for synchronization parameter module.
"""

import pytest
import numpy as np
from src.synchronization import (
    SynchronizationParameter,
    phase_diagram
)


class TestSynchronizationParameter:
    """Tests for the SynchronizationParameter class."""
    
    @pytest.fixture
    def sync(self):
        return SynchronizationParameter(
            mu_squared=1.0,
            lambda_quartic=1.0
        )
    
    def test_equilibrium_positive_mu(self, sync):
        # For μ² > 0, Φ_* = sqrt(μ²/λ)
        assert np.isclose(sync.Phi_star, 1.0)
    
    def test_equilibrium_negative_mu(self):
        sync = SynchronizationParameter(mu_squared=-1.0, lambda_quartic=1.0)
        assert sync.Phi_star == 0.0
    
    def test_potential_minimum(self, sync):
        # Derivative at equilibrium should be zero
        dV = sync.potential_derivative(sync.Phi_star)
        assert np.isclose(dV, 0.0, atol=1e-10)
    
    def test_potential_convex_at_minimum(self, sync):
        # Second derivative should be positive at minimum
        d2V = sync.second_derivative(sync.Phi_star)
        assert d2V > 0
    
    def test_lambda_sync_at_phi_1(self, sync):
        Lambda = sync.lambda_sync(1.0)
        # At Φ=1, Λ_sync = Λ_0
        assert np.isclose(Lambda, 1e-52)
    
    def test_lambda_sync_at_phi_0(self, sync):
        Lambda_0 = sync.lambda_sync(0.0)
        Lambda_1 = sync.lambda_sync(1.0)
        # At Φ=0, should be larger
        assert Lambda_0 > Lambda_1
    
    def test_from_entropy_curvature(self, sync):
        # Zero entropy, zero variance → Φ = 1
        Phi = sync.from_entropy_curvature(entropy=0.0, curvature_variance=0.0)
        assert np.isclose(Phi, 1.0)
    
    def test_high_entropy_reduces_phi(self, sync):
        Phi_low = sync.from_entropy_curvature(entropy=0.1, curvature_variance=0.0)
        Phi_high = sync.from_entropy_curvature(entropy=1.0, curvature_variance=0.0)
        assert Phi_high < Phi_low


class TestEvolution:
    """Tests for synchronization evolution."""
    
    def test_evolution_converges(self):
        sync = SynchronizationParameter(mu_squared=1.0, lambda_quartic=1.0,
                                        tau_Phi=1.0)
        t = np.linspace(0, 10, 100)
        Phi_t = sync.evolve(Phi_0=0.1, t_span=t)
        
        # Should converge to equilibrium
        assert np.isclose(Phi_t[-1], sync.Phi_star, atol=0.1)
    
    def test_evolution_monotonic(self):
        sync = SynchronizationParameter(tau_Phi=1.0)
        t = np.linspace(0, 10, 100)
        Phi_t = sync.evolve(Phi_0=0.1, t_span=t)
        
        # Should be monotonically increasing toward Φ_*
        differences = np.diff(Phi_t)
        assert np.all(differences >= -1e-10)


class TestCriticalExponents:
    """Tests for mean-field critical exponents."""
    
    def test_exponent_values(self):
        sync = SynchronizationParameter()
        exponents = sync.critical_exponents()
        
        # Mean-field values
        assert exponents['beta'] == 0.5
        assert exponents['gamma'] == 1.0
        assert exponents['nu'] == 0.5
        assert exponents['delta'] == 3.0


class TestPhaseDiagram:
    """Tests for phase diagram computation."""
    
    def test_phase_transition(self):
        mu_range = np.linspace(-1, 1, 100)
        mu_vals, Phi_vals = phase_diagram(mu_range)
        
        # Φ should be 0 for μ² < 0
        assert np.all(Phi_vals[mu_range < 0] == 0)
        
        # Φ should be positive for μ² > 0
        assert np.all(Phi_vals[mu_range > 0] > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
