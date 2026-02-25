"""
Unit tests for constraint algebra verification.
"""

import pytest
import numpy as np
from src.constraint_algebra import (
    epsilon_tensor,
    poisson_bracket_Qab,
    verify_sp2r_closure,
    verify_so22_closure,
    symbolic_constraint_verification
)


class TestEpsilonTensor:
    """Tests for the antisymmetric tensor."""
    
    def test_antisymmetric(self):
        eps = epsilon_tensor()
        assert eps[0, 1] == -eps[1, 0]
    
    def test_diagonal_zero(self):
        eps = epsilon_tensor()
        assert eps[0, 0] == 0
        assert eps[1, 1] == 0
    
    def test_values(self):
        eps = epsilon_tensor()
        assert eps[0, 1] == 1
        assert eps[1, 0] == -1


class TestPoissonBrackets:
    """Tests for constraint Poisson brackets."""
    
    def test_bracket_returns_dict(self):
        result = poisson_bracket_Qab(0, 0, 1, 1)
        assert isinstance(result, dict)
    
    def test_bracket_keys_are_tuples(self):
        result = poisson_bracket_Qab(0, 0, 0, 1)
        for key in result.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2


class TestAlgebraClosure:
    """Tests for algebra closure verification."""
    
    def test_sp2r_closes(self):
        assert verify_sp2r_closure() is True
    
    def test_so22_closure(self):
        success, results = verify_so22_closure()
        assert success is True
        assert results['sp2r_closure'] is True
    
    def test_physical_dof_count(self):
        _, results = verify_so22_closure()
        # Should have 6 physical DOF for 4D physics
        assert results['physical_dof'] == 6


class TestSymbolicVerification:
    """Tests for symbolic algebra."""
    
    def test_symbolic_constraints_defined(self):
        constraints = symbolic_constraint_verification()
        assert 'X^2' in constraints
        assert 'P^2' in constraints
        assert 'X·P' in constraints
    
    def test_metric_signature(self):
        constraints = symbolic_constraint_verification()
        eta = constraints['metric']
        # Check (4,2) signature: (-,+,+,+,+,-)
        diag = [eta[i, i] for i in range(6)]
        assert diag[0] == -1  # Timelike
        assert diag[5] == -1  # Second timelike


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
