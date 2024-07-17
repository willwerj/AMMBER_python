"""
Tests functions
"""

import ammber.BinarySystems as BS


def test_01():
    phase1 = BS.BinaryIsothermal2ndOrderPhase(fmin = 0, kwell= 1, cmin= 0.5)
    assert (phase1.free_energy(0.5) == 0)