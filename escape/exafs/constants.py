"""
Physical constants used in EXAFS data reduction.

The only constant that really matters for day-to-day EXAFS work is
ETOK: it converts an energy above the absorption edge (in eV) into
the photoelectron wavenumber k (in inverse Angstroms).

    k [Ang^-1] = sqrt( ETOK * (E - E0) [eV] )

This comes from the free-electron (plane wave) approximation for the
ejected photoelectron:

    E - E0 = hbar^2 k^2 / (2 m_e)
    =>  k   = sqrt( 2 m_e (E - E0) / hbar^2 )

ETOK = 2 m_e / hbar^2, expressed in eV^-1 Ang^-2. Its numerical value
(0.2624682917 ...) is the same constant used by IFEFFIT / Larch /
Athena, so results from this package are directly comparable to
those tools.
"""

# 2*m_e/hbar^2 in eV^-1 * Angstrom^-2
ETOK = 0.2624682917

# Angstrom <-> inverse-Angstrom relationship is just 2*pi, kept here
# for readability in the Fourier-transform code.
import math
TWO_PI = 2.0 * math.pi
