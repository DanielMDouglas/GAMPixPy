# TODO: use a more sophisticated system for this.
# When do quntities acquere units?
# - in config file?
# - when config is loaded?
# - when additional parameters are computed?

"""

The basic units are :
    kilogram                (kg)
    centimeter              (cm)
    microsecond             (us)
    positron charge         (e)
    Kelvin                  (kelvin)
    Mega electron-volt      (MeV)
    the amount of substance (mole)
    luminous intensity      (candela)
    radian                  (radian)
    steradian               (steradian)
"""

# mass
me = 9.109e-31
kg = 1.
g = 1.e-3*kg

# temperature
K = 1

# distance
km = 1.e5
m = 1.e2
cm = 1.
mm = 1.e-1
um = 1.e-4
nm = 1.e-7

# volume
mL = cm*cm*cm
L = 1.e3*mL

# time
s = 1.e6
ms = 1.e3
us = 1.
ns = 1.e-3

minute = 60*s
hour = 60*m

# charge
e = 1.
ke = 1.e3

C = 6.242e18*e

# potential
mV = 1.e-9
V = 1.e-6
kV = 1.e-3
MV = 1.

# energy
meV = 1.e-9
eV = 1.e-6
keV = 1.e-3
MeV = 1.
GeV = 1.e3
TeV = 1.e6

J = eV/e*C

# power
W = J/s

# angle
from math import pi
radians = 1.
degrees = 360./(2*pi)

def unit_parser(unit_string_expression):
    """
    unit_parser(unit_string_expression)

    Parser function for converting unit string expressions into a
    numerical representation based on the unit scheme defined in units.py
    
    Parameters
    ----------
    unit_string_expression : string
        String expression of units, in the form of "cm*cm/us/MeV".

    Returns
    -------
    unit_product : float
        Numerical representation of the unit string provided.
    
    """
    # units need to be in formats like:
    # cm*cm/s
    # no carrot notation! (right now)
    unit_words = []
    unit_power = [1]
    this_word = ''
    for char in unit_string_expression:
        if char == '*':
            unit_power.append(1)
            unit_words.append(this_word)
            this_word = ''
        elif char == '/':
            unit_power.append(-1)
            unit_words.append(this_word)
            this_word = ''
        elif char == ' ':
            continue
        else:
            this_word += char

    unit_words.append(this_word)

    unit_factor = []
    for this_word in unit_words:
        assert (this_word in globals()), this_word + " not found!"
        unit_factor.append(globals()[this_word])
        
    unit_product = 1
    for unit, power in zip(unit_factor, unit_power):
        unit_product *= pow(unit, power)

    return unit_product
