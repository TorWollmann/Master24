from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np

import porepy as pp

number = pp.number
class SolidConstantsWithTangentialStiffness(pp.SolidConstants):
    """Solid material with unit values.

    Each constant (class attribute) typically corresponds to exactly one method which
    scales the value and broadcasts to relevant size, typically number of cells in the
    specified subdomains or interfaces.

    Parameters:
        constants (dict): Dictionary of constants. Only keys corresponding to a constant
            in the class will be used. If not specified, default values are used, mostly
            0 or 1. See the soucre code for permissible keys and default values.
    """



    @property
    def default_constants(self) -> dict[str, number]:
        """Default constants of the material.

        Returns:
            Dictionary of constants.

        """
        # Default values, sorted alphabetically

        
        constants = super().default_constants
        constants.update({"tangential_fracture_stiffness": 1.0})
        return constants

    
    def tangential_fracture_stiffness(self) -> number:       #TorVariable
        """Modulus for the elastic tangential fracture displacement.

        Returns:
            The elastic displacement modulus.
        """
        return self.constants["tangential_fracture_stiffness"]
