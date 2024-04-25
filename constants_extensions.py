class SolidConstants(MaterialConstants):
    """Solid material with unit values.

    Each constant (class attribute) typically corresponds to exactly one method which
    scales the value and broadcasts to relevant size, typically number of cells in the
    specified subdomains or interfaces.

    Parameters:
        constants (dict): Dictionary of constants. Only keys corresponding to a constant
            in the class will be used. If not specified, default values are used, mostly
            0 or 1. See the soucre code for permissible keys and default values.
    """

    def __init__(self, constants: Optional[dict] = None):
        default_constants = self.default_constants
        self.verify_constants(constants, default_constants)
        if constants is not None:
            default_constants.update(constants)
        super().__init__(default_constants)

    @property
    def default_constants(self) -> dict[str, number]:
        """Default constants of the material.

        Returns:
            Dictionary of constants.

        """
        # Default values, sorted alphabetically
        # TODO: Numerical method parameters may find a better home soon.
        default_constants = {
            "biot_coefficient": 1,
            "density": 1,
            "dilation_angle": 0,
            "fracture_gap": 0,
            "fracture_normal_stiffness": 1,
            "friction_coefficient": 1,
            "lame_lambda": 1,
            "maximum_fracture_closure": 0,
            "normal_permeability": 1,
            "permeability": 1,
            "porosity": 0.1,
            "residual_aperture": 0.1,
            "shear_modulus": 1,
            "skin_factor": 0,
            "specific_heat_capacity": 1,
            "specific_storage": 1,
            "temperature": 0,
            "thermal_conductivity": 1,
            "thermal_expansion": 0,
            "well_radius": 0.1,
            "displacement_modulus": 1,
            "tangential_characteristic_tol": 1e-5,  # Numerical method parameter
            "contact_mechanics_scaling": 1e-1,  # Numerical method parameter
        }
        return default_constants

    def biot_coefficient(self) -> number:
        """Biot coefficient [-].

        Returns:
            Biot coefficient.

        """
        return self.constants["biot_coefficient"]

    def density(self) -> number:
        """Density [kg/m^3].

        Returns:
            Density in converted mass and length units.

        """
        return self.convert_units(self.constants["density"], "kg * m^-3")

    def thermal_expansion(self) -> number:
        """Thermal expansion coefficient [1/K].

        Returns:
            Thermal expansion coefficient in converted temperature units.

        """
        return self.convert_units(self.constants["thermal_expansion"], "K^-1")

    def specific_heat_capacity(self) -> number:
        """Specific heat [energy / (mass * temperature)].

        Returns:
            Specific heat in converted energy, mass and temperature units.

        """
        return self.convert_units(
            self.constants["specific_heat_capacity"], "J * kg^-1 * K^-1"
        )

    def normal_permeability(self) -> number:
        """Normal permeability [m^2].

        Returns:
            Normal permeability in converted length units.

        """
        return self.convert_units(self.constants["normal_permeability"], "m^2")

    def thermal_conductivity(self) -> number:
        """Thermal conductivity [W/m/K].

        Returns:
            Thermal conductivity in converted energy, length and temperature units.

        """
        return self.convert_units(
            self.constants["thermal_conductivity"], "W * m^-1 * K^-1"
        )

    def porosity(self) -> number:
        """Porosity [-].

        Returns:
            Porosity.

        """
        return self.convert_units(self.constants["porosity"], "-")

    def permeability(self) -> number:
        """Permeability [m^2].

        Returns:
            Permeability in converted length units.

        """
        return self.convert_units(self.constants["permeability"], "m^2")

    def residual_aperture(self) -> number:
        """Residual aperture [m].

        Returns:
            Residual aperture.

        """
        return self.convert_units(self.constants["residual_aperture"], "m")

    def shear_modulus(self) -> number:
        """Shear modulus [Pa].

        Returns:
            Shear modulus in converted pressure units.

        """
        return self.convert_units(self.constants["shear_modulus"], "Pa")

    def specific_storage(self) -> number:
        """Specific storage [1/Pa].

        Returns:
            Specific storage in converted pressure units.

        """
        return self.convert_units(self.constants["specific_storage"], "Pa^-1")

    def lame_lambda(self) -> number:
        """Lame's first parameter [Pa].

        Returns:
            Lame's first parameter in converted pressure units.

        """
        return self.convert_units(self.constants["lame_lambda"], "Pa")

    def fracture_gap(self) -> number:
        """Fracture gap [m].

        Returns:
            Fracture gap in converted length units.

        """
        return self.convert_units(self.constants["fracture_gap"], "m")

    def friction_coefficient(self) -> number:
        """Friction coefficient [-].

        Returns:
            Friction coefficient.

        """
        return self.constants["friction_coefficient"]

    def dilation_angle(self) -> number:
        """Dilation angle.

        Returns:
            Dilation angle in converted angle units.

        """
        return self.convert_units(self.constants["dilation_angle"], "rad")

    def skin_factor(self) -> number:
        """Skin factor [-].

        Returns:
            Skin factor.

        """
        return self.constants["skin_factor"]

    def temperature(self) -> number:
        """Temperature [K].

        Returns:
            Temperature in converted temperature units.

        """
        return self.convert_units(self.constants["temperature"], "K")

    def well_radius(self) -> number:
        """Well radius [m].

        Returns:
            Well radius in converted length units.

        """
        return self.convert_units(self.constants["well_radius"], "m")

    def fracture_normal_stiffness(self) -> number:
        """The normal stiffness of a fracture [Pa * m^-1].

        Intended use is in Barton-Bandis-type models for elastic fracture deformation.

        Returns:
            The fracture normal stiffness in converted units.

        """
        return self.convert_units(
            self.constants["fracture_normal_stiffness"], "Pa*m^-1"
        )

    def maximum_fracture_closure(self) -> number:
        """The maximum closure of a fracture [m].

        Intended use is in Barton-Bandis-type models for elastic fracture deformation.

        Returns:
            The maximal closure of a fracture.

        """
        return self.convert_units(self.constants["maximum_fracture_closure"], "m")

    def tangential_characteristic_tol(self) -> number:
        """Tolerance parameter for the tangential characteristic contact mechanics [-].

        FIXME:
            Revisit the tolerance.

        Returns:
            The tolerance parameter.

        """
        return self.constants["tangential_characteristic_tol"]

    def contact_mechanics_scaling(self) -> number:
        """Safety scaling factor, making fractures softer than the matrix [-].

        Returns:
            The softening factor.

        """
        return self.constants["contact_mechanics_scaling"]
    
    def displacement_modulus(self) -> number:
        """Modulus for the elastic tangential fracture displacement.

        Returns:
            The elastic displacement modulus.
        """
        return self.constants["displacement_modulus"]
