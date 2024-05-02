#TestCase¨
import numpy as np
import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.models.momentum_balance import MomentumBalance
from Master24.model_extensions import ElastoPlasticFractureGap
from Master24.constants_extensions import SolidConstantsWithTangentialStiffness




class TestCaseBC2D():
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type for the problem."""
        bounds = self.domain_boundary_sides(sd)

        # Set the type of west and east boundaries to Dirichlet. North and south are
        # Neumann by default.
        bc = pp.BoundaryConditionVectorial(sd, bounds.north, "dir")
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting stress boundary condition values at north and south boundaries.

        Specifically, we assign different values for the x- and y-component of the
        boundary value vector.

        """
        values = np.ones((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        # Assigning x-component values
        values[0][bounds.east + bounds.west + bounds.south] *= self.solid.convert_units(1, "Pa")

        # Assigning y-component values
        values[1][bounds.west + bounds.east + bounds.south] *= self.solid.convert_units(1, "Pa")

        return values.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting displacement boundary condition values.

        This method returns an array of boundary condition values with the value 5t for
        western boundaries and ones for the eastern boundary.

        """
        # Fetch the time of the current time-step
        t = self.time_manager.time

        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        # Assign a time dependent value to the x-component of the western boundary
        values[0][bounds.north] += self.solid.convert_units(1.0 * t, "m")

        # The convention for flattening nd-arrays of vector values in PorePy is by using
        # the Fortran-style ordering (chosen by string "F" when giving a call to ravel).
        # That is, the first index changes the fastest and the last index changes
        # slowest.
        return values.ravel("F")



class TestCaseBC3D():
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type for the problem."""
        bounds = self.domain_boundary_sides(sd)

        # Set the type of west and east boundaries to Dirichlet. North and south are
        # Neumann by default.
        bc = pp.BoundaryConditionVectorial(sd, bounds.north, "dir")
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting stress boundary condition values at north and south boundaries.

        Specifically, we assign different values for the x- and y-component of the
        boundary value vector.

        """
        values = np.ones((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        # Assigning x-component values
        values[0][bounds.east + bounds.west + bounds.south + bounds.top + bounds.bottom] *= self.solid.convert_units(1, "Pa")

        # Assigning y-component values
        values[1][bounds.west + bounds.east + bounds.south + bounds.top + bounds.bottom] *= self.solid.convert_units(1, "Pa")

        return values.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting displacement boundary condition values.

        This method returns an array of boundary condition values with the value 5t for
        western boundaries and ones for the eastern boundary.

        """
        # Fetch the time of the current time-step
        t = self.time_manager.time

        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        # Assign a time dependent value to the x-component of the western boundary
        values[0][bounds.north] += self.solid.convert_units(1.0 * t, "m")

        # The convention for flattening nd-arrays of vector values in PorePy is by using
        # the Fortran-style ordering (chosen by string "F" when giving a call to ravel).
        # That is, the first index changes the fastest and the last index changes
        # slowest.
        return values.ravel("F")



class MyMomentumBalance(ElastoPlasticFractureGap,MomentumBalance):
    ...




class TestCase2D(
TestCaseBC2D,
SquareDomainOrthogonalFractures,
MyMomentumBalance,
):
    ...

class TestCase3D(
TestCaseBC3D,
CubeDomainOrthogonalFractures,
MyMomentumBalance,
):
    ...

solid_constants = SolidConstantsWithTangentialStiffness
time_manager = pp.TimeManager(
    schedule=[0, 5],
    dt_init=1,
    constant_dt=True,
)

params = {
    "material_constants": {"solid": solid_constants},
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 0.25},
    "time_manager": time_manager
}




model = [TestCase2D(params),TestCase3D(params)]


pp.run_time_dependent_model(model[0], params)

print("complete")
