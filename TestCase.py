#TestCase¨
import numpy as np
import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.models.momentum_balance import MomentumBalance
from model_extensions import ElastoPlasticFractureGap
from constants_extensions import SolidConstantsWithTangentialStiffness


import time

timed0 = time.time()

class DisplacementJumpExporting:
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            vals = self._evaluate_and_scale(sd, "displacement_jump", "m")
            vals2 = self._evaluate_and_scale(sd, "plastic_displacement_jump", "m")
            vals3 = self._evaluate_and_scale(sd, "elastic_displacement_jump", "m")
            data.append((sd, "displacement_jump", vals))
            data.append((sd, "plastic_displacement_jump", vals2))            
            data.append((sd, "elastic_displacement_jump", vals3))           
        return data






class Tor3DGeometry:
    def set_fractures(self) -> None:
        f_1 = pp.PlaneFracture(np.array([[1, 1, 2, 2], [1, 2, 1, 2], [1, 1, 1, 1]]))
        self._fractures = [f_1]

    def set_domain(self) -> None:
        bounding_box = {
            "xmin": 0,
            "xmax": 3,
            "ymin": 0,
            "ymax": 3,
            "zmin": 0,
            "zmax": 3,
        }
        self._domain = pp.Domain(bounding_box=bounding_box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")


class Tor2DGeometry:
    def set_fractures(self) -> None:
        f_1 = pp.LineFracture(np.array([[1, 2], [1, 1]]))
        self._fractures = [f_1]

    def set_domain(self) -> None:
        bounding_box = {
            "xmin": 0,
            "xmax": 3,
            "ymin": 0,
            "ymax": 3,
        }
        self._domain = pp.Domain(bounding_box=bounding_box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")
    


class Test2DGeometry:
    def set_fractures(self) -> None:
        f_1 = pp.LineFracture(np.array([[0, 2], [1, 1]]))
        self._fractures = [f_1]

    def set_domain(self) -> None:
        bounding_box = {
            "xmin": 0,
            "xmax": 2,
            "ymin": 0,
            "ymax": 2,
        }
        self._domain = pp.Domain(bounding_box=bounding_box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")




class TestCaseBC2D():
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type for the problem."""
        bounds = self.domain_boundary_sides(sd)

        # Set the type of west and east boundaries to Dirichlet. North and south are
        # Neumann by default.
        bc = pp.BoundaryConditionVectorial(sd, bounds.south+bounds.north, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting stress boundary condition values at north and south boundaries.

        Specifically, we assign different values for the x- and y-component of the
        boundary value vector.

        """
        values = np.ones((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        # Assigning x-component values
        values[0][bounds.west + bounds.east] *= self.solid.convert_units(0, "Pa")

        # Assigning y-component values
        values[1][bounds.west + bounds.east] *= self.solid.convert_units(0, "Pa")

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
        values[0][bounds.north] += self.solid.convert_units(1.0e-1 * t, "m")
        values[1][bounds.north] += self.solid.convert_units(-1.0e-1 * t, "m")



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
        bc = pp.BoundaryConditionVectorial(sd, bounds.top + bounds.bottom, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting stress boundary condition values at north and south boundaries.

        Specifically, we assign different values for the x- and y-component of the
        boundary value vector.

        """
        values = np.ones((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        # Assigning x-component values
        values[0][bounds.west + bounds.east + bounds.north + bounds.south] *= self.solid.convert_units(0, "Pa")

        # Assigning y-component values
        values[1][bounds.west + bounds.east + bounds.north + bounds.south] *= self.solid.convert_units(0, "Pa")

        values[2][bounds.west + bounds.east + bounds.north + bounds.south] *= self.solid.convert_units(0, "Pa")

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
        values[2][bounds.top] += self.solid.convert_units(-1.0 * t, "m")

        # The convention for flattening nd-arrays of vector values in PorePy is by using
        # the Fortran-style ordering (chosen by string "F" when giving a call to ravel).
        # That is, the first index changes the fastest and the last index changes
        # slowest.
        return values.ravel("F")



class MyMomentumBalance(
        DisplacementJumpExporting,
        ElastoPlasticFractureGap,
        MomentumBalance):
    ...




class TestCase2D(
        TestCaseBC2D,
        Tor2DGeometry,
        MyMomentumBalance,
):
    ...

class TestCase3D(
        TestCaseBC3D,
        Tor3DGeometry,
        MyMomentumBalance,
):
    ...




temp=SolidConstantsWithTangentialStiffness(
    {
        "tangential_fracture_stiffness" : 0.01,
        "friction_coefficient": 1.0e5,
        "shear_modulus": 1e6,
        "lame_lambda": 1e6,
     }
    )
solid_constants = temp
time_manager = pp.TimeManager(
    schedule=[0, 10],
    dt_init=1,
    constant_dt=True,
)

params = {
    "material_constants": {"solid": solid_constants},
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 0.25},
    "time_manager": time_manager,
}




model = [TestCase2D(params),TestCase3D(params)]


pp.run_time_dependent_model(model[0], params)


"""Pytest setup for paraview"""

params1 = {
    "material_constants": {"solid": solid_constants},
    "fracture_indices":[1],
    "time_manager": time_manager
}

class LinearModel(
    TestCaseBC2D,
    SquareDomainOrthogonalFractures,
    MyMomentumBalance,
):

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.25}
        return mesh_args 
 
    pass


setup = LinearModel(params1)
#pp.run_time_dependent_model(setup, params1)








timed1 = time.time()
totaltime = timed1 - timed0
print(totaltime)
print("complete")



