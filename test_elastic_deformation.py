

# %%

from __future__ import annotations

import copy

import numpy as np
import pytest

import porepy as pp
"""from porepy.applications.test_utils.models import (
    MomentumBalance,
)"""

from porepy.models.momentum_balance import MomentumBalance
from model_extensions import ElastoPlasticFractureGap
from constants_extensions import SolidConstantsWithTangentialStiffness
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)

class MyMomentumBalance(
    ElastoPlasticFractureGap,
    MomentumBalance,
):
    ...

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


class LinearModel(
    SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    MyMomentumBalance,
):

    pass




@pytest.mark.parametrize(
    "north_displacement,u_e_expected,u_p_expected,u_x_expected",
    [
        ([1.0,-1.0],[-1,0], [0,0], [1]),
        ([1.0,1.0],[0,0],  [-1,1], [1]),
    ],
)
def test_2d_single_fracture(north_displacement,u_e_expected,u_p_expected,u_x_expected):
    """Test that the solution is qualitatively sound.

    Parameters:
        north_displacement (list): Value of displacement on the north boundary.
        u_e_expected (list): Expected values of the elastic displacement jump in the x and y.
            directions.
        u_p_expected (list): Expected values of the plastic displacement jump in the x and y.
            directions.
        u_x_expected (list): Expected value of displacement in the x direction of cells above the fracture.

    """
    # Instantiate constants and store in params.
    solid_vals = {  "tangential_fracture_stiffness": 1e-5,
                    "shear_modulus": 1e6,
                    "lame_lambda": 1e3,
                    }
    solid = SolidConstantsWithTangentialStiffness(solid_vals)
    params = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid},
        "ux_north": north_displacement[0],
        "uy_north": north_displacement[1],
        "fracture_indices":[1],
    }

    # Create model and run simulation
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup, params)


    sd = setup.mdg.subdomains(dim=setup.nd)
    tempsd = setup.mdg.subdomains(dim=setup.nd)[0]
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)



    #rot = setup.local_coordinates(sd_frac).transpose()
    #Rotasjonen virker ikke
    #lagt på hyllen intil videre

    u_p=setup.plastic_displacement_jump(sd_frac).value(setup.equation_system)

    u_e=setup.elastic_displacement_jump(sd_frac).value(setup.equation_system)

    u_domain=setup.displacement(sd).value(setup.equation_system)





    print(f"u_e={u_e}")
    print(f"u_e_x={u_e[::2]}")
    print(f"u_e_y={u_e[1::2]}")


    print(f"u_p={u_p}")
    print(f"u_p_x={u_p[::2]}")
    print(f"u_p_x={u_p[1::2]}")

    tol=1e-10

    #Extracting values for the cells above the fracture
    #For the parameters of this test (stiff domain, elastic transverse fracture) these should
    #match the displacement on the top boundary.
    u_x=u_domain[::2][tempsd.cell_centers[1,:]>0.5]
    u_y=u_domain[1::2][tempsd.cell_centers[1,:]>0.5]

    print(f"u_x{u_x}")
    print(f"u_y{u_y}")

    #When looking at the elastic displacement it is important to note that the sign of the
    #value is dependent on local coordinates, that are set during grid construction.
    #For this specific grid we except a value of -1,even if physically it should be 1.
    assert np.allclose(u_e[::2],u_e_expected[0])
    assert np.allclose(u_e[1::2],u_e_expected[1])


    #Again, because of the local coordinates (that depend on the grid construction)
    #here we will expect a -1, instead of 1.
    assert np.allclose(u_p[::2],u_p_expected[0])
    assert np.allclose(u_p[1::2],u_p_expected[1])


    assert np.allclose(u_x,u_x_expected[0], atol=1e-2)



#test_2d_single_fracture([1.0,-1.0],[-1,0], [0,0], [1])



#test_2d_single_fracture([1.0,1.0],[0,0],  [-1,1],[1])