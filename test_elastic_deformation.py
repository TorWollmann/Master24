

# %%

from __future__ import annotations
import numpy as np
import pytest
import porepy as pp
from porepy.models.momentum_balance import MomentumBalance
from model_extensions import ElastoPlasticFractureGap
from constants_extensions import SolidConstantsWithTangentialStiffness
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)

class MyMomentumBalance(
    ElastoPlasticFractureGap,
    MomentumBalance,
):
    ...

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
    sd0 = setup.mdg.subdomains(dim=setup.nd)[0]
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)



    #The values extracted for displacement and displacement_jump below are stored in this
    #particular configuration :[u_x0,u_y0,u_x1,u_y1, ...] so we slice them to extract
    #individual x and y values.
    u_p=setup.plastic_displacement_jump(sd_frac).value(setup.equation_system)

    u_e=setup.elastic_displacement_jump(sd_frac).value(setup.equation_system)

    u_domain=setup.displacement(sd).value(setup.equation_system)



    #Extracting values for the cells above the fracture
    #For the parameters of this test (stiff domain, elastic transverse fracture) these should
    #match the displacement on the top boundary.
    u_x=u_domain[::2][sd0.cell_centers[1,:]>0.5]
    u_y=u_domain[1::2][sd0.cell_centers[1,:]>0.5]

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
