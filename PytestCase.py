

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
    Test2DGeometry,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    MyMomentumBalance,
):
    pass




@pytest.mark.parametrize(
    "north_displacement_x, north_displacement_y",
    [
        (1.0,-1.0,"north_displacement_x",0.0),
        (1.0,1.0,"north_displacement_x","north_displacement_y"),
    ],
)

def test_2d_single_fracture(north_displacement_x, north_displacement_y,u_e_expected,u_p_expected):
    """Test that the solution is qualitatively sound.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """
    # Instantiate constants and store in params.
    solid_vals = {  "tangential_fracture_stiffness":0.01,
                    "shear_modulus": 10000,
                    "lame_lambda": 10000,
                    }
    solid = SolidConstantsWithTangentialStiffness(solid_vals)
    params = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid},
        "uy_north": north_displacement_y,
        "ux_north": north_displacement_x,
    }

    # Create model and run simulation
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup, params)

    # Check that the pressure is linear

    sd = setup.mdg.subdomains(dim=setup.nd)
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)
    
    #var = setup.equation_system.get_variables([setup.displacement_jump,setup.elastic_displacement_jump,setup.plastic_displacement_jump,], [sd])
    #vals = setup.equation_system.get_variable_values(variables=var, time_step_index=0)

    u_p=setup.plastic_displacement_jump(sd_frac).value(setup.equation_system)

    u_e=setup.elastic_displacement_jump(sd_frac).value(setup.equation_system)

#    temp=setup.u
#    temp1=temp.tangential_component(sd)
#    jump = setup.u(sd).value(setup.equation_system)

    #var = setup.equation_system.get_variables([setup.u], [sd])
    #vals = setup.equation_system.get_variable_values(variables="u", time_step_index=0)
    #test=setup.mdg.subdomains(return_data=True)[0]
    #test1=test[pp.TIME_STEP_SOLUTIONS].keys()
    
    
    #subdomain_states = []
    #for sd, data in setup.mdg.subdomains(return_data=True):
    #    subdomain_states += data[pp.TIME_STEP_SOLUTIONS].values()
    
    #tor1=subdomain_states[0]
    #tor2=tor1.tangential_component(sd)

    #fetcher = pp.get_solution_values(name='u',data=setup.mdg.subdomains(return_data=True), time_step_index=0)

    u_real=setup.displacement(sd).value(setup.equation_system)


    print(f"u_e={u_e}")
    print(f"vals={u_real[::2]}")
    tol=1e-10
    
    temptest=u_real.reshape(2, -1, order='F')[0]

    assert np.allclose(u_e,temptest)

    #assert np.allclose(u_p,u_p_expected)


test_2d_single_fracture(1.0,-1.0,"north_displacement_x",0.0)