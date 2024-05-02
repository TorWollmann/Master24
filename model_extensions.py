from functools import partial
import numpy as np
import porepy as pp

class ElastoPlasticFractureGap:
    def elastic_displacement_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return an operator that represents the elastic component of the displacement jump."""
        # TODO: Implement the elastic displacement jump. Is this where the equation:
        # T_t = K_t u_t is implemented?
        t_t = self.contact_traction(subdomains)
        K_t = self.solid.tangential_fracture_stiffness()
        u_elastic = t_t / K_t
        return u_elastic

    def plastic_displacement_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return an operator that represents the plastic component of the displacement jump."""
        total_jump = self.self.displacement_jump(subdomains)
        elastic_jump = self.elastic_displacement_jump(subdomains)
        return total_jump - elastic_jump

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the tangential constraints.

        The function reads
        .. math::
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)

        with `u` being displacement jump increments, `t` denoting tangential component
        and `b_p` the friction bound.

        For `b_p = 0`, the equation `C_t = 0` does not in itself imply `T_t = 0`, which
        is what the contact conditions require. The case is handled through the use of a
        characteristic function.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            complementary_eq: Contact mechanics equation for the tangential constraints.

        """
        # The lines below is an implementation of equations (25) and (27) in the paper
        #
        # Berge et al. (2020): Finite volume discretization for poroelastic media with
        #   fractures modeled by contact mechanics (IJNME, DOI: 10.1002/nme.6238). The
        #
        # Note that:
        #  - We do not directly implement the matrix elements of the contact traction
        #    as are derived by Berge in their equations (28)-(32). Instead, we directly
        #    implement the complimentarity function, and let the AD framework take care
        #    of the derivatives.
        #  - Related to the previous point, we do not implement the regularization that
        #    is discussed in Section 3.2.1 of the paper.

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        # Ignore mypy complaint on unknown keyword argument
        tangential_basis: list[pp.ad.SparseArray] = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis functions have shape (Nc * (self.nd - 1), Nc), where Nc is
        # the total number of cells in the subdomain. The sum will have the same shape,
        # but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell. EK: mypy insists
        # that the argument to sum should be a list of booleans. Ignore this error.
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        # TODO: Make the increment relative to current reference state, which might differ
        # from the current time state.

        # NOTE: This is the increment of PLASTIC deformation jump.
        u_t_increment: pp.ad.Operator = u_t - u_t.previous_timestep()

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")


        # The numerical constant is used to loosen the sensitivity in the transition
        # between sticking and sliding.
        # Expanding using only left multiplication to with scalar_to_tangential does not
        # work for an array, unlike the operators below. Arrays need right
        # multiplication as well.
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        # The numerical parameter is a cell-wise scalar which must be extended to a
        # vector quantity to be used in the equation (multiplied from the right).
        # Spelled out, from the right: Restrict the vector quantity to one dimension in
        # the tangential plane (e_i.T), multiply with the numerical parameter, prolong
        # to the full vector quantity (e_i), and sum over all all directions in the
        # tangential plane. EK: mypy insists that the argument to sum should be a list
        # of booleans. Ignore this error.
        c_num = pp.ad.sum_operator_list(
            [e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis]
        )

        # Combine the above into expressions that enter the equation. c_num will
        # effectively be a sum of SparseArrays, thus we use a matrix-vector product @
        tangential_sum = t_t + c_num @ u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        # Remove parentheses to make the equation more readable if possible. The product
        # between (the SparseArray) scalar_to_tangential and b_p is of matrix-vector
        # type (thus @), and the result is then multiplied elementwise with
        # tangential_sum.
        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        # For the use of @, see previous comment.
        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)
    
        # The characteristic function below reads "1 if (abs(b_p) < tol) else 0".
        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases. The tolerance is a numerical method parameter
        # and can be tailored.
        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)

        # Compose the equation itself. The last term handles the case bound=0, in which
        # case t_t = 0 cannot be deduced from the standard version of the complementary
        # function (i.e. without the characteristic function). Filter out the other
        # terms in this case to improve convergence
        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def tangential_fracture_stiffness(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return the tangential component of the fracture stiffness."""
        return pp.ad.Scalar(self.solid.tangential_fracture_stiffness())
    

    def shear_dilation_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear dilation [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise shear dilation.

        """
        angle: pp.ad.Operator = self.dilation_angle(subdomains)
        f_norm = pp.ad.Function(
            partial(pp.ad.functions.l2_norm, self.nd - 1), "norm_function"
        )
        f_tan = pp.ad.Function(pp.ad.functions.tan, "tan_function")
        shear_dilation: pp.ad.Operator = f_tan(angle) * f_norm(
            self.tangential_component(subdomains) @ self.plastic_displacement_jump(subdomains)
        )

        shear_dilation.set_name("shear_dilation")
        return shear_dilation
    
class MyMomentumBalance(ElastoPlasticFractureGap, pp.momentum_balance.MomentumBalance):
    ...



    # script med geometri orthognal fracture og randbetingelser