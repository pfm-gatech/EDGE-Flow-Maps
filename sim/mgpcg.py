import math
import time

import taichi as ti
from util.taichi_utils import sample as sample3d
from util.taichi_utils import *
from util.memory_profile import sum_memory_usage

# based on https://github.com/yitongdeng-projects/neural_flow_maps_code/blob/main/mgpcg.py


@ti.data_oriented
class MGPCG:
    '''
Grid-based MGPCG solver for the possion equation.

.. note::

    This solver only runs on CPU and CUDA backends since it requires the
    ``pointer`` SNode.
    '''

    def __init__(self, boundary_types, N, dim=2, base_level=3, real=float):
        '''
        :parameter dim: Dimensionality of the fields.
        :parameter N: Grid resolutions.
        :parameter n_mg_levels: Number of multigrid levels.
        '''

        # grid parameters
        self.use_multigrid = True

        self.N = N
        self.n_mg_levels = int(math.log2(min(N))) - base_level + 1
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 20
        self.dim = dim
        self.real = real

        self.r = [
            ti.field(dtype=real, shape=[n // (2 ** l) for n in self.N])
            for l in range(self.n_mg_levels)
        ] # residual
        self.z = [
            ti.field(dtype=real, shape=[n // (2 ** l) for n in self.N])
            for l in range(self.n_mg_levels)
        ] # M^-1 self.r
        self.x = ti.field(dtype=self.real, shape=self.N) # solution
        self.p = ti.field(dtype=self.real, shape=self.N) # conjugate gradient
        self.Ap = ti.field(dtype=self.real, shape=self.N) # matrix-vector product
        self.alpha = ti.field(dtype=self.real, shape=()) # step size
        self.beta = ti.field(dtype=self.real, shape=()) # step size
        self.sum = ti.field(dtype=self.real, shape=()) # storage for reductions
        self.r_mean = ti.field(dtype=self.real, shape=()) # storage for avg of r
        self.num_entries = math.prod(self.N)

        self.boundary_types = boundary_types

    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        '''
        Set up the solver for $\nabla^2 x = k r$, a scaled Poisson problem.
        :parameter k: (scalar) A scaling factor of the right-hand side.
        :parameter r: (ti.field) Unscaled right-hand side.
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            self.init_r(I, r[I] * k)

    @ti.kernel
    def get_result(self, x: ti.template()):
        '''
        Get the solution field.

        :parameter x: (ti.field) The field to store the solution
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            x[I] = self.x[I]

    @ti.func
    def neighbor_sum(self, x, I):
        dims = x.shape
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # add right if has right
            if I[i] < dims[i] - 1:
                ret += x[I + offset]
            # add left if has left
            if I[i] > 0:
                ret += x[I - offset]
        return ret

    @ti.func
    def num_fluid_neighbors(self, x, I):
        dims = x.shape
        num = 2.0 * self.dim
        for i in ti.static(range(self.dim)):
            if I[i] <= 0 and (self.boundary_types[i, 0] == 2):
                num -= 1.0
            if I[i] >= dims[i] - 1 and (self.boundary_types[i, 1] == 2):
                num -= 1.0
        return num

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            multiplier = self.num_fluid_neighbors(self.p, I)
            self.Ap[I] = multiplier * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def get_Ap(self, p: ti.template(), Ap: ti.template()):
        for I in ti.grouped(Ap):
            multiplier = self.num_fluid_neighbors(p, I)
            Ap[I] = multiplier * p[I] - self.neighbor_sum(
                p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            multiplier = self.num_fluid_neighbors(self.z[l], I)
            res = self.r[l][I] - (multiplier * self.z[l][I] -
                                  self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += res * 1.0 / (self.dim-1.0)

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                multiplier = self.num_fluid_neighbors(self.z[l], I)
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / multiplier

    @ti.kernel
    def recenter(self, r: ti.template()):  # so that the mean value of r is 0
        self.r_mean[None] = 0.0
        for I in ti.grouped(r):
            self.r_mean[None] += r[I] / self.num_entries
        for I in ti.grouped(r):
            r[I] -= self.r_mean[None]

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self,
              max_iters=-1,
              eps=1e-12,
              abs_tol=1e-12,
              rel_tol=1e-12,
              verbose=False):
        '''
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        '''
        all_neumann = (self.boundary_types.sum() == 2 * 2 * self.dim)

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p

        if all_neumann:
            self.recenter(self.r[0])
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        tol = max(abs_tol, old_zTr * rel_tol)

        # Conjugate gradients
        it = 0
        ti.sync()
        start_t = time.time()
        while max_iters == -1 or it < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f'iter {it}, |residual|_2={math.sqrt(rTr)}')

            if rTr < tol:
                ti.sync()
                end_t = time.time()
                print("[MGPCG] Converged at iter: ", it, " with final error: ", math.sqrt(
                    rTr), " using time: ", end_t-start_t)
                return

            if all_neumann:
                self.recenter(self.r[0])
            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            it += 1

        ti.sync()
        end_t = time.time()
        print("[MGPCG] Return without converging at iter: ", it,
              " with final error: ", math.sqrt(rTr), " using time: ", end_t-start_t)

    def get_memory_usage(self):
        return sum_memory_usage(
            [*self.r, *self.z, self.x, self.p, self.Ap, self.alpha, self.beta, self.sum, self.r_mean])


class MGPCG_2(MGPCG):

    def __init__(self, boundary_types, N, base_level=3, abs_tol=1e-12, rel_tol=1e-12, real=float):
        super().__init__(boundary_types, N, dim=2, base_level=base_level, real=real)

        self.u_div = ti.field(float, shape=N)
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    @ti.kernel
    def apply_bc(self, u_horizontal: ti.template(), u_vertical: ti.template()):
        u_dim, v_dim = u_horizontal.shape
        for i, j in u_horizontal:
            if i == 0 and self.boundary_types[0, 0] == 2:
                u_horizontal[i, j] = 0
            if i == u_dim - 1 and self.boundary_types[0, 1] == 2:
                u_horizontal[i, j] = 0

        u_dim, v_dim = u_vertical.shape
        for i, j in u_vertical:
            if j == 0 and self.boundary_types[1, 0] == 2:
                u_vertical[i, j] = 0
            if j == v_dim - 1 and self.boundary_types[1, 1] == 2:
                u_vertical[i, j] = 0

    @ti.kernel
    def divergence(self, u_horizontal: ti.template(), u_vertical: ti.template()):
        u_dim, v_dim = self.u_div.shape
        for i, j in self.u_div:
            vl = sample2d(u_horizontal, i, j)
            vr = sample2d(u_horizontal, i + 1, j)
            vb = sample2d(u_vertical, i, j)
            vt = sample2d(u_vertical, i, j + 1)
            self.u_div[i, j] = vr - vl + vt - vb

    @ti.kernel
    def subtract_grad_p(self, u_horizontal: ti.template(), u_vertical: ti.template()):
        u_dim, v_dim = self.p.shape
        for i, j in u_horizontal:
            pr = sample2d(self.p, i, j)
            pl = sample2d(self.p, i-1, j)
            if i-1 < 0:
                pl = 0
            if i >= u_dim:
                pr = 0
            u_horizontal[i, j] -= (pr - pl)
        for i, j in u_vertical:
            pt = sample2d(self.p, i, j)
            pb = sample2d(self.p, i, j-1)
            if j-1 < 0:
                pb = 0
            if j >= v_dim:
                pt = 0
            u_vertical[i, j] -= pt - pb

    def solve_pressure_MGPCG(self, max_iters=400, verbose=False):
        self.init(self.u_div, -1)
        self.solve(max_iters=max_iters, abs_tol=self.abs_tol,
                   rel_tol=self.rel_tol, verbose=verbose)
        self.get_result(self.p)

    def Poisson(self, u_horizontal, u_vertical, max_iters=400, verbose=False):
        self.apply_bc(u_horizontal, u_vertical)
        self.divergence(u_horizontal, u_vertical)
        self.solve_pressure_MGPCG(max_iters=max_iters, verbose=verbose)
        self.subtract_grad_p(u_horizontal, u_vertical)
        self.apply_bc(u_horizontal, u_vertical)

    def get_memory_usage(self):
        memory_usage = super().get_memory_usage()
        memory_usage += sum_memory_usage([self.u_div])
        return memory_usage


class MGPCG_3(MGPCG):

    def __init__(self, boundary_types, N, base_level=3, abs_tol=1e-12, rel_tol=1e-12, real=float):
        super().__init__(boundary_types, N, dim=3, base_level=base_level, real=real)

        self.u_div = ti.field(float, shape=N)
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    @ti.kernel
    def apply_bc(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = u_x.shape
        for i, j, k in u_x:
            if i == 0 and self.boundary_types[0, 0] == 2:
                u_x[i, j, k] = 0
            if i == u_dim - 1 and self.boundary_types[0, 1] == 2:
                u_x[i, j, k] = 0
        u_dim, v_dim, w_dim = u_y.shape
        for i, j, k in u_y:
            if j == 0 and self.boundary_types[1, 0] == 2:
                u_y[i, j, k] = 0
            if j == v_dim - 1 and self.boundary_types[1, 1] == 2:
                u_y[i, j, k] = 0
        u_dim, v_dim, w_dim = u_z.shape
        for i, j, k in u_z:
            if k == 0 and self.boundary_types[2, 0] == 2:
                u_z[i, j, k] = 0
            if k == w_dim - 1 and self.boundary_types[2, 1] == 2:
                u_z[i, j, k] = 0

    @ti.kernel
    def divergence(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = self.u_div.shape
        for i, j, k in self.u_div:
            vl = sample3d(u_x, i, j, k)
            vr = sample3d(u_x, i + 1, j, k)
            vb = sample3d(u_y, i, j, k)
            vt = sample3d(u_y, i, j + 1, k)
            va = sample3d(u_z, i, j, k)
            vc = sample3d(u_z, i, j, k + 1)
            self.u_div[i, j, k] = vr - vl + vt - vb + vc - va

    @ti.kernel
    def subtract_grad_p(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = self.p.shape
        for i, j, k in u_x:
            pr = sample3d(self.p, i, j, k)
            pl = sample3d(self.p, i-1, j, k)
            if i-1 < 0:
                pl = 0
            if i >= u_dim:
                pr = 0
            u_x[i, j, k] -= (pr - pl)
        for i, j, k in u_y:
            pt = sample3d(self.p, i, j, k)
            pb = sample3d(self.p, i, j-1, k)
            if j-1 < 0:
                pb = 0
            if j >= v_dim:
                pt = 0
            u_y[i, j, k] -= pt - pb
        for i, j, k in u_z:
            pc = sample3d(self.p, i, j, k)
            pa = sample3d(self.p, i, j, k-1)
            if k-1 < 0:
                pa = 0
            if k >= w_dim:
                pc = 0
            u_z[i, j, k] -= pc - pa

    def solve_pressure_MGPCG(self, max_iters=400, verbose=False):
        self.init(self.u_div, -1)
        self.solve(max_iters=400, verbose=verbose,
                   abs_tol=self.abs_tol, rel_tol=self.rel_tol)
        self.get_result(self.p)

    def Poisson(self, u_x, u_y, u_z, max_iters=400, verbose=False):
        self.apply_bc(u_x, u_y, u_z)
        self.divergence(u_x, u_y, u_z)
        self.solve_pressure_MGPCG(max_iters=max_iters, verbose=verbose)
        self.subtract_grad_p(u_x, u_y, u_z)
        self.apply_bc(u_x, u_y, u_z)

    @ti.kernel
    def subtract_grad_p_with_buffer(
        self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template(),
        f_x: ti.template(), f_y: ti.template(), f_z: ti.template()
    ):
        u_dim, v_dim, w_dim = self.p.shape
        for i, j, k in u_x:
            pr = sample3d(self.p, i, j, k)
            pl = sample3d(self.p, i-1, j, k)
            if i-1 < 0:
                pl = 0
            if i >= u_dim:
                pr = 0
            u_x[i, j, k] -= (pr - pl)
            f_x[i, j, k] = -(pr - pl)
        for i, j, k in u_y:
            pt = sample3d(self.p, i, j, k)
            pb = sample3d(self.p, i, j-1, k)
            if j-1 < 0:
                pb = 0
            if j >= v_dim:
                pt = 0
            u_y[i, j, k] -= pt - pb
            f_y[i, j, k] = -(pt - pb)
        for i, j, k in u_z:
            pc = sample3d(self.p, i, j, k)
            pa = sample3d(self.p, i, j, k-1)
            if k-1 < 0:
                pa = 0
            if k >= w_dim:
                pc = 0
            u_z[i, j, k] -= pc - pa
            f_z[i, j, k] = -(pc - pa)

    def Poisson_with_buffer(self, u_x, u_y, u_z,f_x,f_y,f_z, max_iters=400, verbose=False):
        self.apply_bc(u_x, u_y, u_z)
        self.divergence(u_x, u_y, u_z)
        self.solve_pressure_MGPCG(max_iters=max_iters, verbose=verbose)
        self.subtract_grad_p_with_buffer(u_x, u_y, u_z,f_x,f_y,f_z)
        self.apply_bc(u_x, u_y, u_z)

    def get_memory_usage(self):
        memory_usage = super().get_memory_usage()
        memory_usage += sum_memory_usage([self.u_div])
        return memory_usage
