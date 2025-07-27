import math
import time

import taichi as ti
from util.taichi_utils import *
from util.memory_profile import sum_memory_usage

@ti.data_oriented
class MGPCG_Solid:
    '''
Grid-based MGPCG solver for the possion equation.

.. note::

    This solver only runs on CPU and CUDA backends since it requires the
    ``pointer`` SNode.
    '''
    def __init__(self, boundary_types, boundary_mask, N, dim=2, base_level=3, real=float):
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

        self.bm = [boundary_mask]
        for l in range(1, self.n_mg_levels):
            self.bm.append(ti.field(dtype=ti.i32, shape=[n // (2**l) for n in self.N]))
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

    @ti.kernel
    def downsample_bm(self, bm_fine: ti.template(), bm_coarse: ti.template()):
        for I in ti.grouped(bm_coarse):
            I2 = I * 2
            all_solid = 1
            range_d = 2 * ti.Vector.one(ti.i32, self.dim)
            for J in ti.grouped(ti.ndrange(*range_d)):
                if bm_fine[I2 + J] <= 0:
                    all_solid = 0
            
            bm_coarse[I] = all_solid

    @ti.func
    def neighbor_sum(self, x, I, bm):
        dims = x.shape
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # add right if has right (and also not boundary)
            if (I[i] < dims[i] - 1) and bm[I+offset] <= 0:
                ret += x[I + offset]
            # add left if has left (and also not boundary)
            if (I[i] > 0) and bm[I-offset] <= 0:
                ret += x[I - offset]
        return ret
    
    @ti.func
    def num_fluid_neighbors(self, x, I, bm): # l is the level
        dims = x.shape
        num = 2.0 * self.dim
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # check low
            if I[i] <= 0 and self.boundary_types[i,0] == 2: # if on lower boundary
                num -= 1.0
            elif I[i] > 0 and bm[I-offset] > 0:
                num -= 1.0
            # check high
            if I[i] >= dims[i] - 1 and self.boundary_types[i,1] == 2: # if on upper boundary
                num -= 1.0
            elif I[i] < dims[i] - 1 and bm[I+offset] > 0:
                num -= 1.0
            
        return num

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                multiplier = self.num_fluid_neighbors(self.p, I, self.bm[0])
                self.Ap[I] = multiplier * self.p[I] - self.neighbor_sum(
                    self.p, I, self.bm[0])

    @ti.kernel
    def get_Ap(self, p: ti.template(), Ap: ti.template()):
        for I in ti.grouped(Ap):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                multiplier = self.num_fluid_neighbors(p, I, self.bm[0])
                Ap[I] = multiplier * p[I] - self.neighbor_sum(
                    p, I, self.bm[0])

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if self.bm[l][I] <= 0: # only if a cell is not a solid, on level = l
                multiplier = self.num_fluid_neighbors(self.z[l], I, self.bm[l])
                res = self.r[l][I] - (multiplier * self.z[l][I] -
                                    self.neighbor_sum(self.z[l], I, self.bm[l]))
                self.r[l + 1][I // 2] += res * 1.0 / (self.dim-1.0)

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (self.bm[l][I] <= 0) and ((I.sum()) & 1 == phase): # only if a cell is not a solid, on level = l
                multiplier = self.num_fluid_neighbors(self.z[l], I, self.bm[l])
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I, self.bm[l])) / multiplier

    @ti.kernel
    def recenter(self, r: ti.template()): # so that the mean value of r is 0
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
              tol=1e-12,
              verbose=False):
        '''
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        '''
        # downsample boundary mask before each solve
        for l in range(1, self.n_mg_levels):
            self.downsample_bm(self.bm[l - 1], self.bm[l])

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

        # Conjugate gradients
        it = 0
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
                end_t = time.time()
                print("[MGPCG] Converged at iter: ", it, " with final error: ", math.sqrt(rTr), " using time: ", end_t-start_t)
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

        end_t = time.time()
        print("[MGPCG] Return without converging at iter: ", it, " with final error: ", math.sqrt(rTr), " using time: ", end_t-start_t)
    
    def get_memory_usage(self):
        return sum_memory_usage(
            [*self.bm[1:], *self.r, *self.z, self.x, self.p, self.Ap, self.alpha, self.beta, self.sum, self.r_mean])


class MGPCG_Solid_3(MGPCG_Solid):

    def __init__(self, boundary_types, boundary_mask, boundary_vel, N, base_level=3, real=float):
        super().__init__(boundary_types, boundary_mask, N, dim=3, base_level=base_level, real=real)

        self.u_div = ti.field(float, shape=N)
        self.boundary_types = boundary_types
        self.boundary_mask = boundary_mask
        self.boundary_vel = boundary_vel

        self.u_div_sum = ti.field(float, shape=())
        self.num_fluid_cells = ti.field(ti.i32, shape=())

    @ti.kernel
    def apply_bc(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = u_x.shape
        for i, j, k in u_x:
            if i == 0 and self.boundary_types[0,0] == 2:
                u_x[i,j,k] = 0
            if i == u_dim - 1 and self.boundary_types[0,1] == 2:
                u_x[i,j,k] = 0
        u_dim, v_dim, w_dim = u_y.shape
        for i, j, k in u_y:
            if j == 0 and self.boundary_types[1,0] == 2:
                u_y[i,j,k] = 0
            if j == v_dim - 1 and self.boundary_types[1,1] == 2:
                u_y[i,j,k] = 0
        u_dim, v_dim, w_dim = u_z.shape
        for i, j, k in u_z:
            if k == 0 and self.boundary_types[2,0] == 2:
                u_z[i,j,k] = 0
            if k == w_dim - 1 and self.boundary_types[2,1] == 2:
                u_z[i,j,k] = 0

        # account for boundary mask
        for i, j, k in self.boundary_mask:
            if self.boundary_mask[i,j,k] > 0:
                vel = self.boundary_vel[i,j,k]
                u_x[i,j,k] = vel.x
                u_x[i+1,j,k] = vel.x
                u_y[i,j,k] = vel.y
                u_y[i,j+1,k] = vel.y
                u_z[i,j,k] = vel.z
                u_z[i,j,k+1] = vel.z

    @ti.kernel
    def divergence(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        self.u_div_sum[None] *= 0
        self.num_fluid_cells[None] *= 0
        u_dim, v_dim, w_dim = self.u_div.shape
        for i, j, k in self.u_div:
            if self.boundary_mask[i,j,k] <= 0:
                vl = sample(u_x, i, j, k)
                vr = sample(u_x, i + 1, j, k)
                vb = sample(u_y, i, j, k)
                vt = sample(u_y, i, j + 1, k)
                va = sample(u_z, i, j, k)
                vc = sample(u_z, i, j, k + 1)
                self.u_div[i,j,k] = vr - vl + vt - vb + vc - va
                self.u_div_sum[None] += self.u_div[i,j,k]
                self.num_fluid_cells[None] += 1
            else:
                self.u_div[i,j,k] = 0.0
        
    @ti.kernel
    def correct_divergence(self):
        avg_correction = self.u_div_sum[None] / self.num_fluid_cells[None]
        for i, j, k in self.u_div:
            if self.boundary_mask[i,j,k] <= 0:
                self.u_div[i,j,k] -= avg_correction
                self.u_div_sum[None] -= avg_correction

    @ti.kernel
    def subtract_grad_p(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = self.p.shape
        for i, j, k in u_x:
            pr = sample(self.p, i, j, k)
            pl = sample(self.p, i-1, j, k)
            if i-1 < 0:
                pl = 0
            if i >= u_dim:
                pr = 0
            u_x[i,j,k] -= (pr - pl)
        for i, j, k in u_y:
            pt = sample(self.p, i, j, k)
            pb = sample(self.p, i, j-1, k)
            if j-1 < 0:
                pb = 0
            if j >= v_dim:
                pt = 0
            u_y[i,j,k] -= pt - pb
        for i, j, k in u_z:
            pc = sample(self.p, i, j, k)
            pa = sample(self.p, i, j, k-1)
            if k-1 < 0:
                pa = 0
            if k >= w_dim:
                pc = 0
            u_z[i,j,k] -= pc - pa

    def solve_pressure_MGPCG(self, verbose):
        self.init(self.u_div, -1)
        self.solve(max_iters=100, verbose=verbose, tol = 1.e-12)
        self.get_result(self.p)

    def Poisson(self, u_x, u_y, u_z, verbose = False):
        self.apply_bc(u_x, u_y, u_z)
        self.divergence(u_x, u_y, u_z)
        print("[Poisson] Sum of fluid div: ", self.u_div_sum[None])
        if self.u_div_sum[None] > 0.0:
            self.correct_divergence()
            print("[Poisson] Sum of fluid div after correction: ", self.u_div_sum[None])
        self.solve_pressure_MGPCG(verbose = verbose)
        self.subtract_grad_p(u_x, u_y, u_z)
        self.apply_bc(u_x, u_y, u_z)
    
    def get_memory_usage(self):
        memory_usage = super().get_memory_usage()
        memory_usage += sum_memory_usage([self.u_div])
        return memory_usage
