import sys
import argparse
from datetime import datetime
from util.taichi_utils import *
from util.io_utils import *
from util.timer import SimpleTimer
from util.logger import Logger
from util.memory_profile import sum_memory_usage
from sim.mgpcg import *
from sim.init_conditions import *
from sim.hermite import *

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Experiment name")
parser.add_argument("-c", "--case", help="Demo case number",
                    type=int, default=0)
args = parser.parse_args()

case = args.case

if case == 0:
    init_condition = "leapfrog"
    res_x = 384
    res_y = 128
    res_z = 128
    dx = 1./res_y
    visualize_dt = 0.1
    reinit_every = 20
    ckpt_every = 1
    CFL = 0.5
    from_frame = 0
    total_frames = 500
    BFECC_clamp = False
    real_t = ti.f32
    exp_name = "leapfrog"
elif case == 1:
    init_condition = "fourvorts"
    res_x = 128
    res_y = 128
    res_z = 256
    dx = 1./res_y
    visualize_dt = 0.1
    reinit_every = 20
    ckpt_every = 1
    CFL = 0.5
    from_frame = 0
    total_frames = 500
    BFECC_clamp = True
    real_t = ti.f32
    exp_name = "fourvorts"

if args.name is None:
    exp_name += datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
else:
    exp_name = args.name
logsdir = os.path.join('logs', exp_name)
os.makedirs(logsdir, exist_ok=True)

from_frame = max(0, from_frame)
if from_frame <= 0 and args.name is not None:
    remove_everything_in(logsdir)

logger = Logger(os.path.join(logsdir, 'log.txt'))
SimpleTimer.set_logger(logger)

ti.init(arch=ti.cuda, debug=False, default_fp=real_t)

boundary_types = ti.Matrix([[2, 2], [2, 2], [2, 2]], ti.i32)
solver = MGPCG_3(boundary_types=boundary_types, N=[
                 res_x, res_y, res_z], base_level=3, real=real_t)

X = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
X_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
X_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
X_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))
center_coords_func(X, dx)
x_coords_func(X_x, dx)
y_coords_func(X_y, dx)
z_coords_func(X_z, dx)

# back flow map
T_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
T_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
T_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))
psi_x = ti.Vector.field(3, float, shape=(
    res_x+1, res_y, res_z))
psi_y = ti.Vector.field(3, float, shape=(
    res_x, res_y+1, res_z))
psi_z = ti.Vector.field(3, float, shape=(
    res_x, res_y, res_z+1))
F_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
F_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
F_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))
phi_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
phi_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
phi_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))

u = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
w = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
u_z = ti.field(float, shape=(res_x, res_y, res_z+1))

u_x_cache = [ti.field(float, shape=(res_x+1, res_y, res_z))
             for _ in range(reinit_every)]
u_y_cache = [ti.field(float, shape=(res_x, res_y+1, res_z))
             for _ in range(reinit_every)]
u_z_cache = [ti.field(float, shape=(res_x, res_y, res_z+1))
             for _ in range(reinit_every)]

init_u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
init_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
init_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
err_u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
err_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
err_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
tmp_u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
tmp_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
tmp_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))

max_speed = ti.field(float, shape=())
dts = ti.field(float, shape=(reinit_every))

init_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
tmp_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
err_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))

memory_usage = sum_memory_usage([
    T_x, T_y, T_z, psi_x, psi_y, psi_z,
    *u_x_cache, *u_y_cache, *u_z_cache
])
logger.info(f"flow map mem usage: {memory_usage/2**30:.3f} GB")
memory_usage += sum_memory_usage([
    F_x, F_y, F_z, phi_x, phi_y, phi_z,
    u_x, u_y, u_z, init_u_x, init_u_y, init_u_z,
    err_u_x, err_u_y, err_u_z, tmp_u_x, tmp_u_y, tmp_u_z,
])
memory_usage += solver.get_memory_usage()
logger.info(f"gross sim mem usage: {memory_usage/2**30:.3f} GB")

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
    max_speed[None] = 1.e-3  # avoid dividing by zero
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        u = 0.5 * (u_x[i, j, k] + u_x[i+1, j, k])
        v = 0.5 * (u_y[i, j, k] + u_y[i, j+1, k])
        w = 0.5 * (u_z[i, j, k] + u_z[i, j, k+1])
        speed = ti.sqrt(u ** 2 + v ** 2 + w ** 2)
        ti.atomic_max(max_speed[None], speed)


@ti.kernel
def reset_to_identity(psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(),
                      T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y[I]
    for I in ti.grouped(psi_z):
        psi_z[I] = X_z[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)


def march_no_neural(psi_x, T_x, psi_y, T_y, psi_z, T_z, step):
    tmp_u_x.copy_from(u_x_cache[step])
    tmp_u_y.copy_from(u_y_cache[step])
    tmp_u_z.copy_from(u_z_cache[step])
    advance_map(psi_x, T_x, tmp_u_x, tmp_u_y, tmp_u_z, -dts[step])
    advance_map(psi_y, T_y, tmp_u_x, tmp_u_y, tmp_u_z, -dts[step])
    advance_map(psi_z, T_z, tmp_u_x, tmp_u_y, tmp_u_z, -dts[step])


def backtrack_psi_grid(curr_step):
    reset_to_identity(psi_x, psi_y, psi_z, T_x, T_y, T_z)
    advance_map(psi_x, T_x, u_x, u_y, u_z, -dts[curr_step])
    advance_map(psi_y, T_y, u_x, u_y, u_z, -dts[curr_step])
    advance_map(psi_z, T_z, u_x, u_y, u_z, -dts[curr_step])
    for step in reversed(range(curr_step)):
        march_no_neural(psi_x, T_x, psi_y, T_y, psi_z, T_z, step)


def march_phi_grid(curr_step):
    advance_map(phi_x, F_x, u_x, u_y, u_z, dts[curr_step])
    advance_map(phi_y, F_y, u_x, u_y, u_z, dts[curr_step])
    advance_map(phi_z, F_z, u_x, u_y, u_z, dts[curr_step])


@ti.kernel
def advance_map(psi_x: ti.template(), T_x: ti.template(),
                u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I]
        psi_x1 = psi_x[I] + 0.5 * dt * u1
        T_x1 = T_x[I] + 0.5 * dt * dT_x_dt1
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1
        psi_x2 = psi_x[I] + 0.5 * dt * u2
        T_x2 = T_x[I] + 0.5 * dt * dT_x_dt2
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2
        psi_x3 = psi_x[I] + 1.0 * dt * u3
        T_x3 = T_x[I] + 1.0 * dt * dT_x_dt3
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3
        psi_x[I] = psi_x[I] + dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + dt * 1./6 * \
            (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)


@ti.kernel
def map_u(u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
          u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(),
          T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
          psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    for I in ti.grouped(u_x1):
        u_at_psi = interp_u_MAC(u_x0, u_y0, u_z0, psi_x[I], dx)
        u_x1[I] = T_x[I].dot(u_at_psi)
    for I in ti.grouped(u_y1):
        u_at_psi = interp_u_MAC(u_x0, u_y0, u_z0, psi_y[I], dx)
        u_y1[I] = T_y[I].dot(u_at_psi)
    for I in ti.grouped(u_z1):
        u_at_psi = interp_u_MAC(u_x0, u_y0, u_z0, psi_z[I], dx)
        u_z1[I] = T_z[I].dot(u_at_psi)


@ti.kernel
def map_smoke(smoke0: ti.template(), smoke1: ti.template(),
              psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] +
                        psi_y[i, j, k] + psi_y[i, j+1, k] +
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        smoke1[i, j, k] = interp_1(smoke0, psi_c, dx)


@ti.kernel
def clamp_smoke(smoke0: ti.template(), smoke1: ti.template(),
                psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] +
                        psi_y[i, j, k] + psi_y[i, j+1, k] +
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        mini, maxi = sample_min_max_1(smoke0, psi_c, dx)
        smoke1[i, j, k] = ti.math.clamp(smoke1[i, j, k], mini, maxi)


@ti.kernel
def clamp_u(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i, j, k in u:
        u1_l = sample(u1, i-1, j, k)
        u1_r = sample(u1, i+1, j, k)
        u1_b = sample(u1, i, j-1, k)
        u1_t = sample(u1, i, j+1, k)
        u1_a = sample(u1, i, j, k-1)
        u1_c = sample(u1, i, j, k+1)
        maxi = ti.math.max(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        mini = ti.math.min(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        u2[i, j, k] = ti.math.clamp(u[i, j, k], mini, maxi)


def main(from_frame=0):
    ti.sync()
    runtime0 = time.time()

    vtkdir = "vtks"
    vtkdir = os.path.join(logsdir, vtkdir)
    os.makedirs(vtkdir, exist_ok=True)
    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)
    shutil.copy(__file__, logsdir)

    if from_frame <= 0:
        if init_condition == "leapfrog":
            init_vorts_leapfrog(X, u, smoke, tmp_smoke)
            split_central_vector(u, u_x, u_y, u_z)
        elif init_condition == "fourvorts":
            init_four_vorts(X, u, smoke, tmp_smoke)
            split_central_vector(u, u_x, u_y, u_z)
        solver.Poisson(u_x, u_y, u_z)
    else:
        u_x.from_numpy(np.load(os.path.join(
            ckptdir, "vel_x_numpy_" + str(from_frame) + ".npy")))
        u_y.from_numpy(np.load(os.path.join(
            ckptdir, "vel_y_numpy_" + str(from_frame) + ".npy")))
        u_z.from_numpy(np.load(os.path.join(
            ckptdir, "vel_z_numpy_" + str(from_frame) + ".npy")))
        smoke.from_numpy(np.load(os.path.join(
            ckptdir, "smoke_numpy_" + str(from_frame) + ".npy")))

    get_central_vector(u_x, u_y, u_z, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_norm = np.linalg.norm(w_numpy, axis=-1)
    smoke_numpy = smoke.to_numpy()
    smoke_norm = smoke_numpy[..., -1]
    write_vtks_4channel_smoke(w_norm, smoke_numpy, vtkdir, from_frame)

    np.save(os.path.join(ckptdir, "vel_x_numpy_" +
            str(from_frame)), u_x.to_numpy())
    np.save(os.path.join(ckptdir, "vel_y_numpy_" +
            str(from_frame)), u_y.to_numpy())
    np.save(os.path.join(ckptdir, "vel_z_numpy_" +
            str(from_frame)), u_z.to_numpy())

    sub_t = 0.
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0
    i = -1
    while True:
        SimpleTimer.start("substep")
        SimpleTimer.start("pre-advect")
        i += 1
        j = i % reinit_every

        # determine dt
        calc_max_speed(u_x, u_y, u_z)  # saved to max_speed[None]
        curr_dt = CFL * dx / max_speed[None]
        if sub_t+curr_dt >= visualize_dt:  # if over
            curr_dt = visualize_dt-sub_t
            sub_t = 0.  # empty sub_t
            frame_idx += 1
            output_frame = True
        elif (sub_t + curr_dt * 2) >= visualize_dt:
            curr_dt = (visualize_dt - sub_t)/2
            sub_t += curr_dt
            output_frame = False
        else:
            sub_t += curr_dt
            output_frame = False
        dts[j] = curr_dt

        if j == 0:
            reset_to_identity(phi_x, phi_y, phi_z, F_x,
                              F_y, F_z)
            init_u_x.copy_from(u_x)
            init_u_y.copy_from(u_y)
            init_u_z.copy_from(u_z)
            init_smoke.copy_from(smoke)
            num_reinits += 1

        advect_u_semi_lagrangian(u_x, u_y, u_z, tmp_u_x, tmp_u_y,
                                 tmp_u_z, dx, 0.5 * curr_dt, X_x, X_y, X_z)
        u_x.copy_from(tmp_u_x)
        u_y.copy_from(tmp_u_y)
        u_z.copy_from(tmp_u_z)
        SimpleTimer.stop("pre-advect", reinit_every+1)
        solver.Poisson(u_x, u_y, u_z)

        SimpleTimer.start("advect")
        u_x_cache[j].copy_from(u_x)
        u_y_cache[j].copy_from(u_y)
        u_z_cache[j].copy_from(u_z)

        backtrack_psi_grid(j)
        march_phi_grid(j)

        map_u(init_u_x, init_u_y, init_u_z, u_x, u_y, u_z,
              T_x, T_y, T_z, psi_x, psi_y, psi_z, dx)
        map_smoke(init_smoke, smoke, psi_x, psi_y, psi_z, dx)
        map_u(u_x, u_y, u_z, err_u_x, err_u_y, err_u_z,
              F_x, F_y, F_z, phi_x, phi_y, phi_z, dx)
        map_smoke(smoke, err_smoke, phi_x, phi_y, phi_z, dx)

        add_fields(err_u_x, init_u_x, err_u_x, -1.)
        add_fields(err_u_y, init_u_y, err_u_y, -1.)
        add_fields(err_u_z, init_u_z, err_u_z, -1.)
        add_fields(err_smoke, init_smoke, err_smoke, -1.)
        scale_field(err_u_x, 0.5, err_u_x)
        scale_field(err_u_y, 0.5, err_u_y)
        scale_field(err_u_z, 0.5, err_u_z)
        scale_field(err_smoke, 0.5, err_smoke)
        map_u(err_u_x, err_u_y, err_u_z,
              tmp_u_x, tmp_u_y, tmp_u_z,
              T_x, T_y, T_z, psi_x, psi_y, psi_z, dx)
        map_smoke(err_smoke, tmp_smoke, psi_x,
                  psi_y, psi_z, dx)
        add_fields(u_x, tmp_u_x, err_u_x, -1.)
        add_fields(u_y, tmp_u_y, err_u_y, -1.)
        add_fields(u_z, tmp_u_z, err_u_z, -1.)
        add_fields(smoke, tmp_smoke, smoke, -1.)
        clamp_smoke(init_smoke, smoke, psi_x, psi_y, psi_z, dx)
        if BFECC_clamp:
            clamp_u(err_u_x, u_x, tmp_u_x)
            clamp_u(err_u_y, u_y, tmp_u_y)
            clamp_u(err_u_z, u_z, tmp_u_z)
            u_x.copy_from(tmp_u_x)
            u_y.copy_from(tmp_u_y)
            u_z.copy_from(tmp_u_z)
        else:
            u_x.copy_from(err_u_x)
            u_y.copy_from(err_u_y)
            u_z.copy_from(err_u_z)

        SimpleTimer.stop("advect", reinit_every+1)

        solver.Poisson(u_x, u_y, u_z)

        SimpleTimer.stop("substep", reinit_every+1)

        relstep = visualize_dt / curr_dt
        relstep = int(relstep)
        logger.info(
            f"[Simulate] Done step {i} substep {j} relstep 1/{relstep}")

        if output_frame:
            # visualization
            get_central_vector(u_x, u_y, u_z, u)
            curl(u, w, dx)
            w_numpy = w.to_numpy()
            w_norm = np.linalg.norm(w_numpy, axis=-1)
            smoke_numpy = smoke.to_numpy()
            smoke_norm = smoke_numpy[..., -1]
            write_vtks_4channel_smoke(w_norm, smoke_numpy, vtkdir, frame_idx)

            if frame_idx % ckpt_every == 0:
                np.save(os.path.join(ckptdir, "vel_x_numpy_" +
                        str(frame_idx)), u_x.to_numpy())
                np.save(os.path.join(ckptdir, "vel_y_numpy_" +
                        str(frame_idx)), u_y.to_numpy())
                np.save(os.path.join(ckptdir, "vel_z_numpy_" +
                        str(frame_idx)), u_z.to_numpy())

            ti.sync()
            runtime1 = time.time()
            logger.info(
                f"[Simulate] Done Frame {frame_idx} substep {i-last_output_substep} time {runtime1-runtime0:.2f}s\n")
            last_output_substep = i

            if frame_idx >= total_frames:
                break


if __name__ == '__main__':
    logger.info("[Main] Begin")
    main(from_frame=from_frame)
    logger.info("[Main] Complete")
    SimpleTimer.print_times()
