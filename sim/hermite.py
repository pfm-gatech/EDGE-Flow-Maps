from util.taichi_utils import *


@ti.func
def hermite_f(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 1.0:
        result = 2*abs_x**3-3*abs_x**2+1
    return result


@ti.func
def hermite_df(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 1.0:
        result = 6*abs_x**2-6*abs_x
    if (x < 0):
        result = -result
    return result


@ti.func
def hermite_g(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 1.0:
        result = abs_x**3-2*abs_x**2 + abs_x
    if (x < 0):
        result = -result
    return result


@ti.func
def hermite_dg(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 1.0:
        result = 3*abs_x**2-4*abs_x + 1
    return result


@ti.func
def hermite_kernel(x, ind):
    res = 0.0
    if (ind == 0):
        res = hermite_f(x)
    else:
        res = hermite_g(x)
    return res


@ti.func
def hermite_dkernel(x, ind):
    res = 0.0
    if (ind == 0):
        res = hermite_df(x)
    else:
        res = hermite_dg(x)
    return res


@ti.func
def interp_hermite_fd_unroll(
    vf, index, grad,
    p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5
):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    interped = 0.

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)

                hxf = hermite_f(x_p_x_i)
                hxg = hermite_g(x_p_x_i)
                hyf = hermite_f(y_p_y_i)
                hyg = hermite_g(y_p_y_i)
                hzf = hermite_f(z_p_z_i)
                hzg = hermite_g(z_p_z_i)

                ii = iu + i
                jj = iv + j
                kk = iw + k

                value = sample(vf, ii, jj, kk)[index]
                interped += value * hxf * hyf * hzf
                value_1 = sample(grad, ii, jj, kk) * dx
                interped += value_1[0] * hxg * hyf * hzf
                interped += value_1[1] * hxf * hyg * hzf
                interped += value_1[2] * hxf * hyf * hzg

                # fmt: off
                value_2x = (
                    (sample(grad, ii, jj, kk+1)[1]-sample(grad, ii, jj, kk-1)[1]) +
                    (sample(grad, ii, jj+1, kk)[2]-sample(grad, ii, jj-1, kk)[2])
                )/4*dx
                value_2y = (
                    (sample(grad, ii, jj, kk+1)[0]-sample(grad, ii, jj, kk-1)[0]) +
                    (sample(grad, ii+1, jj, kk)[2]-sample(grad, ii-1, jj, kk)[2])
                )/4*dx
                value_2z = (
                    (sample(grad, ii, jj+1, kk)[0]-sample(grad, ii, jj-1, kk)[0]) +
                    (sample(grad, ii+1, jj, kk)[1]-sample(grad, ii-1, jj, kk)[1])
                )/4*dx

                interped += value_2x * hxf * hyg * hzg
                interped += value_2y * hxg * hyf * hzg
                interped += value_2z * hxg * hyg * hzf

                value_30 = (
                    (sample(grad, ii+1, jj, kk+1)[1]-sample(grad, ii+1, jj, kk-1)[1]) +
                    (sample(grad, ii+1, jj+1, kk)[2]-sample(grad, ii+1, jj-1, kk)[2])
                )/4
                value_31 = (
                    (sample(grad, ii-1, jj, kk+1)[1]-sample(grad, ii-1, jj, kk-1)[1]) +
                    (sample(grad, ii-1, jj+1, kk)[2]-sample(grad, ii-1, jj-1, kk)[2])
                )/4
                value_32 = (
                    (sample(grad, ii, jj+1, kk+1)[0]-sample(grad, ii, jj+1, kk-1)[0]) +
                    (sample(grad, ii+1, jj+1, kk)[2]-sample(grad, ii-1, jj+1, kk)[2])
                )/4
                value_33 = (
                    (sample(grad, ii, jj-1, kk+1)[0]-sample(grad, ii, jj-1, kk-1)[0]) +
                    (sample(grad, ii+1, jj-1, kk)[2]-sample(grad, ii-1, jj-1, kk)[2])
                )/4
                value_34 = (
                    (sample(grad, ii, jj+1, kk+1)[0]-sample(grad, ii, jj-1, kk+1)[0]) +
                    (sample(grad, ii+1, jj, kk+1)[1]-sample(grad, ii-1, jj, kk+1)[1])
                )/4
                value_35 = (
                    (sample(grad, ii, jj+1, kk-1)[0]-sample(grad, ii, jj-1, kk-1)[0]) +
                    (sample(grad, ii+1, jj, kk-1)[1]-sample(grad, ii-1, jj, kk-1)[1])
                )/4
                # fmt: on

                value_3 = (value_30-value_31+value_32 -
                           value_33+value_34-value_35)/6*dx
                interped += value_3 * hxg * hyg * hzg

    return interped


@ti.func
def interp_grad_hermite_fd_unroll(
    vf, index, grad,
    p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5
):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.
    inv_dx = 1./dx

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)

                hxf = hermite_f(x_p_x_i)
                hxg = hermite_g(x_p_x_i)
                dhxf = hermite_df(x_p_x_i)
                dhxg = hermite_dg(x_p_x_i)

                hyf = hermite_f(y_p_y_i)
                hyg = hermite_g(y_p_y_i)
                dhyf = hermite_df(y_p_y_i)
                dhyg = hermite_dg(y_p_y_i)

                hzf = hermite_f(z_p_z_i)
                hzg = hermite_g(z_p_z_i)
                dhzf = hermite_df(z_p_z_i)
                dhzg = hermite_dg(z_p_z_i)

                ii = iu + i
                jj = iv + j
                kk = iw + k

                value = sample(vf, ii, jj, kk)[index]
                interped += value * hxf * hyf * hzf
                partial_x += value * dhxf * hyf * hzf
                partial_y += value * hxf * dhyf * hzf
                partial_z += value * hxf * hyf * dhzf
                value_1 = sample(grad, ii, jj, kk) * dx
                interped += value_1[0] * hxg * hyf * hzf
                partial_x += value_1[0] * dhxg * hyf * hzf
                partial_y += value_1[0] * hxg * dhyf * hzf
                partial_z += value_1[0] * hxg * hyf * dhzf
                interped += value_1[1] * hxf * hyg * hzf
                partial_x += value_1[1] * dhxf * hyg * hzf
                partial_y += value_1[1] * hxf * dhyg * hzf
                partial_z += value_1[1] * hxf * hyg * dhzf
                interped += value_1[2] * hxf * hyf * hzg
                partial_x += value_1[2] * dhxf * hyf * hzg
                partial_y += value_1[2] * hxf * dhyf * hzg
                partial_z += value_1[2] * hxf * hyf * dhzg

                # fmt: off
                value_2x = (
                    (sample(grad, ii, jj, kk+1)[1]-sample(grad, ii, jj, kk-1)[1]) +
                    (sample(grad, ii, jj+1, kk)[2]-sample(grad, ii, jj-1, kk)[2])
                )/4*dx
                value_2y = (
                    (sample(grad, ii, jj, kk+1)[0]-sample(grad, ii, jj, kk-1)[0]) +
                    (sample(grad, ii+1, jj, kk)[2]-sample(grad, ii-1, jj, kk)[2])
                )/4*dx
                value_2z = (
                    (sample(grad, ii, jj+1, kk)[0]-sample(grad, ii, jj-1, kk)[0]) +
                    (sample(grad, ii+1, jj, kk)[1]-sample(grad, ii-1, jj, kk)[1])
                )/4*dx

                interped += value_2x * hxf * hyg * hzg
                partial_x += value_2x * dhxf * hyg * hzg
                partial_y += value_2x * hxf * dhyg * hzg
                partial_z += value_2x * hxf * hyg * dhzg
                interped += value_2y * hxg * hyf * hzg
                partial_x += value_2y * dhxg * hyf * hzg
                partial_y += value_2y * hxg * dhyf * hzg
                partial_z += value_2y * hxg * hyf * dhzg
                interped += value_2z * hxg * hyg * hzf
                partial_x += value_2z * dhxg * hyg * hzf
                partial_y += value_2z * hxg * dhyg * hzf
                partial_z += value_2z * hxg * hyg * dhzf

                value_30 = (
                    (sample(grad, ii+1, jj, kk+1)[1]-sample(grad, ii+1, jj, kk-1)[1]) +
                    (sample(grad, ii+1, jj+1, kk)[2]-sample(grad, ii+1, jj-1, kk)[2])
                )/4
                value_31 = (
                    (sample(grad, ii-1, jj, kk+1)[1]-sample(grad, ii-1, jj, kk-1)[1]) +
                    (sample(grad, ii-1, jj+1, kk)[2]-sample(grad, ii-1, jj-1, kk)[2])
                )/4
                value_32 = (
                    (sample(grad, ii, jj+1, kk+1)[0]-sample(grad, ii, jj+1, kk-1)[0]) +
                    (sample(grad, ii+1, jj+1, kk)[2]-sample(grad, ii-1, jj+1, kk)[2])
                )/4
                value_33 = (
                    (sample(grad, ii, jj-1, kk+1)[0]-sample(grad, ii, jj-1, kk-1)[0]) +
                    (sample(grad, ii+1, jj-1, kk)[2]-sample(grad, ii-1, jj-1, kk)[2])
                )/4
                value_34 = (
                    (sample(grad, ii, jj+1, kk+1)[0]-sample(grad, ii, jj-1, kk+1)[0]) +
                    (sample(grad, ii+1, jj, kk+1)[1]-sample(grad, ii-1, jj, kk+1)[1])
                )/4
                value_35 = (
                    (sample(grad, ii, jj+1, kk-1)[0]-sample(grad, ii, jj-1, kk-1)[0]) +
                    (sample(grad, ii+1, jj, kk-1)[1]-sample(grad, ii-1, jj, kk-1)[1])
                )/4
                # fmt: on

                value_3 = (value_30-value_31+value_32 -
                           value_33+value_34-value_35)/6*dx
                interped += value_3 * hxg * hyg * hzg
                partial_x += value_3 * dhxg * hyg * hzg
                partial_y += value_3 * hxg * dhyg * hzg
                partial_z += value_3 * hxg * hyg * dhzg

    partial_x *= inv_dx
    partial_y *= inv_dx
    partial_z *= inv_dx

    return interped, ti.Vector([partial_x, partial_y, partial_z])


@ti.func
def interp_hermite_garm_fd_unroll(
    vf, index, grad, grad_grad,
    p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5
):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    interped = 0.

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)

                hxf = hermite_f(x_p_x_i)
                hxg = hermite_g(x_p_x_i)
                hyf = hermite_f(y_p_y_i)
                hyg = hermite_g(y_p_y_i)
                hzf = hermite_f(z_p_z_i)
                hzg = hermite_g(z_p_z_i)

                ii = iu + i
                jj = iv + j
                kk = iw + k

                value = sample(vf, ii, jj, kk)[index]
                interped += value * hxf * hyf * hzf
                value_1 = sample(grad, ii, jj, kk) * dx
                interped += value_1[0] * hxg * hyf * hzf
                interped += value_1[1] * hxf * hyg * hzf
                interped += value_1[2] * hxf * hyf * hzg
                value_2 = sample(grad_grad, ii, jj, kk) * dx * dx
                interped += value_2[0] * hxf * hyg * hzg
                interped += value_2[1] * hxg * hyf * hzg
                interped += value_2[2] * hxg * hyg * hzf
                # fmt: off
                value_3 = (
                    (sample(grad_grad, ii+1, jj, kk)[0]-sample(grad_grad, ii-1, jj, kk)[0]) +
                    (sample(grad_grad, ii, jj+1, kk)[1]-sample(grad_grad, ii, jj-1, kk)[1]) +
                    (sample(grad_grad, ii, jj, kk+1)[2]-sample(grad_grad, ii, jj, kk-1)[2])
                )/6 * dx * dx
                # fmt: on
                interped += value_3 * hxg * hyg * hzg
    return interped


@ti.func
def interp_grad_hermite_garm_fd_unroll(
    vf, index, grad, grad_grad,
    p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5
):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.
    inv_dx = 1./dx

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)

                hxf = hermite_f(x_p_x_i)
                hxg = hermite_g(x_p_x_i)
                dhxf = hermite_df(x_p_x_i)
                dhxg = hermite_dg(x_p_x_i)

                hyf = hermite_f(y_p_y_i)
                hyg = hermite_g(y_p_y_i)
                dhyf = hermite_df(y_p_y_i)
                dhyg = hermite_dg(y_p_y_i)

                hzf = hermite_f(z_p_z_i)
                hzg = hermite_g(z_p_z_i)
                dhzf = hermite_df(z_p_z_i)
                dhzg = hermite_dg(z_p_z_i)

                ii = iu + i
                jj = iv + j
                kk = iw + k

                value = sample(vf, ii, jj, kk)[index]
                interped += value * hxf * hyf * hzf
                partial_x += value * dhxf * hyf * hzf
                partial_y += value * hxf * dhyf * hzf
                partial_z += value * hxf * hyf * dhzf
                value_1 = sample(grad, ii, jj, kk) * dx
                interped += value_1[0] * hxg * hyf * hzf
                partial_x += value_1[0] * dhxg * hyf * hzf
                partial_y += value_1[0] * hxg * dhyf * hzf
                partial_z += value_1[0] * hxg * hyf * dhzf
                interped += value_1[1] * hxf * hyg * hzf
                partial_x += value_1[1] * dhxf * hyg * hzf
                partial_y += value_1[1] * hxf * dhyg * hzf
                partial_z += value_1[1] * hxf * hyg * dhzf
                interped += value_1[2] * hxf * hyf * hzg
                partial_x += value_1[2] * dhxf * hyf * hzg
                partial_y += value_1[2] * hxf * dhyf * hzg
                partial_z += value_1[2] * hxf * hyf * dhzg
                value_2 = sample(grad_grad, ii, jj, kk) * dx * dx
                interped += value_2[0] * hxf * hyg * hzg
                partial_x += value_2[0] * dhxf * hyg * hzg
                partial_y += value_2[0] * hxf * dhyg * hzg
                partial_z += value_2[0] * hxf * hyg * dhzg
                interped += value_2[1] * hxg * hyf * hzg
                partial_x += value_2[1] * dhxg * hyf * hzg
                partial_y += value_2[1] * hxg * dhyf * hzg
                partial_z += value_2[1] * hxg * hyf * dhzg
                interped += value_2[2] * hxg * hyg * hzf
                partial_x += value_2[2] * dhxg * hyg * hzf
                partial_y += value_2[2] * hxg * dhyg * hzf
                partial_z += value_2[2] * hxg * hyg * dhzf
                # fmt: off
                value_3 = (
                    (sample(grad_grad, ii+1, jj, kk)[0]-sample(grad_grad, ii-1, jj, kk)[0]) +
                    (sample(grad_grad, ii, jj+1, kk)[1]-sample(grad_grad, ii, jj-1, kk)[1]) +
                    (sample(grad_grad, ii, jj, kk+1)[2]-sample(grad_grad, ii, jj, kk-1)[2])
                )/6 * dx * dx
                # fmt: on
                interped += value_3 * hxg * hyg * hzg
                partial_x += value_3 * dhxg * hyg * hzg
                partial_y += value_3 * hxg * dhyg * hzg
                partial_z += value_3 * hxg * hyg * dhzg

    partial_x *= inv_dx
    partial_y *= inv_dx
    partial_z *= inv_dx

    return interped, ti.Vector([partial_x, partial_y, partial_z])


@ti.func
def interp_hermite(
    vf, index, grad, grad_grad, grad_grad_grad,
    p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5
):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    interped = 0.

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)

                hxf = hermite_f(x_p_x_i)
                hxg = hermite_g(x_p_x_i)
                hyf = hermite_f(y_p_y_i)
                hyg = hermite_g(y_p_y_i)
                hzf = hermite_f(z_p_z_i)
                hzg = hermite_g(z_p_z_i)

                ii = iu + i
                jj = iv + j
                kk = iw + k

                value = sample(vf, ii, jj, kk)[index]
                interped += value * hxf * hyf * hzf
                value_1 = sample(grad, ii, jj, kk) * dx
                interped += value_1[0] * hxg * hyf * hzf
                interped += value_1[1] * hxf * hyg * hzf
                interped += value_1[2] * hxf * hyf * hzg
                value_2 = sample(grad_grad, ii, jj, kk) * dx * dx
                interped += value_2[0] * hxf * hyg * hzg
                interped += value_2[1] * hxg * hyf * hzg
                interped += value_2[2] * hxg * hyg * hzf
                value_3 = sample(grad_grad_grad, iu + i, iv +
                                 j, iw + k) * dx * dx * dx
                interped += value_3 * hxg * hyg * hzg

    return interped


@ti.func
def interp_grad_hermite(
    vf, index, grad, grad_grad, grad_grad_grad,
    p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5
):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.
    inv_dx = 1./dx

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)

                hxf = hermite_f(x_p_x_i)
                hxg = hermite_g(x_p_x_i)
                dhxf = hermite_df(x_p_x_i)
                dhxg = hermite_dg(x_p_x_i)

                hyf = hermite_f(y_p_y_i)
                hyg = hermite_g(y_p_y_i)
                dhyf = hermite_df(y_p_y_i)
                dhyg = hermite_dg(y_p_y_i)

                hzf = hermite_f(z_p_z_i)
                hzg = hermite_g(z_p_z_i)
                dhzf = hermite_df(z_p_z_i)
                dhzg = hermite_dg(z_p_z_i)

                ii = iu + i
                jj = iv + j
                kk = iw + k

                value = sample(vf, ii, jj, kk)[index]
                interped += value * hxf * hyf * hzf
                partial_x += value * dhxf * hyf * hzf
                partial_y += value * hxf * dhyf * hzf
                partial_z += value * hxf * hyf * dhzf
                value_1 = sample(grad, ii, jj, kk) * dx
                interped += value_1[0] * hxg * hyf * hzf
                partial_x += value_1[0] * dhxg * hyf * hzf
                partial_y += value_1[0] * hxg * dhyf * hzf
                partial_z += value_1[0] * hxg * hyf * dhzf
                interped += value_1[1] * hxf * hyg * hzf
                partial_x += value_1[1] * dhxf * hyg * hzf
                partial_y += value_1[1] * hxf * dhyg * hzf
                partial_z += value_1[1] * hxf * hyg * dhzf
                interped += value_1[2] * hxf * hyf * hzg
                partial_x += value_1[2] * dhxf * hyf * hzg
                partial_y += value_1[2] * hxf * dhyf * hzg
                partial_z += value_1[2] * hxf * hyf * dhzg
                value_2 = sample(grad_grad, ii, jj, kk) * dx * dx
                interped += value_2[0] * hxf * hyg * hzg
                partial_x += value_2[0] * dhxf * hyg * hzg
                partial_y += value_2[0] * hxf * dhyg * hzg
                partial_z += value_2[0] * hxf * hyg * dhzg
                interped += value_2[1] * hxg * hyf * hzg
                partial_x += value_2[1] * dhxg * hyf * hzg
                partial_y += value_2[1] * hxg * dhyf * hzg
                partial_z += value_2[1] * hxg * hyf * dhzg
                interped += value_2[2] * hxg * hyg * hzf
                partial_x += value_2[2] * dhxg * hyg * hzf
                partial_y += value_2[2] * hxg * dhyg * hzf
                partial_z += value_2[2] * hxg * hyg * dhzf
                value_3 = sample(grad_grad_grad, ii, jj, kk) * dx * dx * dx
                interped += value_3 * hxg * hyg * hzg
                partial_x += value_3 * dhxg * hyg * hzg
                partial_y += value_3 * hxg * dhyg * hzg
                partial_z += value_3 * hxg * hyg * dhzg

    partial_x *= inv_dx
    partial_y *= inv_dx
    partial_z *= inv_dx

    return interped, ti.Vector([partial_x, partial_y, partial_z])


"""
In nfm, 
Tx = (X/x, Y/x, Z/x)
Ty = (X/y, Y/y, Z/y)
Tz = (X/z, Y/z, Z/z)

But In hermit, we need is
Tx = (X/x, X/y, X/z)
Ty = (Y/x, Y/y, Y/z)
Tz = (Z/z, Z/y, Z/z)
"""


@ti.func
def transpose_matrix(
    T_x, T_y, T_z
):
    new_T_x = ti.Vector([T_x[0], T_y[0], T_z[0]])
    new_T_y = ti.Vector([T_x[1], T_y[1], T_z[1]])
    new_T_z = ti.Vector([T_x[2], T_y[2], T_z[2]])
    return new_T_x, new_T_y, new_T_z


@ti.func
def inverse_matrix(
    T_x, T_y, T_z
):
    T = ti.Matrix.rows([T_x, T_y, T_z])
    T = ti.math.inverse(T)
    return T[0, :], T[1, :], T[2, :]


@ti.kernel
def reset_to_identity_1(
    psi: ti.template(),
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    X: ti.template()
):
    for I in ti.grouped(psi):
        psi[I] = X[I]
        T_x[I] = ti.Vector.unit(3, 0)
        T_y[I] = ti.Vector.unit(3, 1)
        T_z[I] = ti.Vector.unit(3, 2)


@ti.kernel
def reset_to_identity_2(
    psi: ti.template(),
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    dT_x: ti.template(),
    dT_y: ti.template(),
    dT_z: ti.template(),
    X: ti.template()
):
    for I in ti.grouped(psi):
        psi[I] = X[I]
        T_x[I] = ti.Vector.unit(3, 0)
        T_y[I] = ti.Vector.unit(3, 1)
        T_z[I] = ti.Vector.unit(3, 2)
        dT_x[I] = ti.Vector([0.0, 0.0, 0.0])
        dT_y[I] = ti.Vector([0.0, 0.0, 0.0])
        dT_z[I] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def reset_to_identity_3(
    psi: ti.template(),
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    dT_x: ti.template(),
    dT_y: ti.template(),
    dT_z: ti.template(),
    d2T_x: ti.template(),
    d2T_y: ti.template(),
    d2T_z: ti.template(),
    X: ti.template()
):
    for I in ti.grouped(psi):
        psi[I] = X[I]
        T_x[I] = ti.Vector.unit(3, 0)
        T_y[I] = ti.Vector.unit(3, 1)
        T_z[I] = ti.Vector.unit(3, 2)
        dT_x[I] = ti.Vector([0.0, 0.0, 0.0])
        dT_y[I] = ti.Vector([0.0, 0.0, 0.0])
        dT_z[I] = ti.Vector([0.0, 0.0, 0.0])
        d2T_x[I] = 0.0
        d2T_y[I] = 0.0
        d2T_z[I] = 0.0


@ti.kernel
def reset_to_identity_ed8(
    psi_000: ti.template(),
    psi_001: ti.template(),
    psi_010: ti.template(),
    psi_011: ti.template(),
    psi_100: ti.template(),
    psi_101: ti.template(),
    psi_110: ti.template(),
    psi_111: ti.template(),
    X: ti.template(),
    epsilon: float
):
    for I in ti.grouped(X):
        psi_000[I] = X[I] + ti.Vector([-epsilon, -epsilon, -epsilon])
        psi_001[I] = X[I] + ti.Vector([epsilon, -epsilon, -epsilon])
        psi_010[I] = X[I] + ti.Vector([-epsilon, epsilon, -epsilon])
        psi_011[I] = X[I] + ti.Vector([epsilon, epsilon, -epsilon])
        psi_100[I] = X[I] + ti.Vector([-epsilon, -epsilon, epsilon])
        psi_101[I] = X[I] + ti.Vector([epsilon, -epsilon, epsilon])
        psi_110[I] = X[I] + ti.Vector([-epsilon, epsilon, epsilon])
        psi_111[I] = X[I] + ti.Vector([epsilon, epsilon, epsilon])


@ti.kernel
def reset_to_identity_ed4(
    psi_0: ti.template(),
    psi_1: ti.template(),
    psi_2: ti.template(),
    psi_3: ti.template(),
    X: ti.template(),
    epsilon: float
):
    for I in ti.grouped(X):
        # 000 011 110 101
        psi_0[I] = X[I] + ti.Vector([-epsilon, -epsilon, -epsilon])
        psi_1[I] = X[I] + ti.Vector([epsilon, epsilon, -epsilon])
        psi_2[I] = X[I] + ti.Vector([-epsilon, epsilon, epsilon])
        psi_3[I] = X[I] + ti.Vector([epsilon, -epsilon,  epsilon])


@ti.kernel
def reset_to_identity_garm_ed4(
    psi_0: ti.template(),
    T_x_0: ti.template(),
    T_y_0: ti.template(),
    T_z_0: ti.template(),
    psi_1: ti.template(),
    T_x_1: ti.template(),
    T_y_1: ti.template(),
    T_z_1: ti.template(),
    psi_2: ti.template(),
    T_x_2: ti.template(),
    T_y_2: ti.template(),
    T_z_2: ti.template(),
    psi_3: ti.template(),
    T_x_3: ti.template(),
    T_y_3: ti.template(),
    T_z_3: ti.template(),
    X: ti.template(),
    epsilon: float
):
    for I in ti.grouped(X):
        # 000 011 110 101
        psi_0[I] = X[I] + ti.Vector([-epsilon, -epsilon, -epsilon])
        psi_1[I] = X[I] + ti.Vector([epsilon, epsilon, -epsilon])
        psi_2[I] = X[I] + ti.Vector([-epsilon, epsilon, epsilon])
        psi_3[I] = X[I] + ti.Vector([epsilon, -epsilon,  epsilon])
        T_x_0[I] = ti.Vector.unit(3, 0)
        T_y_0[I] = ti.Vector.unit(3, 1)
        T_z_0[I] = ti.Vector.unit(3, 2)
        T_x_1[I] = ti.Vector.unit(3, 0)
        T_y_1[I] = ti.Vector.unit(3, 1)
        T_z_1[I] = ti.Vector.unit(3, 2)
        T_x_2[I] = ti.Vector.unit(3, 0)
        T_y_2[I] = ti.Vector.unit(3, 1)
        T_z_2[I] = ti.Vector.unit(3, 2)
        T_x_3[I] = ti.Vector.unit(3, 0)
        T_y_3[I] = ti.Vector.unit(3, 1)
        T_z_3[I] = ti.Vector.unit(3, 2)


@ti.kernel
def advance_map_0(
    psi: ti.template(),
    u_x0: ti.template(),
    u_y0: ti.template(),
    u_z0: ti.template(),
    dx: float,
    dt: float
):
    for I in ti.grouped(psi):
        u1 = interp_u_MAC(u_x0, u_y0, u_z0, psi[I], dx)
        psi1 = psi[I] + 0.5 * dt * u1
        u2 = interp_u_MAC(u_x0, u_y0, u_z0, psi1, dx)
        psi2 = psi[I] + 0.5 * dt * u2
        u3 = interp_u_MAC(u_x0, u_y0, u_z0, psi2, dx)
        psi3 = psi[I] + dt * u3
        u4 = interp_u_MAC(u_x0, u_y0, u_z0, psi3, dx)
        psi[I] = psi[I] + dt * 1./6 * (u1 + 2.0 * u2 + 2.0 * u3 + u4)


@ti.func
def advance_map_1_transpose_func(
    psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dx: float, dt: float
):
    T_x, T_y, T_z = transpose_matrix(
        T_x, T_y, T_z,
    )

    u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi, dx)
    dT_x_dt1 = grad_u_at_psi @ T_x
    dT_y_dt1 = grad_u_at_psi @ T_y
    dT_z_dt1 = grad_u_at_psi @ T_z

    psi_x1 = psi + 0.5 * dt * u1
    T_x1 = T_x + 0.5 * dt * dT_x_dt1
    T_y1 = T_y + 0.5 * dt * dT_y_dt1
    T_z1 = T_z + 0.5 * dt * dT_z_dt1

    u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
    dT_x_dt2 = grad_u_at_psi @ T_x1
    dT_y_dt2 = grad_u_at_psi @ T_y1
    dT_z_dt2 = grad_u_at_psi @ T_z1

    psi_x2 = psi + 0.5 * dt * u2
    T_x2 = T_x + 0.5 * dt * dT_x_dt2
    T_y2 = T_y + 0.5 * dt * dT_y_dt2
    T_z2 = T_z + 0.5 * dt * dT_z_dt2

    u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
    dT_x_dt3 = grad_u_at_psi @ T_x2
    dT_y_dt3 = grad_u_at_psi @ T_y2
    dT_z_dt3 = grad_u_at_psi @ T_z2

    psi_x3 = psi + dt * u3
    T_x3 = T_x + dt * dT_x_dt3
    T_y3 = T_y + dt * dT_y_dt3
    T_z3 = T_z + dt * dT_z_dt3

    u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
    dT_x_dt4 = grad_u_at_psi @ T_x3
    dT_y_dt4 = grad_u_at_psi @ T_y3
    dT_z_dt4 = grad_u_at_psi @ T_z3

    psi = psi + dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
    T_x = T_x + dt * 1./6 * \
        (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)
    T_y = T_y + dt * 1./6 * \
        (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)
    T_z = T_z + dt * 1./6 * \
        (dT_z_dt1 + 2 * dT_z_dt2 + 2 * dT_z_dt3 + dT_z_dt4)

    T_x, T_y, T_z = transpose_matrix(
        T_x, T_y, T_z
    )


@ti.kernel
def advance_map_1_transpose(
    psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dx: float, dt: float
):
    for I in ti.grouped(psi):
        T_x[I], T_y[I], T_z[I] = transpose_matrix(
            T_x[I], T_y[I], T_z[I],
        )

        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I]
        dT_y_dt1 = grad_u_at_psi @ T_y[I]
        dT_z_dt1 = grad_u_at_psi @ T_z[I]

        psi_x1 = psi[I] + 0.5 * dt * u1
        T_x1 = T_x[I] + 0.5 * dt * dT_x_dt1
        T_y1 = T_y[I] + 0.5 * dt * dT_y_dt1
        T_z1 = T_z[I] + 0.5 * dt * dT_z_dt1

        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1
        dT_y_dt2 = grad_u_at_psi @ T_y1
        dT_z_dt2 = grad_u_at_psi @ T_z1

        psi_x2 = psi[I] + 0.5 * dt * u2
        T_x2 = T_x[I] + 0.5 * dt * dT_x_dt2
        T_y2 = T_y[I] + 0.5 * dt * dT_y_dt2
        T_z2 = T_z[I] + 0.5 * dt * dT_z_dt2

        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2
        dT_y_dt3 = grad_u_at_psi @ T_y2
        dT_z_dt3 = grad_u_at_psi @ T_z2

        psi_x3 = psi[I] + dt * u3
        T_x3 = T_x[I] + dt * dT_x_dt3
        T_y3 = T_y[I] + dt * dT_y_dt3
        T_z3 = T_z[I] + dt * dT_z_dt3

        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3
        dT_y_dt4 = grad_u_at_psi @ T_y3
        dT_z_dt4 = grad_u_at_psi @ T_z3

        psi[I] = psi[I] + dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + dt * 1./6 * \
            (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)
        T_y[I] = T_y[I] + dt * 1./6 * \
            (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)
        T_z[I] = T_z[I] + dt * 1./6 * \
            (dT_z_dt1 + 2 * dT_z_dt2 + 2 * dT_z_dt3 + dT_z_dt4)

        T_x[I], T_y[I], T_z[I] = transpose_matrix(
            T_x[I], T_y[I], T_z[I]
        )


@ti.kernel
def interp_psi_hermite(
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    dT_x: ti.template(),
    dT_y: ti.template(),
    dT_z: ti.template(),
    d2T_x: ti.template(),
    d2T_y: ti.template(),
    d2T_z: ti.template(),
    psi: ti.template(),
    psi_tmp: ti.template(),
    dx: float
):
    for I in ti.grouped(psi_tmp):
        psi_x_0 = interp_hermite(psi, 0, T_x, dT_x, d2T_x, psi_tmp[I], dx)
        psi_y_0 = interp_hermite(psi, 1, T_y, dT_y, d2T_y, psi_tmp[I], dx)
        psi_z_0 = interp_hermite(psi, 2, T_z, dT_z, d2T_z, psi_tmp[I], dx)
        psi_tmp[I] = ti.Vector([psi_x_0, psi_y_0, psi_z_0])


@ti.kernel
def interp_psi_hermite_fd_unroll(
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    psi: ti.template(),
    psi_tmp: ti.template(),
    dx: float
):
    for I in ti.grouped(psi_tmp):
        psi_x_0 = interp_hermite_fd_unroll(psi, 0, T_x, psi_tmp[I], dx)
        psi_y_0 = interp_hermite_fd_unroll(psi, 1, T_y, psi_tmp[I], dx)
        psi_z_0 = interp_hermite_fd_unroll(psi, 2, T_z, psi_tmp[I], dx)
        psi_tmp[I] = ti.Vector([psi_x_0, psi_y_0, psi_z_0])


@ti.func
def interp_psi_hermite_garm_fd_unroll_func(
    psi_tmp: ti.template(),
    T_x_tmp: ti.template(), T_y_tmp: ti.template(), T_z_tmp: ti.template(),
    psi: ti.template(),
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    dT_x: ti.template(), dT_y: ti.template(), dT_z: ti.template(),
    dx: float
):
    T_new = ti.Matrix.rows([T_x_tmp, T_y_tmp, T_z_tmp])
    psi_x_0, T_x_0 = interp_grad_hermite_garm_fd_unroll(
        psi, 0, T_x, dT_x, psi_tmp, dx)
    psi_y_0, T_y_0 = interp_grad_hermite_garm_fd_unroll(
        psi, 1, T_y, dT_y, psi_tmp, dx)
    psi_z_0, T_z_0 = interp_grad_hermite_garm_fd_unroll(
        psi, 2, T_z, dT_z, psi_tmp, dx)
    T_old = ti.Matrix.rows([T_x_0, T_y_0, T_z_0])
    T = T_old @ T_new
    T_x_tmp = T[0, :]
    T_y_tmp = T[1, :]
    T_z_tmp = T[2, :]
    psi_tmp = ti.Vector([psi_x_0, psi_y_0, psi_z_0])


@ti.kernel
def interp_psi_hermite_garm_fd_unroll(
    psi_tmp: ti.template(),
    T_x_tmp: ti.template(), T_y_tmp: ti.template(), T_z_tmp: ti.template(),
    psi: ti.template(),
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    dT_x: ti.template(), dT_y: ti.template(), dT_z: ti.template(),
    dx: float
):
    for I in ti.grouped(T_x):
        interp_psi_hermite_garm_fd_unroll_func(
            psi_tmp[I], T_x_tmp[I], T_y_tmp[I], T_z_tmp[I],
            psi[I], T_x[I], T_y[I], T_z[I], dT_x[I], dT_y[I], dT_z[I], dx)


@ti.func
def interp_psi_hermite_garm_func(
    psi_tmp: ti.template(),
    T_x_tmp: ti.template(), T_y_tmp: ti.template(), T_z_tmp: ti.template(),
    psi: ti.template(),
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    dT_x: ti.template(), dT_y: ti.template(), dT_z: ti.template(),
    d2T_x: ti.template(), d2T_y: ti.template(), d2T_z: ti.template(),
    dx: float
):
    T_new = ti.Matrix.rows([T_x_tmp, T_y_tmp, T_z_tmp])
    psi_x_0, T_x_0 = interp_grad_hermite(
        psi, 0, T_x, dT_x, d2T_x, psi_tmp, dx)
    psi_y_0, T_y_0 = interp_grad_hermite(
        psi, 1, T_y, dT_y, d2T_y, psi_tmp, dx)
    psi_z_0, T_z_0 = interp_grad_hermite(
        psi, 2, T_z, dT_z, d2T_z, psi_tmp, dx)
    T_old = ti.Matrix.rows([T_x_0, T_y_0, T_z_0])
    T = T_old @ T_new
    T_x_tmp = T[0, :]
    T_y_tmp = T[1, :]
    T_z_tmp = T[2, :]
    psi_tmp = ti.Vector([psi_x_0, psi_y_0, psi_z_0])


@ti.kernel
def interp_psi_hermite_garm(
    psi_tmp: ti.template(),
    T_x_tmp: ti.template(), T_y_tmp: ti.template(), T_z_tmp: ti.template(),
    psi: ti.template(),
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    dT_x: ti.template(), dT_y: ti.template(), dT_z: ti.template(),
    d2T_x: ti.template(), d2T_y: ti.template(), d2T_z: ti.template(),
    dx: float
):
    for I in ti.grouped(T_x):
        T_new = ti.Matrix.rows([T_x_tmp[I], T_y_tmp[I], T_z_tmp[I]])
        psi_x_0, T_x_0 = interp_grad_hermite(
            psi, 0, T_x, dT_x, d2T_x, psi_tmp[I], dx)
        psi_y_0, T_y_0 = interp_grad_hermite(
            psi, 1, T_y, dT_y, d2T_y, psi_tmp[I], dx)
        psi_z_0, T_z_0 = interp_grad_hermite(
            psi, 2, T_z, dT_z, d2T_z, psi_tmp[I], dx)
        T_old = ti.Matrix.rows([T_x_0, T_y_0, T_z_0])
        T = T_old @ T_new
        T_x_tmp[I] = T[0, :]
        T_y_tmp[I] = T[1, :]
        T_z_tmp[I] = T[2, :]
        psi_tmp[I] = ti.Vector([psi_x_0, psi_y_0, psi_z_0])


@ti.func
def calc_ed4_garm_func(
    T_x_0: ti.template(), T_y_0: ti.template(), T_z_0: ti.template(),
    T_x_1: ti.template(), T_y_1: ti.template(), T_z_1: ti.template(),
    T_x_2: ti.template(), T_y_2: ti.template(), T_z_2: ti.template(),
    T_x_3: ti.template(), T_y_3: ti.template(), T_z_3: ti.template(),
    dT_x: ti.template(), dT_y: ti.template(), dT_z: ti.template(),
    epsilon: float
):
    dT_x[0] = ((-T_x_0[1]-T_x_1[1]+T_x_2[1]+T_x_3[1])
               + (-T_x_0[2]+T_x_1[2]+T_x_2[2]-T_x_3[2]))/8/epsilon
    dT_x[1] = ((-T_x_0[0]-T_x_1[0]+T_x_2[0]+T_x_3[0])
               + (-T_x_0[2]+T_x_1[2]-T_x_2[2]+T_x_3[2]))/8/epsilon
    dT_x[2] = ((-T_x_0[0]+T_x_1[0]+T_x_2[0]-T_x_3[0])
               + (-T_x_0[1]+T_x_1[1]-T_x_2[1]+T_x_3[1]))/8/epsilon
    dT_y[0] = ((-T_y_0[1]-T_y_1[1]+T_y_2[1]+T_y_3[1])
               + (-T_y_0[2]+T_y_1[2]+T_y_2[2]-T_y_3[2]))/8/epsilon
    dT_y[1] = ((-T_y_0[0]-T_y_1[0]+T_y_2[0]+T_y_3[0])
               + (-T_y_0[2]+T_y_1[2]-T_y_2[2]+T_y_3[2]))/8/epsilon
    dT_y[2] = ((-T_y_0[0]+T_y_1[0]+T_y_2[0]-T_y_3[0])
               + (-T_y_0[1]+T_y_1[1]-T_y_2[1]+T_y_3[1]))/8/epsilon
    dT_z[0] = ((-T_z_0[1]-T_z_1[1]+T_z_2[1]+T_z_3[1])
               + (-T_z_0[2]+T_z_1[2]+T_z_2[2]-T_z_3[2]))/8/epsilon
    dT_z[1] = ((-T_z_0[0]-T_z_1[0]+T_z_2[0]+T_z_3[0])
               + (-T_z_0[2]+T_z_1[2]-T_z_2[2]+T_z_3[2]))/8/epsilon
    dT_z[2] = ((-T_z_0[0]+T_z_1[0]+T_z_2[0]-T_z_3[0])
               + (-T_z_0[1]+T_z_1[1]-T_z_2[1]+T_z_3[1]))/8/epsilon


@ti.kernel
def calc_ed4_garm(
    T_x_0: ti.template(), T_y_0: ti.template(), T_z_0: ti.template(),
    T_x_1: ti.template(), T_y_1: ti.template(), T_z_1: ti.template(),
    T_x_2: ti.template(), T_y_2: ti.template(), T_z_2: ti.template(),
    T_x_3: ti.template(), T_y_3: ti.template(), T_z_3: ti.template(),
    dT_x: ti.template(), dT_y: ti.template(), dT_z: ti.template(),
    epsilon: float
):
    for I in ti.grouped(T_x_0):
        calc_ed4_garm_func(
            T_x_0[I], T_y_0[I], T_z_0[I],
            T_x_1[I], T_y_1[I], T_z_1[I],
            T_x_2[I], T_y_2[I], T_z_2[I],
            T_x_3[I], T_y_3[I], T_z_3[I],
            dT_x[I], dT_y[I], dT_z[I],
            epsilon
        )


@ti.kernel
def calc_ed8_3(
    psi_000: ti.template(),
    psi_001: ti.template(),
    psi_010: ti.template(),
    psi_011: ti.template(),
    psi_100: ti.template(),
    psi_101: ti.template(),
    psi_110: ti.template(),
    psi_111: ti.template(),
    psi: ti.template(),
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    dT_x: ti.template(),
    dT_y: ti.template(),
    dT_z: ti.template(),
    d2T_x: ti.template(),
    d2T_y: ti.template(),
    d2T_z: ti.template(),
    epsilon: float
):
    for I in ti.grouped(psi):
        psi[I] = (psi_000[I]+psi_001[I]+psi_010[I]+psi_011[I] +
                  psi_100[I]+psi_101[I]+psi_110[I]+psi_111[I])/8
        psi_partial_x = (-psi_000[I]+psi_001[I]-psi_010[I]+psi_011[I] -
                         psi_100[I]+psi_101[I]-psi_110[I]+psi_111[I])/8/epsilon
        psi_partial_y = (-psi_000[I]-psi_001[I]+psi_010[I]+psi_011[I] -
                         psi_100[I]-psi_101[I]+psi_110[I]+psi_111[I])/8/epsilon
        psi_partial_z = (-psi_000[I]-psi_001[I]-psi_010[I]-psi_011[I] +
                         psi_100[I]+psi_101[I]+psi_110[I]+psi_111[I])/8/epsilon
        T_x[I] = ti.Vector(
            [psi_partial_x[0], psi_partial_y[0], psi_partial_z[0]])
        T_y[I] = ti.Vector(
            [psi_partial_x[1], psi_partial_y[1], psi_partial_z[1]])
        T_z[I] = ti.Vector(
            [psi_partial_x[2], psi_partial_y[2], psi_partial_z[2]])
        psi_partial_yz = (psi_000[I]+psi_001[I]-psi_010[I]-psi_011[I] -
                          psi_100[I]-psi_101[I]+psi_110[I]+psi_111[I])/8/epsilon/epsilon
        psi_partial_zx = (psi_000[I]-psi_001[I]+psi_010[I]-psi_011[I] -
                          psi_100[I]+psi_101[I]-psi_110[I]+psi_111[I])/8/epsilon/epsilon
        psi_partial_xy = (psi_000[I]-psi_001[I]-psi_010[I]+psi_011[I] +
                          psi_100[I]-psi_101[I]-psi_110[I]+psi_111[I])/8/epsilon/epsilon
        dT_x[I] = ti.Vector(
            [psi_partial_yz[0], psi_partial_zx[0], psi_partial_xy[0]])
        dT_y[I] = ti.Vector(
            [psi_partial_yz[1], psi_partial_zx[1], psi_partial_xy[1]])
        dT_z[I] = ti.Vector(
            [psi_partial_yz[2], psi_partial_zx[2], psi_partial_xy[2]])
        psi_partial_xyz = (-psi_000[I]+psi_001[I]+psi_010[I]-psi_011[I] +
                           psi_100[I]-psi_101[I]-psi_110[I]+psi_111[I])/8/epsilon/epsilon/epsilon
        d2T_x[I] = psi_partial_xyz[0]
        d2T_y[I] = psi_partial_xyz[1]
        d2T_z[I] = psi_partial_xyz[2]


@ti.kernel
def calc_ed4_1(
    psi_0: ti.template(),
    psi_1: ti.template(),
    psi_2: ti.template(),
    psi_3: ti.template(),
    psi: ti.template(),
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    epsilon: float
):
    # 000 011 110 101
    for I in ti.grouped(psi):
        psi[I] = (psi_0[I]+psi_1[I]+psi_2[I]+psi_3[I])/4
        psi_partial_x = (-psi_0[I]+psi_1[I]-psi_2[I]+psi_3[I])/4/epsilon
        psi_partial_y = (-psi_0[I]+psi_1[I]+psi_2[I]-psi_3[I])/4/epsilon
        psi_partial_z = (-psi_0[I]-psi_1[I]+psi_2[I]+psi_3[I])/4/epsilon
        T_x[I] = ti.Vector(
            [psi_partial_x[0], psi_partial_y[0], psi_partial_z[0]])
        T_y[I] = ti.Vector(
            [psi_partial_x[1], psi_partial_y[1], psi_partial_z[1]])
        T_z[I] = ti.Vector(
            [psi_partial_x[2], psi_partial_y[2], psi_partial_z[2]])


@ti.kernel
def calc_fd_3_from_grad(
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
    dT_x: ti.template(),
    dT_y: ti.template(),
    dT_z: ti.template(),
    d2T_x: ti.template(),
    d2T_y: ti.template(),
    d2T_z: ti.template(),
    dx: float
):

    for i, j, k in dT_x:
        dT_x[i, j, k][0] = (
            (sample(T_x, i, j, k+1)[1]-sample(T_x, i, j, k-1)[1]) +
            (sample(T_x, i, j+1, k)[2]-sample(T_x, i, j-1, k)[2])
        )/4/dx
        dT_x[i, j, k][1] = (
            (sample(T_x, i, j, k+1)[0]-sample(T_x, i, j, k-1)[0]) +
            (sample(T_x, i+1, j, k)[2]-sample(T_x, i-1, j, k)[2])
        )/4/dx
        dT_x[i, j, k][2] = (
            (sample(T_x, i, j+1, k)[0]-sample(T_x, i, j-1, k)[0]) +
            (sample(T_x, i+1, j, k)[1]-sample(T_x, i-1, j, k)[1])
        )/4/dx

    for i, j, k in dT_y:
        dT_y[i, j, k][0] = (
            (sample(T_y, i, j, k+1)[1]-sample(T_y, i, j, k-1)[1]) +
            (sample(T_y, i, j+1, k)[2]-sample(T_y, i, j-1, k)[2])
        )/4/dx
        dT_y[i, j, k][1] = (
            (sample(T_y, i, j, k+1)[0]-sample(T_y, i, j, k-1)[0]) +
            (sample(T_y, i+1, j, k)[2]-sample(T_y, i-1, j, k)[2])
        )/4/dx
        dT_y[i, j, k][2] = (
            (sample(T_y, i, j+1, k)[0]-sample(T_y, i, j-1, k)[0]) +
            (sample(T_y, i+1, j, k)[1]-sample(T_y, i-1, j, k)[1])
        )/4/dx

    for i, j, k in dT_z:
        dT_z[i, j, k][0] = (
            (sample(T_z, i, j, k+1)[1]-sample(T_z, i, j, k-1)[1]) +
            (sample(T_z, i, j+1, k)[2]-sample(T_z, i, j-1, k)[2])
        )/4/dx
        dT_z[i, j, k][1] = (
            (sample(T_z, i, j, k+1)[0]-sample(T_z, i, j, k-1)[0]) +
            (sample(T_z, i+1, j, k)[2]-sample(T_z, i-1, j, k)[2])
        )/4/dx
        dT_z[i, j, k][2] = (
            (sample(T_z, i, j+1, k)[0]-sample(T_z, i, j-1, k)[0]) +
            (sample(T_z, i+1, j, k)[1]-sample(T_z, i-1, j, k)[1])
        )/4/dx

    for i, j, k in d2T_x:
        d2T_x[i, j, k] = (
            (sample(dT_x, i+1, j, k)[0]-sample(dT_x, i-1, j, k)[0]) +
            (sample(dT_x, i, j+1, k)[1]-sample(dT_x, i, j-1, k)[1]) +
            (sample(dT_x, i, j, k+1)[2]-sample(dT_x, i, j, k-1)[2])
        )/6/dx

    for i, j, k in d2T_y:
        d2T_y[i, j, k] = (
            (sample(dT_y, i+1, j, k)[0]-sample(dT_y, i-1, j, k)[0]) +
            (sample(dT_y, i, j+1, k)[1]-sample(dT_y, i, j-1, k)[1]) +
            (sample(dT_y, i, j, k+1)[2]-sample(dT_y, i, j, k-1)[2])
        )/6/dx

    for i, j, k in d2T_z:
        d2T_z[i, j, k] = (
            (sample(dT_z, i+1, j, k)[0]-sample(dT_z, i-1, j, k)[0]) +
            (sample(dT_z, i, j+1, k)[1]-sample(dT_z, i, j-1, k)[1]) +
            (sample(dT_z, i, j, k+1)[2]-sample(dT_z, i, j, k-1)[2])
        )/6/dx


@ti.kernel
def calc_fd_3_from_grad_grad(
    dT_x: ti.template(),
    dT_y: ti.template(),
    dT_z: ti.template(),
    d2T_x: ti.template(),
    d2T_y: ti.template(),
    d2T_z: ti.template(),
    dx: float
):
    for i, j, k in d2T_x:
        d2T_x[i, j, k] = (
            (sample(dT_x, i+1, j, k)[0]-sample(dT_x, i-1, j, k)[0]) +
            (sample(dT_x, i, j+1, k)[1]-sample(dT_x, i, j-1, k)[1]) +
            (sample(dT_x, i, j, k+1)[2]-sample(dT_x, i, j, k-1)[2])
        )/6/dx

    for i, j, k in d2T_y:
        d2T_y[i, j, k] = (
            (sample(dT_y, i+1, j, k)[0]-sample(dT_y, i-1, j, k)[0]) +
            (sample(dT_y, i, j+1, k)[1]-sample(dT_y, i, j-1, k)[1]) +
            (sample(dT_y, i, j, k+1)[2]-sample(dT_y, i, j, k-1)[2])
        )/6/dx

    for i, j, k in d2T_z:
        d2T_z[i, j, k] = (
            (sample(dT_z, i+1, j, k)[0]-sample(dT_z, i-1, j, k)[0]) +
            (sample(dT_z, i, j+1, k)[1]-sample(dT_z, i, j-1, k)[1]) +
            (sample(dT_z, i, j, k+1)[2]-sample(dT_z, i, j, k-1)[2])
        )/6/dx


def advect_ps_hermite_ed8(
    psi_000, psi_001, psi_010, psi_011,
    psi_100, psi_101, psi_110, psi_111,
    psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z,
    u_x, u_y, u_z, X, dx, dt, epsilon
):
    reset_to_identity_ed8(
        psi_000, psi_001, psi_010, psi_011,
        psi_100, psi_101, psi_110, psi_111,
        X, epsilon
    )
    advance_map_0(psi_000, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_001, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_010, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_011, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_100, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_101, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_110, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_111, u_x, u_y, u_z, dx, -dt)
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_000, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_001, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_010, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_011, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_100, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_101, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_110, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_111, dx
    )
    calc_ed8_3(
        psi_000, psi_001, psi_010, psi_011,
        psi_100, psi_101, psi_110, psi_111,
        psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z,
        epsilon
    )


def advect_ps_hermite_ed4(
    psi_0, psi_1, psi_2, psi_3,
    psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z,
    u_x, u_y, u_z, X, dx, dt, epsilon
):
    reset_to_identity_ed4(
        psi_0, psi_1, psi_2, psi_3,
        X, epsilon
    )
    advance_map_0(psi_0, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_1, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_2, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_3, u_x, u_y, u_z, dx, -dt)
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_0, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_1, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_2, dx
    )
    interp_psi_hermite(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, psi, psi_3, dx
    )
    calc_ed4_1(
        psi_0, psi_1, psi_2, psi_3,
        psi, T_x, T_y, T_z, epsilon
    )
    calc_fd_3_from_grad(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, dx
    )


def advect_ps_hermite_ed4_fd_unroll(
    psi_0, psi_1, psi_2, psi_3,
    psi, T_x, T_y, T_z,
    u_x, u_y, u_z, X, dx, dt, epsilon
):
    reset_to_identity_ed4(
        psi_0, psi_1, psi_2, psi_3,
        X, epsilon
    )
    advance_map_0(psi_0, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_1, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_2, u_x, u_y, u_z, dx, -dt)
    advance_map_0(psi_3, u_x, u_y, u_z, dx, -dt)
    interp_psi_hermite_fd_unroll(T_x, T_y, T_z, psi, psi_0, dx)
    interp_psi_hermite_fd_unroll(T_x, T_y, T_z, psi, psi_1, dx)
    interp_psi_hermite_fd_unroll(T_x, T_y, T_z, psi, psi_2, dx)
    interp_psi_hermite_fd_unroll(T_x, T_y, T_z, psi, psi_3, dx)
    calc_ed4_1(
        psi_0, psi_1, psi_2, psi_3,
        psi, T_x, T_y, T_z, epsilon
    )


def advect_ps_garm(
    psi_tmp,
    T_x_tmp, T_y_tmp, T_z_tmp,
    psi,
    T_x, T_y, T_z,
    dT_x, dT_y, dT_z,
    d2T_x, d2T_y, d2T_z,
    u_x, u_y, u_z,
    X, dx, dt
):
    reset_to_identity_1(psi_tmp, T_x_tmp, T_y_tmp, T_z_tmp, X)
    advance_map_1_transpose(psi_tmp, T_x_tmp, T_y_tmp,
                            T_z_tmp, u_x, u_y, u_z, dx, -dt)
    interp_psi_hermite_garm(psi_tmp,
                            T_x_tmp, T_y_tmp, T_z_tmp,
                            psi,
                            T_x, T_y, T_z,
                            dT_x, dT_y, dT_z,
                            d2T_x, d2T_y, d2T_z,
                            dx)
    psi.copy_from(psi_tmp)
    T_x.copy_from(T_x_tmp)
    T_y.copy_from(T_y_tmp)
    T_z.copy_from(T_z_tmp)
    calc_fd_3_from_grad(
        T_x, T_y, T_z, dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, dx
    )

@ti.kernel
def advect_psi_garm_ed4_fusion_unroll_kernel(
    psi_tmp: ti.template(),
    T_x_tmp: ti.template(), T_y_tmp: ti.template(), T_z_tmp: ti.template(),
    dT_x_tmp: ti.template(), dT_y_tmp: ti.template(), dT_z_tmp: ti.template(),
    psi: ti.template(),
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    dT_x: ti.template(), dT_y: ti.template(), dT_z: ti.template(),
    u_x: ti.template(), u_y: ti.template(), u_z: ti.template(),
    X: ti.template(), dx: float, dt: float, epsilon: float
):
    for I in ti.grouped(psi):
        psi_tmp[I] = X[I]
        T_x_tmp[I] = ti.Vector([1.0, 0.0, 0.0])
        T_y_tmp[I] = ti.Vector([0.0, 1.0, 0.0])
        T_z_tmp[I] = ti.Vector([0.0, 0.0, 1.0])
        dT_x_tmp[I] = ti.Vector([0.0, 0.0, 0.0])
        dT_y_tmp[I] = ti.Vector([0.0, 0.0, 0.0])
        dT_z_tmp[I] = ti.Vector([0.0, 0.0, 0.0])

        psi_0 = X[I] + ti.Vector([-epsilon, -epsilon, -epsilon])
        psi_1 = X[I] + ti.Vector([epsilon, epsilon, -epsilon])
        psi_2 = X[I] + ti.Vector([-epsilon, epsilon, epsilon])
        psi_3 = X[I] + ti.Vector([epsilon, -epsilon,  epsilon])
        T_x_0 = ti.Vector([1.0, 0.0, 0.0])
        T_y_0 = ti.Vector([0.0, 1.0, 0.0])
        T_z_0 = ti.Vector([0.0, 0.0, 1.0])
        T_x_1 = ti.Vector([1.0, 0.0, 0.0])
        T_y_1 = ti.Vector([0.0, 1.0, 0.0])
        T_z_1 = ti.Vector([0.0, 0.0, 1.0])
        T_x_2 = ti.Vector([1.0, 0.0, 0.0])
        T_y_2 = ti.Vector([0.0, 1.0, 0.0])
        T_z_2 = ti.Vector([0.0, 0.0, 1.0])
        T_x_3 = ti.Vector([1.0, 0.0, 0.0])
        T_y_3 = ti.Vector([0.0, 1.0, 0.0])
        T_z_3 = ti.Vector([0.0, 0.0, 1.0])

        advance_map_1_transpose_func(
            psi_tmp[I], T_x_tmp[I], T_y_tmp[I], T_z_tmp[I], u_x, u_y, u_z, dx, -dt)
        advance_map_1_transpose_func(
            psi_0, T_x_0, T_y_0, T_z_0, u_x, u_y, u_z, dx, -dt)
        advance_map_1_transpose_func(
            psi_1, T_x_1, T_y_1, T_z_1, u_x, u_y, u_z, dx, -dt)
        advance_map_1_transpose_func(
            psi_2, T_x_2, T_y_2, T_z_2, u_x, u_y, u_z, dx, -dt)
        advance_map_1_transpose_func(
            psi_3, T_x_3, T_y_3, T_z_3, u_x, u_y, u_z, dx, -dt)

        interp_psi_hermite_garm_fd_unroll_func(
            psi_tmp[I], T_x_tmp[I], T_y_tmp[I], T_z_tmp[I],
            psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, dx)
        interp_psi_hermite_garm_fd_unroll_func(
            psi_0, T_x_0, T_y_0, T_z_0,
            psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, dx)
        interp_psi_hermite_garm_fd_unroll_func(
            psi_1, T_x_1, T_y_1, T_z_1,
            psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, dx)
        interp_psi_hermite_garm_fd_unroll_func(
            psi_2, T_x_2, T_y_2, T_z_2,
            psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, dx)
        interp_psi_hermite_garm_fd_unroll_func(
            psi_3, T_x_3, T_y_3, T_z_3,
            psi, T_x, T_y, T_z, dT_x, dT_y, dT_z, dx)

        calc_ed4_garm_func(T_x_0, T_y_0, T_z_0, T_x_1, T_y_1, T_z_1,
                           T_x_2, T_y_2, T_z_2, T_x_3, T_y_3, T_z_3,
                           dT_x_tmp[I], dT_y_tmp[I], dT_z_tmp[I], epsilon)


def advect_psi_garm_ed4_fusion_unroll(
    psi_tmp,
    T_x_tmp, T_y_tmp, T_z_tmp,
    dT_x_tmp, dT_y_tmp, dT_z_tmp,
    psi,
    T_x, T_y, T_z,
    dT_x, dT_y, dT_z,
    u_x, u_y, u_z,
    X, dx, dt, epsilon
):
    advect_psi_garm_ed4_fusion_unroll_kernel(
        psi_tmp,
        T_x_tmp, T_y_tmp, T_z_tmp,
        dT_x_tmp, dT_y_tmp, dT_z_tmp,
        psi,
        T_x, T_y, T_z,
        dT_x, dT_y, dT_z,
        u_x, u_y, u_z,
        X, dx, dt, epsilon
    )
    psi.copy_from(psi_tmp)
    T_x.copy_from(T_x_tmp)
    T_y.copy_from(T_y_tmp)
    T_z.copy_from(T_z_tmp)
    dT_x.copy_from(dT_x_tmp)
    dT_y.copy_from(dT_y_tmp)
    dT_z.copy_from(dT_z_tmp)

def advect_psi_garm_ed4(
    psi_0, T_x_0, T_y_0, T_z_0,
    psi_1, T_x_1, T_y_1, T_z_1,
    psi_2, T_x_2, T_y_2, T_z_2,
    psi_3, T_x_3, T_y_3, T_z_3,
    psi_tmp,
    T_x_tmp, T_y_tmp, T_z_tmp,
    psi,
    T_x, T_y, T_z,
    dT_x, dT_y, dT_z,
    d2T_x, d2T_y, d2T_z,
    u_x, u_y, u_z,
    X, dx, dt, epsilon
):
    reset_to_identity_1(psi_tmp, T_x_tmp, T_y_tmp, T_z_tmp, X)
    reset_to_identity_garm_ed4(psi_0, T_x_0, T_y_0, T_z_0,
                               psi_1, T_x_1, T_y_1, T_z_1,
                               psi_2, T_x_2, T_y_2, T_z_2,
                               psi_3, T_x_3, T_y_3, T_z_3,
                               X, epsilon)
    advance_map_1_transpose(psi_tmp, T_x_tmp, T_y_tmp,
                            T_z_tmp, u_x, u_y, u_z, dx, -dt)
    advance_map_1_transpose(psi_0, T_x_0, T_y_0, T_z_0, u_x, u_y, u_z, dx, -dt)
    advance_map_1_transpose(psi_1, T_x_1, T_y_1, T_z_1, u_x, u_y, u_z, dx, -dt)
    advance_map_1_transpose(psi_2, T_x_2, T_y_2, T_z_2, u_x, u_y, u_z, dx, -dt)
    advance_map_1_transpose(psi_3, T_x_3, T_y_3, T_z_3, u_x, u_y, u_z, dx, -dt)
    interp_psi_hermite_garm(psi_tmp, T_x_tmp, T_y_tmp, T_z_tmp,
                            psi, T_x, T_y, T_z,
                            dT_x, dT_y, dT_z,
                            d2T_x, d2T_y, d2T_z, dx)
    interp_psi_hermite_garm(psi_0, T_x_0, T_y_0, T_z_0,
                            psi, T_x, T_y, T_z,
                            dT_x, dT_y, dT_z,
                            d2T_x, d2T_y, d2T_z, dx)
    interp_psi_hermite_garm(psi_1, T_x_1, T_y_1, T_z_1,
                            psi, T_x, T_y, T_z,
                            dT_x, dT_y, dT_z,
                            d2T_x, d2T_y, d2T_z, dx)
    interp_psi_hermite_garm(psi_2, T_x_2, T_y_2, T_z_2,
                            psi, T_x, T_y, T_z,
                            dT_x, dT_y, dT_z,
                            d2T_x, d2T_y, d2T_z, dx)
    interp_psi_hermite_garm(psi_3, T_x_3, T_y_3, T_z_3,
                            psi, T_x, T_y, T_z,
                            dT_x, dT_y, dT_z,
                            d2T_x, d2T_y, d2T_z, dx)
    psi.copy_from(psi_tmp)
    T_x.copy_from(T_x_tmp)
    T_y.copy_from(T_y_tmp)
    T_z.copy_from(T_z_tmp)
    calc_ed4_garm(T_x_0, T_y_0, T_z_0, T_x_1, T_y_1, T_z_1,
                  T_x_2, T_y_2, T_z_2, T_x_3, T_y_3, T_z_3,
                  dT_x, dT_y, dT_z, epsilon)
    calc_fd_3_from_grad_grad(dT_x, dT_y, dT_z, d2T_x, d2T_y, d2T_z, dx)

@ti.kernel
def advect_u_semi_lagrangian(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(), dx: float, dt: float,
    X_x: ti.template(), X_y: ti.template(), X_z: ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_x[I]
        v1 = interp_u_MAC(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2 = interp_u_MAC(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3 = interp_u_MAC(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4 = interp_u_MAC(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5 = interp_u_MAC(u_x0, u_y0, u_z0, p, dx)
        u_x1[I] = v5[0]

    for I in ti.grouped(u_y1):
        p1 = X_y[I]
        v1 = interp_u_MAC(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2 = interp_u_MAC(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3 = interp_u_MAC(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4 = interp_u_MAC(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5 = interp_u_MAC(u_x0, u_y0, u_z0, p, dx)
        u_y1[I] = v5[1]

    for I in ti.grouped(u_z1):
        p1 = X_z[I]
        v1 = interp_u_MAC(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2 = interp_u_MAC(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3 = interp_u_MAC(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4 = interp_u_MAC(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5 = interp_u_MAC(u_x0, u_y0, u_z0, p, dx)
        u_z1[I] = v5[2]
