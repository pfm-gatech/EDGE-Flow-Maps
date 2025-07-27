#
from util.taichi_utils import *
import math
import numpy as np
import trimesh
import os

# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: ring center position
# unit_x, unit_y: the plane of the circle
@ti.kernel
def add_vortex_ring_and_smoke(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                              unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                              pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), num_samples: int):
    # each sample point has an associated length
    curve_length = (2 * math.pi * rad) / num_samples
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            # position of the sample point
            p_sampled = rad * (ti.cos(theta) * unit_x +
                               ti.sin(theta) * unit_y) + c
            p_diff = pf[i, j, k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i, j, k] += curve_length * (-1/(4 * math.pi * r ** 3) * (
                1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i, j, k][3] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i, j, k][3] > 0.002:
            smokef[i, j, k][3] = 1.0
            smokef[i, j, k].xyz = color
        else:
            smokef[i, j, k] = ti.Vector([0., 0., 0., 0.])

def init_four_vorts(X, u, smoke1, smoke2):
    smoke1.fill(0.)
    smoke2.fill(0.)
    x_offset = 0.16
    y_offset = 0.16
    size = 0.15
    cos45 = ti.cos(math.pi/4)
    add_vortex_ring_and_smoke(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5-x_offset, 0.5-y_offset, 1]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([0., 0., 1.]),
                              pf=X, vf=u, smokef=smoke1, color=ti.Vector([1, 0.0, 0.0]), num_samples=500)

    add_vortex_ring_and_smoke(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5+x_offset, 0.5-y_offset, 1]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([0., 0., 1.]),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0.0, 1.0, 0.0]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5-x_offset, 0.5+y_offset, 1]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([0., 0., 1.]),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0.0, 0.0, 1.0]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5+x_offset, 0.5+y_offset, 1]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([0., 0., 1.]),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0.0, 0.0, 0.0]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)


def init_vorts_leapfrog(X, u, smoke1, smoke2):
    radius = 0.21
    x_gap = 0.625 * radius
    x_start = 0.16
    delta = 0.08 * radius
    w = radius * 0.1
    add_vortex_ring_and_smoke(w=w, rad=radius, delta=delta, c=ti.Vector([x_start, 0.5, 0.5]),
                              unit_x=ti.Vector([0., 0., -1.]), unit_y=ti.Vector([0., 1, 0.]),
                              pf=X, vf=u, smokef=smoke1, color=ti.Vector([1., 0, 0]), num_samples=2000)

    add_vortex_ring_and_smoke(w=w, rad=radius, delta=delta, c=ti.Vector([x_start+x_gap, 0.5, 0.5]),
                              unit_x=ti.Vector([0., 0., -1.]), unit_y=ti.Vector([0., 1, 0.]),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0, 0, 1.]), num_samples=2000)

    add_fields(smoke1, smoke2, smoke1, 1.0)

    # leapfrog_smoke_func(smoke1, X)


def init_vorts_headon(X, u, smoke1, smoke2):
    add_vortex_ring_and_smoke(w=2.e-2, rad=0.065, delta=0.016, c=ti.Vector([0.1, 0.5, 0.5]),
                              unit_x=ti.Vector([0., 0., -1.]), unit_y=ti.Vector([0., 1, 0.]),
                              pf=X, vf=u, smokef=smoke1, color=ti.Vector([1., 0, 0]), num_samples=500)

    add_vortex_ring_and_smoke(w=-2.e-2, rad=0.065, delta=0.016, c=ti.Vector([0.4, 0.5, 0.5]),
                              unit_x=ti.Vector([0., 0., -1.]), unit_y=ti.Vector([0., 1, 0.]),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0, 0, 1.]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)

def init_trefoil(u_x, u_y, u_z, smoke):
    u_x_numpy = np.load(os.path.join("assets", "trefoil_init_vel_x.npy")).astype(np.float32)
    u_y_numpy = np.load(os.path.join("assets", "trefoil_init_vel_y.npy")).astype(np.float32)
    u_z_numpy = np.load(os.path.join("assets", "trefoil_init_vel_z.npy")).astype(np.float32)
    smoke_numpy = np.load(os.path.join("assets", "trefoil_init_smoke.npy")).astype(np.float32)
    smoke_numpy[smoke_numpy > 0.3] = 1
    smoke_numpy[smoke_numpy < 1] = 0
    u_x.from_numpy(u_x_numpy.astype(np.float32))
    u_y.from_numpy(u_y_numpy.astype(np.float32))
    u_z.from_numpy(u_z_numpy.astype(np.float32))
    smoke.from_numpy(smoke_numpy.astype(np.float32))

@ti.kernel
def leapfrog_smoke_func(qf: ti.template(), pf: ti.template()):
    for I in ti.grouped(qf):
        if 0.2 <= pf[I].x <= 0.3 and 0.1 <= pf[I].y <= 0.9 and 0.1 <= pf[I].z <= 0.9:
            center = ti.Vector([0.2, 0.5, 0.5])
            radius = 0.3
            dist = (pf[I]-center).norm()
            if dist < radius:
                qf[I] = ti.Vector([1.0, 0.0, 0.0, 1.0])
            else:
                qf[I] = ti.Vector([0.0, 1.0, 0.0, 2.0])
        else:
            qf[I] = ti.Vector([0.0, 0.0, 0.0, 0.0])


@ti.kernel
def set_boundary_mask_delta_kernel(X: ti.template(), boundary_mask: ti.template(),
                                   boundary_vel: ti.template(), _t: float, dx: float):
    v = 0.6
    angle = ti.math.radians(20)
    for i, j, k in boundary_mask:
        if i == 0 or i == boundary_mask.shape[0]-1 or j == 0 or j == boundary_mask.shape[1]-1 or k == 0 or k == boundary_mask.shape[2]-1:
            boundary_mask[i, j, k] = 1
            boundary_vel[i, j, k] = ti.Vector(
                [v * ti.cos(angle), 0.0, v * ti.sin(angle)])


def set_boundary_mask_delta(X, boundary_mask, boundary_vel, _t, dx):
    angle = np.radians(15)
    attangle = np.radians(20)
    length = 1.0
    x = np.cos(angle) * length
    y = np.sin(angle) * length
    vert = np.array([[0, 0, 0], [x, y, 0], [x, -y, 0]])
    face = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vert, faces=face)
    # mesh.apply_transform(trimesh.transformations.rotation_matrix(attangle, [0, 1, 0]))
    vol = mesh.voxelized(dx)

    mask = np.zeros(boundary_mask.shape, dtype=np.int32)
    obj_mask = vol.matrix
    size = obj_mask.shape
    offset = (0.1//dx, 0.5//dx - size[1]//2, 0.2//dx - size[2]//2)
    offset = np.array(offset, dtype=np.int32)
    mask[offset[0]:offset[0]+size[0], offset[1]:offset[1] +
         size[1], offset[2]:offset[2]+1] += obj_mask
    mask[offset[0]:offset[0]+size[0], offset[1]:offset[1] +
         size[1], offset[2]+1:offset[2]+2] += obj_mask
    mask[offset[0]:offset[0]+size[0], offset[1]:offset[1] +
         size[1], offset[2]+2:offset[2]+3] += obj_mask
    mask[offset[0]:offset[0]+size[0], offset[1]:offset[1] +
         size[1], offset[2]+3:offset[2]+4] += obj_mask
    mask[mask > 0] = 1
    boundary_mask.from_numpy(mask)
    boundary_vel.fill(0.0)
    set_boundary_mask_delta_kernel(X, boundary_mask, boundary_vel, _t, dx)


@ti.kernel
def init_delta(X: ti.template(), u: ti.template(), smoke1: ti.template(), smoke2: ti.template()):
    v = 0.6
    angle = ti.math.radians(20)
    for I in ti.grouped(u):
        u[I] = ti.Vector([v * ti.cos(angle), 0.0, v * ti.sin(angle)])

@ti.kernel
def emit_smoke_delta(X: ti.template(), smoke: ti.template()):
    size = X.shape
    offset = ti.Vector([0.1, 0.5, 0.25])
    angle = ti.math.radians(15)
    radius = 0.015
    for I in ti.grouped(smoke):
        p = X[I]

        x = ti.math.cos(angle) * 0.25
        y = ti.math.sin(angle) * 0.25
        center = ti.Vector([x, y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([1.0, 0.0, 0.0, 0.0])
        
        center = ti.Vector([x, -y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([1.0, 0.0, 0.0, 0.0])
        
        x = ti.math.cos(angle) * 0.45
        y = ti.math.sin(angle) * 0.45
        center = ti.Vector([x, y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([0.0, 1.0, 0.0, 0.0])

        center = ti.Vector([x, -y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([0.0, 1.0, 0.0, 0.0])
        
        x = ti.math.cos(angle) * 0.65
        y = ti.math.sin(angle) * 0.65
        center = ti.Vector([x, y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([0.0, 0.0, 1.0, 0.0])
        
        center = ti.Vector([x, -y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([0.0, 0.0, 1.0, 0.0])

        x = ti.math.cos(angle) * 0.85
        y = ti.math.sin(angle) * 0.85
        center = ti.Vector([x, y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([0.0, 0.0, 0.0, 1.0])

        center = ti.Vector([x, -y, 0.0]) + offset
        dist = (p - center).norm()
        if dist < radius:
            smoke[I] = ti.Vector([0.0, 0.0, 0.0, 1.0])

@ti.kernel
def clamp_smoke_density(smoke: ti.template()):
    for I in ti.grouped(smoke):
        if smoke[I][0] > 1.0:
            smoke[I][0] = 1.0
        if smoke[I][1] > 1.0:
            smoke[I][1] = 1.0
        if smoke[I][2] > 1.0:
            smoke[I][2] = 1.0
        if smoke[I][3] > 1.0:
            smoke[I][3] = 1.0