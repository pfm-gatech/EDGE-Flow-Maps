import taichi as ti
import numpy as np
import random
from util.taichi_utils import *

@ti.kernel
def update_particles_acc(
    u_x:ti.template(),
    u_y:ti.template(),
    u_z:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particles_pos:ti.template(),
    laden_particles_acc:ti.template(),
    dx:float,
    curr_dt:float,
    laden_particle_num:int,
    St: float,
    gravity: float
):
    laden_particles_acc.fill(0.0)
    for I in ti.grouped(laden_particles_acc):
        if(I[0]<laden_particle_num):
            k=1/St
            particle_ux,grad_u_x_p=interp_grad_2(u_x, laden_particles_pos[I], dx, BL_x=0, BL_y=0.5, BL_z=0.5)
            particle_uy,grad_u_y_p=interp_grad_2(u_y, laden_particles_pos[I], dx, BL_x=0.5, BL_y=0, BL_z=0.5)
            particle_uz,grad_u_z_p=interp_grad_2(u_z, laden_particles_pos[I], dx, BL_x=0.5, BL_y=0.5, BL_z=0)
            laden_particles_acc[I]+=ti.Vector([0.0,gravity,0.0])
            laden_particles_acc[I]-=k*(laden_particles_vel[I]-ti.Vector([particle_ux,particle_uy,particle_uz]))
            laden_particles_acc[I]/=(1+k*curr_dt)
            

@ti.kernel
def update_particles_velocity(
    laden_particles_acc:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particles_pos:ti.template(),
    curr_dt:float,
    dx:float,
    laden_particle_num:int,
    domain_range_x:float,
    domain_range_y:float,
    domain_range_z:float
):
    for I in ti.grouped(laden_particles_vel):
        if(I[0]<laden_particle_num):
            laden_particles_vel[I]+=laden_particles_acc[I]*curr_dt

            if(laden_particles_pos[I][1]<dx and laden_particles_vel[I][1]<0):
                laden_particles_vel[I][1]=0.0
                #laden_particles_vel[I]=ti.Vector([0.0,0.0,0.0])
                
            if(laden_particles_pos[I][1]>domain_range_y-3*dx and laden_particles_vel[I][1]>0):
                laden_particles_vel[I][1]=0.0
                #laden_particles_vel[I]=ti.Vector([0.0,0.0,0.0])

            if(laden_particles_pos[I][0]<3*dx and laden_particles_vel[I][0]<0):
                laden_particles_vel[I][0]=0.0
                #laden_particles_vel[I]=ti.Vector([0.0,0.0,0.0])

            if(laden_particles_pos[I][0]>domain_range_x-3*dx and laden_particles_vel[I][0]>0):
                laden_particles_vel[I][0]=0.0
                #laden_particles_vel[I]=ti.Vector([0.0,0.0,0.0])

            if(laden_particles_pos[I][2]<3*dx and laden_particles_vel[I][2]<0):
                laden_particles_vel[I][2]=0.0
                #laden_particles_vel[I]=ti.Vector([0.0,0.0,0.0])

            if(laden_particles_pos[I][2]>domain_range_z-3*dx and laden_particles_vel[I][2]>0):
                laden_particles_vel[I][2]=0.0
                #laden_particles_vel[I]=ti.Vector([0.0,0.0,0.0])


@ti.kernel
def calculate_velocity_diff(
    u_x:ti.template(),
    u_y:ti.template(),
    u_z:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particles_vel_diff:ti.template(),
    laden_particles_pos:ti.template(),
    dx:float,
    laden_particle_num:int,
    St: float,
    density_ratio: float,
    epiral_radius_ratio: float
):

    epiral_ratio=1.0/epiral_radius_ratio**3
    #epiral_ratio=1.0/35**3
    # print("epiral_ratio",epiral_ratio)
    for I in ti.grouped(laden_particles_vel_diff):
        if(I[0]<laden_particle_num):
            k=1.0/St*density_ratio*epiral_ratio
            particle_ux,grad_u_x_p=interp_grad_2(u_x, laden_particles_pos[I], dx, BL_x=0, BL_y=0.5, BL_z=0.5)
            particle_uy,grad_u_y_p=interp_grad_2(u_y, laden_particles_pos[I], dx, BL_x=0.5, BL_y=0, BL_z=0.5)
            particle_uz,grad_u_z_p=interp_grad_2(u_z, laden_particles_pos[I], dx, BL_x=0.5, BL_y=0.5, BL_z=0)
            laden_particles_vel_diff[I]=k*(laden_particles_vel[I]-ti.Vector([particle_ux,particle_uy,particle_uz]))


@ti.kernel
def laden_particles_back_with_computing_particle(
    drag_x:ti.template(),
    drag_y:ti.template(),
    drag_z:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    u_z:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particles_vel_diff:ti.template(),
    laden_particles_pos:ti.template(),
    laden_particle_ref_radius:ti.template(),
    dx:float,
    laden_particle_num:int,
    laden_radius:float,
    domain_range_x:float,
    domain_range_y:float,
    domain_range_z:float,
    res_x:int,
    res_y:int,
    res_z:int
):

    drag_x.fill(0.0)
    drag_y.fill(0.0)
    drag_z.fill(0.0)
    
    for I in ti.grouped(laden_particles_pos):
        if(I[0]<laden_particle_num):
            if(laden_particles_pos[I][0]<3*dx or laden_particles_pos[I][1]<3*dx or laden_particles_pos[I][2]<3*dx or laden_particles_pos[I][0]>domain_range_x-3*dx or laden_particles_pos[I][1]>domain_range_y-3*dx or laden_particles_pos[I][2]>domain_range_z-3*dx 
            ):
                pass
            else:
                pos = laden_particles_pos[I] / dx
                base_face_id = int(pos - 0.5 * ti.Vector.unit(3, 1)-0.5 * ti.Vector.unit(3, 2))
                # for offset in ti.static(ti.grouped(ti.ndrange(*((-1,3),) * 3))):
                for offset in ti.grouped(ti.ndrange(*((-1,3),) * 3)):
                    face_id = base_face_id + offset
                    if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                        weight = N_2((pos[0] - face_id[0])) * N_2((pos[1] - face_id[1] - 0.5)) * N_2((pos[2] - face_id[2] - 0.5))
                        drag_x[face_id] += laden_particles_vel_diff[I][0] * weight*laden_particle_ref_radius[I]**3/laden_radius**3

                base_face_id = int(pos - 0.5 * ti.Vector.unit(3, 0) - 0.5 * ti.Vector.unit(3, 2))
                # for offset in ti.static(ti.grouped(ti.ndrange(*((-1,3),) * 3))):
                for offset in ti.grouped(ti.ndrange(*((-1,3),) * 3)):
                    face_id = base_face_id + offset
                    if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
                        weight = N_2((pos[0] - face_id[0] - 0.5)) * N_2((pos[1] - face_id[1])) * N_2((pos[2] - face_id[2]-0.5))
                        drag_y[face_id] += laden_particles_vel_diff[I][1] * weight*laden_particle_ref_radius[I]**3/laden_radius**3

                base_face_id = int(pos - 0.5 * ti.Vector.unit(3, 0) - 0.5 * ti.Vector.unit(3, 1))
                # for offset in ti.static(ti.grouped(ti.ndrange(*((-1,3),) * 3))):
                for offset in ti.grouped(ti.ndrange(*((-1,3),) * 3)):
                    face_id = base_face_id + offset
                    if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
                        weight = N_2((pos[0] - face_id[0] - 0.5)) * N_2((pos[1] - face_id[1]-0.5)) * N_2((pos[2] - face_id[2]))
                        drag_z[face_id] += laden_particles_vel_diff[I][2] * weight*laden_particle_ref_radius[I]**3/laden_radius**3

@ti.kernel
def update_particles_position(laden_particles_vel:ti.template(),laden_particles_pos:ti.template(),curr_dt:float,laden_particle_num:int, domain_range_x:float, domain_range_y:float, domain_range_z:float):
    for I in ti.grouped(laden_particles_pos):
        if(I[0]<laden_particle_num):
            laden_particles_pos[I]+=laden_particles_vel[I]*curr_dt

            
            if(laden_particles_pos[I][1]<0):
                laden_particles_pos[I][1]=0.0
        
            if(laden_particles_pos[I][1]>domain_range_y):
                laden_particles_pos[I][1]=domain_range_y
            
            if(laden_particles_pos[I][0]<0):
                laden_particles_pos[I][0]=0.0
        
            if(laden_particles_pos[I][0]>domain_range_x):
                laden_particles_pos[I][0]=domain_range_x

            if(laden_particles_pos[I][2]<0):
                laden_particles_pos[I][2]=0.0
        
            if(laden_particles_pos[I][2]>domain_range_z):
                laden_particles_pos[I][2]=domain_range_z

@ti.kernel
def calculate_viscous_force(
    u_x: ti.template(),
    u_y: ti.template(),
    u_z: ti.template(),
    vis_x: ti.template(),
    vis_y: ti.template(),
    vis_z: ti.template(),
    dx: float,
    Re: float
):
    for i, j, k in u_x:
        viscosity = 1.0/Re
        vis_x[i, j,k] = viscosity*(
            sample(u_x, i+1, j, k)+sample(u_x, i-1, j, k) +
            sample(u_x, i, j+1, k)+sample(u_x, i, j-1, k) +
            sample(u_x, i, j, k+1)+sample(u_x, i, j, k-1) +
            -6*sample(u_x, i, j, k)
        )/dx/dx

    for i, j, k in u_y:
        viscosity = 1.0/Re
        vis_y[i, j,k] = viscosity*(
            sample(u_y, i+1, j, k)+sample(u_y, i-1, j, k) +
            sample(u_y, i, j+1, k)+sample(u_y, i, j-1, k) +
            sample(u_y, i, j, k+1)+sample(u_y, i, j, k-1) +
            -6*sample(u_y, i, j, k)
        )/dx/dx

    for i, j, k in u_z:
        viscosity = 1.0/Re
        vis_z[i, j,k] = viscosity*(
            sample(u_z, i+1, j, k)+sample(u_z, i-1, j, k) +
            sample(u_z, i, j+1, k)+sample(u_z, i, j-1, k) +
            sample(u_z, i, j, k+1)+sample(u_z, i, j, k-1) +
            -6*sample(u_z, i, j, k)
        )/dx/dx