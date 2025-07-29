import taichi as ti
import numpy as np
import random
from util.taichi_utils import *

def ink_drop_particle_case0(laden_particle_pos,num):
    particle_num=0
    laden_particle_pos_np=np.zeros(shape=(num,3), dtype=float)
    random.seed(0)
    while(True):
        r=3.0
        center=64.0
        domain_range_y=256.0
        x,y,z=random.random()*r*2+center-r,random.random()*r*2+0.05*domain_range_y,random.random()*r*2+center-r
        if((x-center)**2+(y-0.05*domain_range_y-r)**2+(z-center)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num):
            break
    laden_particle_pos.from_numpy(laden_particle_pos_np)  

@ti.kernel
def ink_drop_vel_case0(laden_particle_vel:ti.template(),laden_particle_pos:ti.template(),laden_particle_num:int):
    for I in ti.grouped(laden_particle_vel):
        if(I[0]<laden_particle_num):
            laden_particle_vel[I]=ti.Vector([0.0,1.0,0.0])

@ti.kernel
def ink_drop_vel_grid_case0(u:ti.template(),X:ti.template()):
    r=3.0
    center=64.0
    domain_range_y=256.0
    for I in ti.grouped(X):
        x,y,z=X[I][0],X[I][1],X[I][2]
        if((x-center)**2+(y-0.05*domain_range_y-r)**2+(z-center)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])

@ti.kernel
def init_ref_radius(
    laden_particle_ref_radius:ti.template(),
    laden_particle_num:int,
    compute_laden_radius:float
):
    for I in ti.grouped(laden_particle_ref_radius):
        if(I[0]<laden_particle_num):
            laden_particle_ref_radius[I]=compute_laden_radius

def ink_init_case0(
    laden_particles_pos:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particle_num:ti.template(),
    u:ti.template(),
    X:ti.template(),
    drag_x:ti.template(),
    drag_y:ti.template(),
    drag_z:ti.template(),
    laden_particle_ref_radius:ti.template(),
    compute_laden_radius:float
):    
    ink_drop_particle_case0(laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_case0(laden_particles_vel,laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_grid_case0(u,X)
    drag_x.fill(0.0)
    drag_y.fill(0.0)
    drag_z.fill(0.0)
    init_ref_radius(laden_particle_ref_radius,laden_particle_num[None],compute_laden_radius)


def ink_drop_particle_case1(laden_particle_pos,num):
    particle_num=0
    laden_particle_pos_np=np.zeros(shape=(num,3), dtype=float)
    random.seed(0)
    while(True):
        r=8.0
        domain_range_x=128.0
        domain_range_y=256.0
        domain_range_z=128.0
        center_x = domain_range_x * 0.3
        center_z = domain_range_z * 0.3
        x,y,z=random.random()*r*2+center_x-r,random.random()*r*2+0.05*domain_range_y,random.random()*r*2+center_z-r
        if((x-center_x)**2+(y-0.05*domain_range_y-r)**2+(z-center_z)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num):
            break
    laden_particle_pos.from_numpy(laden_particle_pos_np)  

@ti.kernel
def ink_drop_vel_case1(laden_particle_vel:ti.template(),laden_particle_pos:ti.template(),laden_particle_num:int):
    for I in ti.grouped(laden_particle_vel):
        if(I[0]<laden_particle_num):
            laden_particle_vel[I]=ti.Vector([0.0,0.0,0.0])

@ti.kernel
def ink_drop_vel_grid_case1(u:ti.template(),X:ti.template()):
    r=8.0
    domain_range_x=128.0
    domain_range_y=256.0
    domain_range_z=128.0
    center_x = domain_range_x * 0.3
    center_z = domain_range_z * 0.3
    for I in ti.grouped(X):
        x,y,z=X[I][0],X[I][1],X[I][2]
        if((x-center_x)**2+(y-0.05*domain_range_y-r)**2+(z-center_z)**2<r**2):
            u[I]=ti.Vector([0.2,0.5,0.3])

def ink_init_case1(
    laden_particles_pos:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particle_num:ti.template(),
    u:ti.template(),
    X:ti.template(),
    drag_x:ti.template(),
    drag_y:ti.template(),
    drag_z:ti.template(),
    laden_particle_ref_radius:ti.template(),
    compute_laden_radius:float
):    
    ink_drop_particle_case1(laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_case1(laden_particles_vel,laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_grid_case1(u,X)
    drag_x.fill(0.0)
    drag_y.fill(0.0)
    drag_z.fill(0.0)
    init_ref_radius(laden_particle_ref_radius,laden_particle_num[None],compute_laden_radius)