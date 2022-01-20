import meep as mp
import math
import scipy.constants as C
import matplotlib.pyplot as plt
import numpy as np

# Setup the main structure parameters and also the simulation size

scale_fac = 50*C.nano    # distance that one grid point represents
resolution = 10    # how many grid points per unit
width = 1.0           # waveguide width = 3microns

# 直-弯侧向耦合波导
separation = 2.0          # spacing between waveguides =4micron
# bend_r = 500        # bend radius = 500 micron
bend_r = 100
air_eps = 1.0
core_eps = 4.0  # 内层
clad_eps = 2.25  # 包层

# s_len = 500*2
s_len = 100*2
theta = s_len/2/bend_r  # unit: rad


edge_x = s_len*0.1
edge_y = edge_x

cell_x = s_len+edge_x*2
cell_y = (bend_r+separation/2+width/2+edge_y)*2
cell = mp.Vector3(cell_x, cell_y, 0)

geometry = [mp.Block(mp.Vector3(mp.inf, mp.inf, mp.inf),
                     center=mp.Vector3(), material=mp.Medium(epsilon=clad_eps))]
geometry.append(mp.Block(mp.Vector3(s_len, width, mp.inf), center=mp.Vector3(0, (width+separation)/2), material=mp.Medium(epsilon=core_eps)))
geometry.append(mp.Cylinder(radius=bend_r+width/2, center=mp.Vector3(0, -(separation/2+width/2+bend_r)), material=mp.Medium(epsilon=core_eps)))
geometry.append(mp.Cylinder(radius=bend_r-width/2, center=mp.Vector3(0, -(separation/2+width/2+bend_r)), material=mp.Medium(epsilon=clad_eps)))
geometry.append(mp.Block(mp.Vector3(2*bend_r/np.tan(theta),bend_r*2),center=mp.Vector3(bend_r/np.sin(theta),-(separation/2+width/2+bend_r)), e1=mp.Vector3(np.cos(theta),-np.sin(theta)), e2=mp.Vector3(np.sin(theta),np.cos(theta)), material=mp.Medium(epsilon=clad_eps)))
geometry.append(mp.Block(mp.Vector3(2*bend_r/np.tan(theta),bend_r*2),center=mp.Vector3(-bend_r/np.sin(theta),-(separation/2+width/2+bend_r)), e1=mp.Vector3(-np.cos(theta),-np.sin(theta)), e2=mp.Vector3(-np.sin(theta),np.cos(theta)), material=mp.Medium(epsilon=clad_eps)))


# monitoring points
mpt_straight_right = mp.Vector3(s_len/2-1,(separation+width)/2)
mpt_straight_center = mp.Vector3(0,(separation+width)/2)
mpt_bend_center = mp.Vector3(0,-(separation/2+width/2))
mpt_bend_left = mp.Vector3(-bend_r*np.sin(theta),-(bend_r*(1-np.cos(theta)) + separation/2 + width/2))
mpt_bend_right = mp.Vector3(bend_r*np.sin(theta),-(bend_r*(1-np.cos(theta)) + separation/2 + width/2))
mpt_vacancy = (mpt_straight_right + mpt_bend_right)/2


srcfrq = 1/((1550*C.nano)/(scale_fac*resolution))  # 1.03
sources = [mp.Source(mp.ContinuousSource(frequency=srcfrq),
                     component=mp.Ey,
                     center=mp.Vector3(-s_len/2-1, (separation+width)/2, 0))]
pml_layers = [mp.PML(1.0)]


sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)


obs_size = mp.Vector3(s_len,width)
obs_center = mp.Vector3(0,(separation+width)/2)
# sim.init_sim()
sim.use_output_directory('wvg-out')
# sim.run(mp.at_beginning(mp.output_epsilon),until=10)

# sim.run(mp.at_beginning(mp.in_volume(mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(cell_x,cell_y,0)), mp.output_png(mp.output_epsilon,"-Zc dkbluered"))),
sim.run(mp.at_beginning(mp.in_volume(mp.Volume(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0)), mp.output_epsilon)),
       mp.to_appended("ey", mp.at_every(200.0, mp.in_volume(mp.Volume(center=mp.Vector3(), size=cell), mp.output_efield_y))),
       mp.to_appended("dpwr", mp.at_every(200.0, mp.in_volume(mp.Volume(center=mp.Vector3(), size=cell), mp.output_dpwr))),
        mp.to_appended("ey_straight_right", mp.at_every(1, mp.in_point(mpt_straight_right, mp.output_efield_y))),
        mp.to_appended("ey_straight_center", mp.at_every(1, mp.in_point(mpt_straight_center, mp.output_efield_y))),
        mp.to_appended("ey_bend_center", mp.at_every(1, mp.in_point(mpt_bend_center, mp.output_efield_y))),
        mp.to_appended("ey_bend_left", mp.at_every(1, mp.in_point(mpt_bend_left, mp.output_efield_y))),
        mp.to_appended("ey_bend_right", mp.at_every(1, mp.in_point(mpt_bend_right, mp.output_efield_y))),
        mp.to_appended("ey_vacancy", mp.at_every(1, mp.in_point(mpt_vacancy, mp.output_efield_y))),
        mp.to_appended("dpwr_straight_right", mp.at_every(1, mp.in_point(mpt_straight_right, mp.output_dpwr))),
        mp.to_appended("dpwr_straight_center", mp.at_every(1, mp.in_point(mpt_straight_center, mp.output_dpwr))),
        mp.to_appended("dpwr_bend_center", mp.at_every(1, mp.in_point(mpt_bend_center, mp.output_dpwr))),
        mp.to_appended("dpwr_bend_left", mp.at_every(1, mp.in_point(mpt_bend_left, mp.output_dpwr))),
        mp.to_appended("dpwr_bend_right", mp.at_every(1, mp.in_point(mpt_bend_right, mp.output_dpwr))),
        mp.to_appended("dpwr_vacancy", mp.at_every(1, mp.in_point(mpt_vacancy, mp.output_dpwr))),
        until=500)


obs_size = mp.Vector3(s_len,width)
obs_center = mp.Vector3(0,(separation+width)/2)
ez_data = sim.get_array(center=obs_center, size=obs_size, component=mp.Ey)
eps_data = sim.get_array(center=obs_center, size=obs_size, component=mp.Dielectric)
eps_data.shape
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.show()