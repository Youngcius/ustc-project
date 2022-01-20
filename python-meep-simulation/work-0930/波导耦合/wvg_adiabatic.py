import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import argparse

parser = argparse.ArgumentParser(description='.....')
parser.add_argument('-theta', type=float, default=1.0, help='unit: degree')
args = parser.parse_args()

# global parameters
theta = args.theta / 180 * C.pi

scale_fac = 50 * C.nano  # distance that one grid point represents
resolution = 10  # how many grid points per unit

core_eps = 4.0
clad_eps = 2.25

width = 1.0
separation = 1.5
s_len = 200
bend_r = 3 * s_len / 4 / np.sin(2 * theta)
print('bend_r:', bend_r)
edge_off = 5
cell_x = s_len + edge_off * 2
# cell_y = (bend_r + separation + width + edge_off) * 2
cell_y = (3 * s_len / 4 * np.tan(theta) + separation + edge_off) * 2

# source
src_freq = 1 / ((1550 * C.nano) / (scale_fac * resolution))  # 0.32
src_x = -s_len / 2
src_y = -3 * s_len / 4 * np.tan(theta) - separation
sources = [mp.Source(mp.ContinuousSource(frequency=src_freq),
                     component=mp.Ey,
                     center=mp.Vector3(src_x, src_y))]

# geometry
wvg_1_x = s_len / 4
wvg_3_x = -s_len / 4
geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=mp.Medium(epsilon=clad_eps)),
            mp.Block(size=mp.Vector3(cell_x, width), center=mp.Vector3(), material=mp.Medium(epsilon=core_eps)),
            mp.Cylinder(bend_r + width / 2, center=mp.Vector3(wvg_1_x, -(separation + bend_r)),
                        material=mp.Medium(epsilon=core_eps)),
            mp.Cylinder(bend_r + width / 2, center=mp.Vector3(wvg_3_x, separation + bend_r),
                        material=mp.Medium(epsilon=core_eps)),
            mp.Cylinder(bend_r - width / 2, center=mp.Vector3(wvg_1_x, -(separation + bend_r)),
                        material=mp.Medium(epsilon=clad_eps)),
            mp.Cylinder(bend_r - width / 2, center=mp.Vector3(wvg_3_x, separation + bend_r),
                        material=mp.Medium(epsilon=clad_eps)),
            mp.Block(size=mp.Vector3((cell_x - s_len) / 2, cell_y), center=mp.Vector3((cell_x + s_len) / 4),
                     material=mp.Medium(epsilon=clad_eps)),
            mp.Block(size=mp.Vector3((cell_x - s_len) / 2, cell_y), center=mp.Vector3(-(cell_x + s_len) / 4),
                     material=mp.Medium(epsilon=clad_eps))]

pml_layers = [mp.PML(1.0)]
cell = mp.Vector3(cell_x, cell_y, 0)
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# monitor points
# mpt_1_start = mp.Vector3(-s_len/2,-3 * s_len / 4 * np.tan(theta) - separation)  # input waveguide
# mpt_1_end = mp.Vector3(s_len/2,)
# mpt_2_start = mp.Vector3(-s_len/2)  # center waveguide
# mpt_2_end = mp.Vector3(s_len)
# mpt_3_start = mp.Vector3(-s_len/2,)  # cross waveguide
# mpt_3_end = mp.Vector3(s_len/2)

volume_all = mp.Volume(center=mp.Vector3(), size=cell)

# transmission spectrum for mode
# sources_trans = [mp.Source(mp.GaussianSource(src_freq, fwidth=5), component=mp.Ey, center=mp.Vector3(src_x, src_y),
#                            size=mp.Vector3(0, width))]
# flux_region_1 = mp.FluxRegion(center=mpt_1_end, size=mp.Vector3(0, 2 * width))
# flux_region_2 = mp.FluxRegion(center=mpt_2_end, size=mp.Vector3(0, 2 * width))
# flux_region_3 = mp.FluxRegion(center=mpt_3_end, size=mp.Vector3(0, 2 * width))

sim.run(mp.at_beginning(mp.in_volume(mp.Volume(center=mp.Vector3(), size=cell), mp.output_epsilon)),
        mp.to_appended('ey', mp.at_every(10, mp.in_volume(volume_all, mp.output_efield_y))),
        # mp.to_appended('dpwr', mp.at_every(10, mp.in_volume(volume_all, mp.output_dpwr))),
        # mp.to_appended("dpwr_wvg_1_start", mp.at_every(1, mp.in_point(mpt_1_start, mp.output_dpwr))),
        # mp.to_appended("dpwr_wvg_1_end", mp.at_every(1, mp.in_point(mpt_2_end, mp.output_dpwr))),
        # mp.to_appended("dpwr_wvg_2_start", mp.at_every(1, mp.in_point(mpt_2_start, mp.output_dpwr))
        # mp.to_appended("dpwr_wvg_2_end", mp.at_every(1, mp.in_point(mpt_2_end, mp.output_dpwr))),
        # mp.to_appended("dpwr_wvg_3_start", mp.at_every(1, mp.in_point(mpt_3_start, mp.output_dpwr))),
        # mp.to_appended("dpwr_wvg_3_end", mp.at_every(1, mp.in_point(mpt_3_end, mp.output_dpwr))),
        until=500)
