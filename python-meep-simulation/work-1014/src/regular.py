import meep as mp
import numpy as np
import scipy.constants as C
import argparse

parser = argparse.ArgumentParser(description='.....')
parser.add_argument('-w', type=float, default=1.0, help='width of waveguide')
parser.add_argument('-s', type=float, default=1.5, help='separation of waveguide')
args = parser.parse_args()

# global parameters

scale_fac = 50 * C.nano  # distance that one grid point represents
resolution = 10  # how many grid points per unit

core_eps = 4.0
clad_eps = 2.25

width = args.w
separation = args.s
s_len = 200
edge_off = 5
cell_x = s_len + edge_off * 2
cell_y = (width / 2 + separation / 2 + edge_off) * 2

# source
src_freq = 1 / ((1550 * C.nano) / (scale_fac * resolution))  # 0.32
src_x = -s_len / 2
src_y = -separation / 2
sources = [mp.Source(mp.ContinuousSource(frequency=src_freq), component=mp.Ey, center=mp.Vector3(src_x, src_y),
                     size=mp.Vector3(0, width, 0))]

# geometry
wvg_1_x = s_len / 4
wvg_3_x = -s_len / 4
geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=mp.Medium(epsilon=clad_eps)),
            mp.Block(size=mp.Vector3(cell_x, width), center=mp.Vector3(0, separation / 2),
                     material=mp.Medium(epsilon=core_eps)),
            mp.Block(size=mp.Vector3(cell_x, width), center=mp.Vector3(0, -separation / 2),
                     material=mp.Medium(epsilon=core_eps))]

pml_layers = [mp.PML(1.0)]
cell = mp.Vector3(cell_x, cell_y, 0)
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# 1号波导在下面， 3号波导在上面，没有2号波导
point_end_1 = mp.Vector3(s_len / 2 - 1, -separation / 2)
point_end_3 = mp.Vector3(s_len / 2 - 1, separation / 2)
volume_end_1 = mp.Volume(center=point_end_1, size=mp.Vector3(0, width, 0))
volume_end_3 = mp.Volume(center=point_end_3, size=mp.Vector3(0, width, 0))

volume_all = mp.Volume(center=mp.Vector3(), size=cell)

sim.run(mp.at_beginning(mp.in_volume(mp.Volume(center=mp.Vector3(), size=cell), mp.output_epsilon)),
        mp.to_appended('ey', mp.at_every(10, mp.in_volume(volume_all, mp.output_efield_y))),
        mp.to_appended('dpwr', mp.at_every(10, mp.in_volume(volume_all, mp.output_dpwr))),
        mp.to_appended('dpwr-pt-1', mp.at_every(1, mp.in_point(point_end_1, mp.output_dpwr))),
        mp.to_appended('dpwr-pt-3', mp.at_every(1, mp.in_point(point_end_3, mp.output_dpwr))),
        mp.to_appended('dpwr-vol-1', mp.at_every(1, mp.in_volume(volume_end_1, mp.output_dpwr))),
        mp.to_appended('dpwr-vol-3', mp.at_every(1, mp.in_volume(volume_end_3, mp.output_dpwr))),
        until=700)
