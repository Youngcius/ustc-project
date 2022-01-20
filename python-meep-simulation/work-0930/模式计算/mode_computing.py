import meep as mp
from meep import mpb
import matplotlib.pyplot as plt
import sys

N = int(sys.argv[1])  # number of holes, e.g. 0, 10, 16
core_eps = 4.0
clad_eps = 2.25
s_len = 20
width = 1
radius = 0.3
edge_off = 2
pml_layers = [mp.PML(1)]
cell = mp.Vector3(s_len + edge_off * 2, width + edge_off * 2)

# source
freq_center = 0.5
freq_width = 5
sources = [mp.Source(mp.GaussianSource(frequency=freq_center, fwidth=freq_width),
                     component=mp.Ey,
                     center=mp.Vector3(-s_len / 2))]

geometry = [mp.Block(size=cell, material=mp.Medium(epsilon=clad_eps)),
            mp.Block(size=mp.Vector3(s_len, width), material=mp.Medium(epsilon=core_eps))]

for i in range(1, int(N / 2)):
    geometry.append(mp.Cylinder(radius=radius, center=mp.Vector3(i), material=mp.Medium(epsilon=clad_eps)))
    geometry.append(mp.Cylinder(radius=radius, center=mp.Vector3(-i), material=mp.Medium(epsilon=clad_eps)))

# transmission spectrum for mode
sim = mp.Simulation(geometry=geometry, cell_size=cell, resolution=10, boundary_layers=pml_layers, sources=sources)

flux_region = mp.FluxRegion(center=mp.Vector3(s_len / 2 - 1), size=mp.Vector3(0, 2 * width))
flux = sim.add_flux(freq_center, freq_width, 500, flux_region)
sim.run(mp.at_beginning(mp.output_epsilon),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(s_len / 2 - 1), 1e-3))

sim.display_fluxes(flux)
