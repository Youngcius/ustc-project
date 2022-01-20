import meep as mp
from meep import mpb
import matplotlib.pyplot as plt

# 1-D band structure
num_bands = 6
k_points = [mp.Vector3(), mp.Vector3(0.5)]
k_points = mp.interpolate(9, k_points)
core_eps = 4.0
clad_eps = 2.25
lattice = mp.Lattice(size=mp.Vector3(1, 7))
radius = 0.3
geometry = [mp.Block(size=mp.Vector3(1, 7), material=mp.Medium(epsilon=clad_eps)),
            mp.Block(size=mp.Vector3(1, 1), material=mp.Medium(epsilon=core_eps)),
            mp.Cylinder(radius=radius, material=mp.Medium(epsilon=clad_eps))]
#
ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice, resolution=16, k_points=k_points,
                    num_bands=num_bands)
#


ms.run_te()
ms.run_tm()

md = mpb.MPBData(rectify=True, periods=1, resolution=16)
eps = ms.get_epsilon()
converted_eps = md.convert(eps)

plt.imshow(converted_eps.T, interpolation='spline36', cmap='binary')
plt.axis('off')
plt.title('1-D period structure (PhCs)')
plt.savefig('period_structure.png')
# plt.show()