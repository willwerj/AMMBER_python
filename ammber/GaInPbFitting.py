from BinaryLiquid import BLPlotter, BinaryLiquid
from BinarySystems import *

InPb_bl = BinaryLiquid.from_cache("In-Pb")
GaPb_bl = BinaryLiquid.from_cache("Ga-Pb", params=[22810, -8.993, 4906, -5.967])
GaIn_bl = BinaryLiquid.from_cache("Ga-In", params=[5141, -0.79, 3145, -9.81])

# GaIn_blp = BLPlotter(GaIn_bl)
# GaIn_blp.show(plot_type='fit+liq')
#
# GaPb_blp = BLPlotter(GaPb_bl)
# GaPb_blp.show(plot_type='fit+liq')

points_dict = GaPb_bl.export_phases_points()
for key, coords in points_dict.items():
    print(key, coords)
