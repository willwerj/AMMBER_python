# TDB Test AlCu
############################################################
import BinarySystems as BS
import matplotlib.pyplot as plt
from pycalphad import Database
tdb_file = "AlCu.TDB"
db = Database(tdb_file)
# pycalphad uses capitalized elements
elements = ["AL", "CU"]
component = "CU"
temperature = 800.0

AlCu_Sys = BS.BinaryIsothermalDiscreteSystem()
AlCu_Sys.fromTDB(db, elements, component, temperature)
AlCu_Fit = BS.BinaryIsothermal2ndOrderSystem()
AlCu_Fit.from_discrete_near_equilibrium(AlCu_Sys, x=0.3)

print(AlCu_Fit.phases)
plt.plot(AlCu_Sys.phases['FCC_A1'].xdata,
         AlCu_Fit.phases['FCC_A1_0'].free_energy(AlCu_Sys.phases['FCC_A1'].xdata),
         linestyle='-', color='tab:orange',
         label="Phase 1 Fit",linewidth=2)

plt.plot(AlCu_Sys.phases['FCC_A1'].xdata,
         AlCu_Sys.phases['FCC_A1'].Gdata,
         linestyle='-', color='k',
         label="Phase 1 Fit",linewidth=1)
#plt.plot(AlCu_Sys.phases['AL2CU_C16'].xdata,
#         AlCu_Fit.phases['AL2CU_C16'].free_energy(AlCu_Sys.phases['AL2CU_C16'].xdata),
#         linestyle='-', color='tab:green',
#         label="Phase 1 Fit",linewidth=2)
#
#plt.plot(AlCu_Sys.phases['AL2CU_C16'].xdata,
#         AlCu_Sys.phases['AL2CU_C16'].Gdata,
#         linestyle='-', color='k',
#         label="Phase 1 Fit",linewidth=1)

#plt.ylim([-56000,-47000])
plt.xlim([0,1])
plt.xticks([])
#plt.yticks([])

############################################################

# Test B
############################################################
