from BinaryLiquid import BLPlotter, BinaryLiquid, X_vals, X_logs
from BinarySystems import *
import pandas as pd
from HSX_nonzeroS import HSX
import matplotlib.pyplot as plt


GaPb_bl = BinaryLiquid.from_cache("Ga-Pb", params=[22810, -8.993, 4906, -5.967])
GaIn_bl = BinaryLiquid.from_cache("Ga-In", params=[5141, -0.79, 3145, -9.81])

# Form: [[InΔH, InΔS PbΔH, PbΔS], [L0_a, L0_b, L1_a, L1_b]] H-> J/mol, S-> J/mol*K
InPb_phases = {'In(A6)': [[0, 0.0, 4644, 0.233], [-2859, 0, 3341, 0]],
               'ɑ': [[38, 0.002, 967, 1.183], [3246, 0, 939, 0]],
               'Pb(A1)': [[42, 0.012, 0, 0.0], [5069, -2.624, 456, 0.875]],
               'L': [[3291, 7.658, 4774, 7.947], [3775, -1.285, 183, 0.381]]}


def gen_phase_points(phases):
    data = {'X': [], 'S': [], 'H': [], 'Phase Name': []}
    R = 8.314

    for phase, params in phases.items():
        data['X'].extend(list(X_vals))
        data['Phase Name'].extend([phase for _ in X_vals])

        H_a = params[0][0]
        H_b = params[0][2]
        H_lc = (H_a * X_vals[-2:0:-1] +
                H_b * X_vals[1:-1])
        H_xs = X_vals[1:-1] * X_vals[-2:0:-1] * (params[1][0] + params[1][2] * (1 - 2 * X_vals[1:-1]))
        phase_H = list(H_lc + H_xs)
        phase_H.insert(0, H_a)
        phase_H.append(H_b)
        data['H'].extend(phase_H)

        S_a = params[0][1]
        S_b = params[0][3]
        S_lc = (S_a * X_vals[-2:0:-1] +
                S_b * X_vals[1:-1])
        S_ideal = -R * (X_vals[1:-1] * X_logs + X_vals[-2:0:-1] * X_logs[::-1])
        S_xs = -X_vals[1:-1] * X_vals[-2:0:-1] * (params[1][1] + params[1][3] * (1 - 2 * X_vals[1:-1]))
        phase_S = list(S_lc + S_ideal + S_xs)
        phase_S.insert(0, S_a)
        phase_S.append(S_b)
        data['S'].extend(phase_S)

    return data


def fit_binary_phases_at_temp(phases_dict, solute_element, temp_c):
    phase_dfs = {phase_name: group for phase_name, group in pd.DataFrame(phases_dict).groupby('Phase Name')}
    temp_k = temp_c + 273.15

    for phase_name, phase_data in phase_dfs.items():
        xdata = np.array(phase_data['X'])
        Gdata = np.array(phase_data['H'] - temp_k*phase_data['S'])

        phase = BinaryIsothermal2ndOrderPhase()
        phase.fit_phase(xdata, Gdata, kwellmax=1e6)

        print(f"Fitted parameters for {phase_name}:")
        print(f"fmin: {phase.fmin}")
        print(f"kwell (curvature): {phase.kwell}")
        print(f"cmin (composition at minimum): {phase.cmin}")
        print()
        plt.plot(xdata, Gdata, label=phase_name)

    plt.xlabel(f"Mole Fraction {solute_element}")
    plt.ylabel("Gibbs Free Energy (J/mol)")
    plt.title(f"Phase Free Energies at {temp_c}°C")
    plt.legend()
    plt.show()


InPb_data = gen_phase_points(InPb_phases)
hsx_dict = {'data': InPb_data, 'phases': [phase for phase in InPb_phases.keys()], 'comps': ['In', 'Pb']}
InPb_hsx = HSX(hsx_dict, [100, 340])
InPb_hsx.plot_tx_scatter()
fit_binary_phases_at_temp(InPb_data, solute_element="PB", temp_c=150)

GaPb_data = GaPb_bl.export_phases_points()
GaPb_bl.update_phase_points()
GaPb_bl.hsx.plot_tx_scatter()
fit_binary_phases_at_temp(GaPb_data, solute_element="PB", temp_c=150)

GaIn_data = GaIn_bl.export_phases_points()
GaIn_bl.update_phase_points()
GaIn_bl.hsx.plot_tx_scatter()
fit_binary_phases_at_temp(GaIn_data, solute_element="IN", temp_c=150)

