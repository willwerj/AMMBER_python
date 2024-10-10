"""
@author: Joshua Willwerth

This script merges the previously implemented Gliquid_Optimization, LIFT_Optimization_HSX.py, and Gliquid_Reconstruction
into a single file for maximum functionality.
"""
from __future__ import annotations

import math
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib import cm
from matplotlib.colors import LogNorm
from itertools import combinations
from pymatgen.analysis.phase_diagram import PDPlotter, PhaseDiagram, PDEntry
from pymatgen.core import Composition, Element

import DataManagement as dm
from HSX import HSX

# reduce verbosity of pulling DFT energies from the MP API
# plt.switch_backend('agg')
plt.rcParams['axes.xmargin'] = 0.005
plt.rcParams['axes.ymargin'] = 0
warnings.filterwarnings("ignore")

X_step = 0.01
X_vals = np.arange(0, 1 + X_step, X_step)
X_logs = np.log(X_vals[1:-1])


def find_local_minima(points):
    # Function to check if a point is a local minimum
    def is_lt_prev(index):
        if index == 0:
            return False
        else:
            return points[index][1] < points[index - 1][1]

    local_minima = []
    current_section = []

    for i in range(len(points)):
        # if lower temp than prev
        if is_lt_prev(i):
            current_section = [points[i]]
        # if current section exists and point is same temp
        elif current_section and current_section[-1][1] == points[i][1]:
            current_section.append(points[i])
        # higher temp and current section exists
        elif current_section:
            local_minima.append(current_section[int(len(current_section) / 2)])
            current_section = []

    return local_minima


def find_local_maxima(points):
    # Function to check if a point is a local minimum
    def is_gt_prev(index):
        if index == 0:
            return False
        else:
            return points[index][1] > points[index - 1][1]

    local_maxima = []
    current_section = []

    for i in range(len(points)):
        # if higher temp than prev
        if is_gt_prev(i):
            current_section = [points[i]]
        # if current section exists and point is same temp
        elif current_section and current_section[-1][1] == points[i][1]:
            current_section.append(points[i])
        # lower temp and current section exists
        elif current_section:
            local_maxima.append(current_section[int(len(current_section) / 2)])
            current_section = []

    return local_maxima


class BinaryLiquid:

    def __init__(self, sys_name, components, init_error=False, **kwargs):
        self.init_error = init_error
        self.sys_name = sys_name
        self.components = components

        self.component_data = kwargs.get('component_data', {})
        self.mpds_json = kwargs.get('mpds_json', {})
        self.mpds_liquidus = kwargs.get('mpds_liquidus', [])
        self.invariants = kwargs.get('invariants', [])

        self.temp_range = kwargs.get('temp_range', [])
        self.liq_temp_span = kwargs.get('liq_temp_span', 0)
        self.comp_range = kwargs.get('comp_range', [0, 100])
        self.ignored_comp_ranges = kwargs.get('ignored_comp_ranges', [])

        self.dft_type = kwargs.get('dft_type', "GGA/GGA+U")
        self.ch = kwargs.get('ch', None)
        self.phases = kwargs.get('phases', [])
        self.params = kwargs.get('params', [0, 0, 0, 0])

        self.guess_symbols = None
        self.constraints = None
        self.opt_path = None
        self.hsx = None

    @classmethod
    def from_cache(cls, system, cache_dir="", dft_type="GGA/GGA+U", mpds_pd_ind=0, params=None, reconstruction=False):

        if isinstance(system, str):
            sys_name = system
            components = system.split('-')
        elif isinstance(system, list):
            components = system
            sys_name = '-'.join(system)
        else:
            print("Error: system must be a hyphenated string or a list")
            return cls("", [], True)

        try:
            [Composition(c) for c in components]
        except ValueError as e:
            print(e)
            return cls(sys_name, [], True)

        if params is None:
            params = [0, 0, 0, 0]

        ch = dm.get_dft_convexhull(components, dft_type)
        phases = []

        # initialize phases from DFT entries on the hull
        for entry in ch.stable_entries:
            try:
                composition = entry.composition.fractional_composition.as_dict()[components[1]]
            except KeyError:
                composition = 0

            # convert eV/atom to J/mol (96,485 J/mol per 1 eV/atom)
            phase = {'name': entry.name, 'comp': composition, 'points': [],
                     'energy': 96485 * ch.get_form_energy_per_atom(entry)}
            phases.append(phase)

        phases.sort(key=lambda x: x['comp'])
        phases.append({'name': 'L', 'points': []})

        # self.components = self.mpds_json['chemical_elements'] compare these to check ordering?
        # self.mpds_json / mpds_liquidus may be undefined here, initialize for specified param values
        mpds_json, component_data, mpds_liquidus = dm.get_MPDS_data(components, pd_ind=mpds_pd_ind)

        if not reconstruction and mpds_liquidus:
            # set component melting temperatures to match the phase diagram
            component_data[components[0]][1] = mpds_liquidus[0][1]
            component_data[components[-1]][1] = mpds_liquidus[-1][1]

        if 'temp' in mpds_json:
            temp_range = [mpds_json['temp'][0] + 273.15, mpds_json['temp'][1] + 273.15]
        else:
            comp_tms = [component_data[comp][1] for comp in components]
            temp_range = [min(comp_tms) - 50, max(comp_tms) * 1.1 + 50]

        if not mpds_liquidus:
            return cls(sys_name, components, True, mpds_json=mpds_json, component_data=component_data,
                       temp_range=temp_range, ch=ch, phases=phases, params=params)

        liq_temp_span = max(mpds_liquidus, key=lambda x: x[1])[1] - min(mpds_liquidus, key=lambda x: x[1])[1]

        return cls(sys_name, components, False, mpds_json=mpds_json, component_data=component_data,
                   mpds_liquidus=mpds_liquidus, temp_range=temp_range, liq_temp_span=liq_temp_span,
                   ch=ch, phases=phases, params=params)

    def L0_a(self):
        return self.params[0]

    def L0_b(self):
        return self.params[1]

    def L1_a(self):
        return self.params[2]

    def L1_b(self):
        return self.params[3]

    def export_phases_points(self, output="dict"):
        data = {'X': list(X_vals), 'S': [], 'H': [], 'Phase Name': ['L' for _ in X_vals]}

        H_a_liq = self.component_data[self.components[0]][0]
        H_b_liq = self.component_data[self.components[1]][0]
        H_lc = (H_a_liq * X_vals[-2:0:-1] +
                H_b_liq * X_vals[1:-1])
        H_xs = X_vals[1:-1] * X_vals[-2:0:-1] * (self.L0_a() + self.L1_a() * (1 - 2 * X_vals[1:-1]))
        data['H'] = list(H_lc + H_xs)
        data['H'].insert(0, H_a_liq)
        data['H'].append(H_b_liq)

        R = 8.314

        S_a_liq = self.component_data[self.components[0]][0] / self.component_data[self.components[0]][1]
        S_b_liq = self.component_data[self.components[1]][0] / self.component_data[self.components[1]][1]
        S_lc = (S_a_liq * X_vals[-2:0:-1] +
                S_b_liq * X_vals[1:-1])
        S_ideal = -R * (X_vals[1:-1] * X_logs + X_vals[-2:0:-1] * X_logs[::-1])
        S_xs = -X_vals[1:-1] * X_vals[-2:0:-1] * (self.L0_b() + self.L1_b() * (1 - 2 * X_vals[1:-1]))
        data['S'] = list(S_lc + S_ideal + S_xs)
        data['S'].insert(0, S_a_liq)
        data['S'].append(S_b_liq)

        for x in X_vals:
            for phase in self.phases:
                if phase['name'] == 'L':
                    continue
                if round(phase['comp'], 2) == round(x, 2):
                    data['X'].append(round(x, 2))
                    data['H'].append(phase['energy'])
                    data['S'].append(0)
                    data['Phase Name'].append(phase['name'])

        if output == "dict":
            return data
        if output == "dataframe":
            return pd.DataFrame(data)

    def update_phase_points(self):
        """
        This function uses the HSX class to calculate the phase points for given parameter values.
        Formerly called 'convex_hull', this converts phase data into the HSX form and uses Abrar's HSX code to
        calculate the liquidus and intermetallic phase decompositions.
        :return: None
        """
        data = self.export_phases_points()
        hsx_dict = {'data': data, 'phases': [phase['name'] for phase in self.phases], 'comps': self.components}
        self.hsx = HSX(hsx_dict, [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15])

        phase_points = self.hsx.get_phase_points()
        for phase in self.phases:
            phase['points'] = phase_points[phase['name']]

    def find_invariant_points(self, verbose=False, t_tol=15):
        """
        This function uses the MPDS json and MPDS liquidus to identify invariant points in the MPDS data.
        This does not take into account DFT phases, which may differ in composition from the phases in the MPDS data.
        To use this, there must be both valid liquid and json data for a binary system. If there is a valid json but
        no liquidus data, use Data_Management.identify_MPDS_phases() instead
        :return: List of invariant points
        """
        if self.mpds_json['reference'] is None:
            print("system JSON does not contain any data!\n")
            return []

        phases = dm.identify_MPDS_phases(self.mpds_json, verbose=True)
        invariants = [phase for phase in phases if phase['type'] == 'mig']  # miscibility gaps are invariants not phases

        mpds_lowt_phases = [phase for phase in phases
                            if (phase['type'] in ['lc', 'ss']  # filter phases not stable at low temperatures
                                and (phase['tbounds'][0][1] < (self.mpds_json['temp'][0] + 273.15) +
                                     (self.mpds_json['temp'][1] - self.mpds_json['temp'][0]) * 0.10) or '(' in phase[
                                    'name'])]

        if verbose:
            print('--- low temperature MPDS phases & component solid solutions ---')
            for phase in mpds_lowt_phases:
                print(phase)

        phase_labels = [label[0] for label in self.mpds_json['labels']]
        ss_label = "(" + self.components[0] + ", " + self.components[1] + ")"
        ss_label_inv = "(" + self.components[1] + ", " + self.components[0] + ")"
        ss_labels = [ss_label, ss_label + ' ht', ss_label + ' rt',
                     ss_label_inv, ss_label_inv + ' ht', ss_label_inv + ' rt']
        full_comp_ss = bool([label for label in phase_labels if label in ss_labels])

        if not full_comp_ss:
            # locate local maxima and minima in liquidus
            maxima = find_local_maxima(self.mpds_liquidus)
            minima = find_local_minima(self.mpds_liquidus)

            # assign line compounds that overlap with liquidus maxima points as congruent melting points
            if mpds_lowt_phases:
                for coords in maxima[:]:
                    mpds_lowt_phases.sort(key=lambda x: abs(x['comp'] - coords[0]))
                    phase = mpds_lowt_phases[0]
                    if (phase['type'] in ['lc', 'ss'] and abs(phase['comp'] - coords[0] <= 0.02)
                            and phase['tbounds'][1][1] + t_tol >= coords[1]):
                        phase['type'] = 'cmp'
                        invariants.append({'type': phase['type'], 'comp': phase['comp'], 'temp': phase['tbounds'][1][1],
                                           'phases': [phase['name']], 'phase_comps': [phase['comp']]})
                        maxima.remove(coords)

            # sort by descending temperature for peritectic identification algorithm
            mpds_lowt_phases.sort(key=lambda x: x['tbounds'][1][1], reverse=True)

            def find_adj_phases(point):
                sorted_phases = (mpds_lowt_phases +
                                 [{'name': self.components[0], 'comp': 0, 'type': 'lc',
                                   'tbounds': [[], [0, self.component_data[self.components[0]][1]]]},
                                  {'name': self.components[1], 'comp': 1, 'type': 'lc',
                                   'tbounds': [[], [1, self.component_data[self.components[1]][1]]]},
                                  ])
                sorted_phases = [p for p in sorted_phases if p['tbounds'][1][1] + t_tol >= point[1]]
                sorted_phases.sort(key=lambda x: abs(x['comp'] - point[0]))
                nearest = sorted_phases.pop(0)
                if nearest['comp'] > point[0]:
                    if not sorted_phases:
                        return None, nearest
                    opposite = sorted_phases.pop(0)
                    while opposite['comp'] > point[0]:
                        if sorted_phases:
                            opposite = sorted_phases.pop(0)
                        else:
                            return None, nearest
                    return opposite, nearest
                if nearest['comp'] < point[0]:
                    if not sorted_phases:
                        return nearest, None
                    opposite = sorted_phases.pop(0)
                    while opposite['comp'] < point[0]:
                        if sorted_phases:
                            opposite = sorted_phases.pop(0)
                        else:
                            return nearest, None
                    return nearest, opposite

            misc_gap_labels = []
            for label in self.mpds_json['labels']:
                delim_label = label[0].split(' ')
                if len(delim_label) == 3 and delim_label[0][0] == 'L' and delim_label[2][0] == 'L':
                    misc_gap_labels.append([label[1][0] / 100.0, label[1][1] + 273.15])

            # for each label, check local maxima to see if there is a nearby inflection in the liquidus that can be used
            for mgl in misc_gap_labels:
                if len(maxima) < 1:
                    break
                nearest_maxima = min(maxima, key=lambda x: abs(x[0] - mgl[0]))

                tbounds = [None, nearest_maxima]
                cbounds = None
                phases = None
                phase_comps = None

                # look for a svgpath shape to find misc gap boundaries
                for shape in self.mpds_json['shapes']:
                    if not shape['nphases'] == 2:
                        continue
                    # split svgpath into tags, ordered pairs
                    data = dm.shape_to_list(shape['svgpath'])
                    if not data:
                        continue

                    # sort by ascending T value
                    data.sort(key=lambda x: x[1])
                    if not (abs(data[-1][1] - nearest_maxima[1]) < t_tol and abs(
                            data[-1][0] - nearest_maxima[0]) < 0.05):
                        continue
                    tbounds = [data[0], data[-1]]

                    # sort by ascending X value
                    data.sort(key=lambda x: x[0])
                    if not data[0][0] < nearest_maxima[0] < data[-1][0]:
                        continue
                    cbounds = [data[0], data[-1]]
                    break

                if len(minima) >= 1:
                    # adj eutectic should be minimum distance in x-t space from the miscibility gap maximum
                    # temp is weighted double to resolve some issues
                    adj_eut = min(minima,
                                  key=lambda x: abs(tbounds[1][0] - x[0]) +
                                                2 * (abs(tbounds[1][1] - x[1]) / self.temp_range[1]))
                    tbounds[0] = [tbounds[1][0], adj_eut[1]]
                    adj_phases = find_adj_phases(adj_eut)

                    if adj_eut[0] < tbounds[1][0]:  # eutectic on left side of misc gap
                        if adj_phases[0] is not None:
                            phase_comps = [adj_phases[0]['comp']]
                            phases = [adj_phases[0]['name']]

                        if not cbounds:
                            lhs_ind = self.mpds_liquidus.index(adj_eut)
                            for i in range(lhs_ind + 1, len(self.mpds_liquidus) - 1):
                                if self.mpds_liquidus[i + 1][1] < adj_eut[1] <= self.mpds_liquidus[i][1]:
                                    m = ((self.mpds_liquidus[i + 1][1] - self.mpds_liquidus[i][1]) /
                                         (self.mpds_liquidus[i + 1][0] - self.mpds_liquidus[i][0]))
                                    rhs_comp = (adj_eut[1] - self.mpds_liquidus[i][1]) / m + self.mpds_liquidus[i][0]
                                    cbounds = [adj_eut, [rhs_comp, adj_eut[1]]]
                                    break

                    elif adj_eut[0] > tbounds[1][0]:  # eutectic on right side of misc gap
                        if adj_phases[1] is not None:
                            phase_comps = [adj_phases[1]['comp']]
                            phases = [adj_phases[1]['name']]

                        if not cbounds:
                            rhs_ind = self.mpds_liquidus.index(adj_eut)
                            for i in reversed(range(1, rhs_ind - 1)):
                                if self.mpds_liquidus[i - 1][1] < adj_eut[1] <= self.mpds_liquidus[i][1]:
                                    m = ((self.mpds_liquidus[i - 1][1] - self.mpds_liquidus[i][1]) /
                                         (self.mpds_liquidus[i - 1][0] - self.mpds_liquidus[i][0]))
                                    lhs_comp = (adj_eut[1] - self.mpds_liquidus[i][1]) / m + self.mpds_liquidus[i][0]
                                    cbounds = [[lhs_comp, adj_eut[1]], adj_eut]
                                    break

                    if cbounds[0][1] != cbounds[1][1]:
                        cbounds[0][1] = adj_eut[1]
                        cbounds[1][1] = adj_eut[1]
                    minima.remove(adj_eut)

                if cbounds:
                    invariants.append({'type': 'mig', 'comp': tbounds[1][0], 'cbounds': cbounds, 'tbounds': tbounds,
                                       'phases': phases, 'phase_comps': phase_comps})
                    maxima.remove(nearest_maxima)
                break

            stable_phase_comps = []
            # main loop for peritectic phase identification
            for phase in mpds_lowt_phases:
                if '(' in phase['name']:  # ignore component SS phases
                    continue
                # congruent melting points will not be considered for peritectic formation but will limit others
                if phase['type'] == 'cmp':
                    stable_phase_comps.append(phase['comp'])
                    continue

                sections = []
                current_section = []
                for i in range(len(self.mpds_liquidus) - 1):
                    # liquidus point is above or equal to phase temp
                    if self.mpds_liquidus[i][1] >= phase['tbounds'][1][1]:
                        current_section.append(self.mpds_liquidus[i])
                        if self.mpds_liquidus[i + 1][1] >= phase['tbounds'][1][1] \
                                and i + 1 == len(self.mpds_liquidus) - 1:
                            current_section.append(self.mpds_liquidus[i + 1])
                            sections.append(current_section)
                    # liquidus point is first point below phase temp
                    elif current_section:
                        # add to section if closer to phase temp than last point above temp:
                        if (abs(phase['tbounds'][1][1] - current_section[-1][1]) >
                                abs(phase['tbounds'][1][1] - self.mpds_liquidus[i][1])):
                            current_section.append(self.mpds_liquidus[i])
                        # end section
                        sections.append(current_section)
                        current_section = []
                    # next liquidus point is above temp
                    elif self.mpds_liquidus[i + 1][1] >= phase['tbounds'][1][1] > self.mpds_liquidus[i][1]:
                        # add to section if current point below phase temp is closer than next point above
                        if (abs(phase['tbounds'][1][1] - self.mpds_liquidus[i + 1][1]) >
                                abs(phase['tbounds'][1][1] - self.mpds_liquidus[i][1])):
                            current_section.append(self.mpds_liquidus[i])

                # find endpoints of liquidus segments excluding the component ends
                endpoints = []
                for section in sections:
                    if section[0] != self.mpds_liquidus[0]:
                        endpoints.append(section[0])
                    if section[-1] != self.mpds_liquidus[-1]:
                        endpoints.append(section[-1])

                for comp in stable_phase_comps:
                    # filter out endpoints if there exists a stable phase between the current phase and the liquidus
                    endpoints = [ep for ep in endpoints
                                 if abs(comp - ep[0]) > abs(phase['comp'] - ep[0]) or
                                 abs(comp - phase['comp']) > abs(phase['comp'] - ep[0])]

                # sort by increasing distance to liquidus to find the shortest distance
                endpoints.sort(key=lambda x: abs(x[0] - phase['comp']))

                # take the closest liquidus point to the phase as the peritectic point
                if endpoints:
                    invariants.append({'type': 'per', 'comp': endpoints[0][0], 'temp': phase['tbounds'][1][1],
                                       'phases': [phase['name']], 'phase_comps': [phase['comp']]})
                stable_phase_comps.append(phase['comp'])

            # add eutectic points
            for coords in minima:
                adj_phases = find_adj_phases(coords)
                phases = []
                phase_comps = []
                for phase in adj_phases:
                    if phase is None:
                        phases.append(None)
                        phase_comps.append(None)
                    else:
                        phases.append(phase['name'])
                        phase_comps.append(phase['comp'])
                invariants.append({'type': 'eut', 'comp': coords[0], 'temp': coords[1],
                                   'phases': phases, 'phase_comps': phase_comps})

        else:  # aze
            minima = find_local_minima(self.mpds_liquidus)
            if len(minima) > 1:
                print("too many azeotrope points identified")
                self.init_error = True
            elif len(minima) == 1:
                invariants.append({'type': 'aze', 'comp': minima[0][0], 'temp': minima[0][1]})
            # process solidus here
            print('solidus processing not implemented')

        invariants.sort(key=lambda x: x['comp'])
        invariants = [inv for inv in invariants if inv['type'] not in ['lc', 'ss']]
        if verbose:
            print('--- identified invariant points ---')
            for inv in invariants:
                print(inv)
            print()
        return invariants

    def solve_params_from_constraints(self, guessed_vals: dict):
        symbols = sp.symbols('a b c d')
        for ind, symbol in enumerate(symbols):
            try:
                if symbol in guessed_vals:
                    self.params[ind] = guessed_vals[symbol]
                elif self.constraints:
                    self.params[ind] = self.constraints[symbol].subs(guessed_vals)
            except TypeError as e:
                raise RuntimeError("error in constraint equations!")

    def liquidus_is_continuous(self, tol=2 * X_step):
        last_coords = None
        for coords in self.phases[-1]['points']:
            if last_coords:
                if coords[0] - last_coords[0] > tol:
                    return False
                # if coords[1] - last_coords[1] > self.liq_temp_span / 2:
                #     return False
            last_coords = coords
        return True

    def calculate_deviation_metrics(self, ignored_ranges=True, num_points=30):
        x1, T1 = zip(*self.mpds_liquidus)
        x2, T2 = zip(*self.phases[-1]['points'])

        def is_within_boundaries(value):
            """Check if a value falls within any of the given boundary ranges."""
            for lower, upper in self.ignored_comp_ranges:
                if lower <= value <= upper:
                    return True
            return False

        # compare (up to) num_points points evenly spaced across composition space, excluding the endpoints
        x_coords = np.linspace(self.comp_range[0] / 100, self.comp_range[-1] / 100, num_points + 2)[1:-1]

        if ignored_ranges:
            x_coords = [x for x in x_coords if not is_within_boundaries(x)]

        if len(x_coords) < 10:
            print(f"Warning: large compostion range filtered out (num_points = {len(x_coords)}")
            return float('inf'), float('inf')

        Y1 = []
        Y2 = []

        for i in range(len(x_coords)):
            MPDS_ind = fit_ind = -1

            for j in range(len(self.mpds_liquidus) - 1):
                if x1[j] <= x_coords[i] < x1[j + 1]:
                    # print(x1[j], x_coords[i], x1[j + 1], j)
                    MPDS_ind = j
                    break
            for j in range(len(self.phases[-1]['points']) - 1):
                if x2[j] <= x_coords[i] < x2[j + 1]:
                    # print(x2[j], x_coords[i], x2[j + 1], j)
                    fit_ind = j
                    break

            # print("i = ", i, "x[i] = ", x_coords[i], "MPDS_ind = ", MPDS_ind, "fit_ind = ", fit_ind)
            if MPDS_ind != -1 and fit_ind != -1:
                m1 = (T1[MPDS_ind] - T1[MPDS_ind + 1]) / (x1[MPDS_ind] - x1[MPDS_ind + 1])
                b1 = (x1[MPDS_ind] * T1[MPDS_ind + 1] - x1[MPDS_ind + 1] * T1[MPDS_ind]) / (
                        x1[MPDS_ind] - x1[MPDS_ind + 1])
                y1 = m1 * x_coords[i] + b1
                m2 = (T2[fit_ind] - T2[fit_ind + 1]) / (x2[fit_ind] - x2[fit_ind + 1])
                b2 = (x2[fit_ind] * T2[fit_ind + 1] - x2[fit_ind + 1] * T2[fit_ind]) / (
                        x2[fit_ind] - x2[fit_ind + 1])
                y2 = m2 * x_coords[i] + b2
                Y1.append(y1)
                Y2.append(y2)

        # find absolute difference at each point
        point_diffs = [abs(Y1[i] - Y2[i]) for i in range(len(Y2))]
        squared_point_diffs = [(Y1[i] - Y2[i]) ** 2 for i in range(len(Y2))]

        return np.mean(point_diffs), math.sqrt(np.mean(squared_point_diffs))

    def f(self, point):
        guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, point)}
        self.solve_params_from_constraints(guess_dict)  # update parameter values
        try:
            self.update_phase_points()
        except (ValueError, TypeError) as e:
            print(e)
            return float('inf')
        if not self.liquidus_is_continuous():
            print(f'liquidus not continuous for guess {guess_dict}')
            return float('inf')
        mae, rmse = self.calculate_deviation_metrics()
        return mae

    def nelder_mead(self, max_iter=128, tol=5e-2, verbose=False):
        """
        Nelder-Mead algorithm for fitting the liquid non-ideal mixing parameters.

        Args:
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence.
            verbose: Determines if updates are printed to the terminal

        Returns:
            mae: The mean average error (MAE) from the fitted liquidus to the MPDS liquidus.
            rmse: The standard deviation from the fitted liquidus to the MPDS liquidus.
            opt_path: An [3 x (num_dims + 1) x num_iterations] Numpy Array of the optimization path
        """
        n = len(self.guess_symbols)
        # x0 init dependent on type / quanitity of self.guess_symbols
        # n = x0.shape[1]  # Number of dimensions
        self.opt_path = np.empty((3, n + 1, max_iter), dtype=float)
        initial_time = time.time()

        print("--- begin nelder mead optimization ---")

        if n == 1:
            # Line search for 1D case
            raise NotImplementedError

        elif 1 < n <= 4:
            # Nelder-Mead for higher dimensions
            if n == 2:
                x0 = np.array([[-20, -20], [-20, 20], [20, -20]], dtype=float)
            elif n == 3:
                x0 = np.array([[-20, self.L1_a(), -20], [-20, self.L1_a(), 20], [20, self.L1_a(), -20]], dtype=float)

            for i in range(max_iter):
                start_time = time.time()
                if verbose:
                    print("iteration #", i)

                f_vals = np.array([self.f(x) for x in x0])
                self.opt_path[:, :n, i] = x0
                self.opt_path[:, n:, i] = np.array([[f] for f in f_vals])
                iworst = np.argmax(f_vals)
                ibest = np.argmin(f_vals)
                centroid = np.mean(x0[f_vals != f_vals[iworst]], axis=0)
                xreflect = centroid + 1.0 * (centroid - x0[iworst, :])
                f_xreflect = self.f(xreflect)

                if iworst == ibest:
                    self.opt_path = self.opt_path[:, :, :i]
                    raise RuntimeError("Nelder-Mead algorithm is unable to find physical parameter values.")

                # Reflection
                if f_vals[iworst] <= f_xreflect < f_vals[n]:
                    x0[iworst, :] = xreflect
                # Expansion
                elif f_xreflect < f_vals[ibest]:
                    xexp = centroid + 2.0 * (xreflect - centroid)
                    if self.f(xexp) < f_xreflect:
                        x0[iworst, :] = xexp
                    else:
                        x0[iworst, :] = xreflect
                # Contraction
                else:
                    if f_xreflect < f_vals[n]:
                        xcontract = centroid + 0.5 * (xreflect - centroid)
                        if self.f(xcontract) < self.f(x0[iworst, :]):
                            x0[iworst, :] = xcontract
                        else:  # Shrink Step
                            x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                            [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                            x0[iworst, :] = x0[imid, :] + 0.5 * (x0[imid, :] - x0[ibest, :])
                    else:
                        xcontract = centroid + 0.5 * (x0[iworst, :] - centroid)
                        if self.f(xcontract) < self.f(x0[iworst, :]):
                            x0[iworst, :] = xcontract
                        else:  # Shrink Step
                            x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                            [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                            x0[imid, :] = x0[ibest, :] + 0.5 * (x0[imid, :] - x0[ibest, :])

                if verbose:
                    print("best guess:", x0[ibest, :], f_vals[ibest])
                    print("1/2 height of triangle = ", np.max(np.abs(x0 - centroid)))
                    print("--- %s seconds ---" % (time.time() - start_time))

                # Check convergence
                if np.max(np.abs(x0 - centroid)) < tol:
                    self.f(x0[ibest, :])
                    print("--- total time %s seconds ---" % (time.time() - initial_time))
                    mae, rmse = self.calculate_deviation_metrics()
                    print("mean temperature deviation per point between curves =", mae, '\n')
                    self.opt_path = self.opt_path[:, :, :i]
                    return mae, rmse, self.opt_path
            raise RuntimeError("Nelder-Mead algorithm did not converge within limit.")
        else:
            raise ValueError("Nelder-Mead algorithm is not implemented for dimensions > 4.")

    def fit_parameters(self, verbose=True, n_opts=1, t_tol=15):
        """Fit the liquidus non-ideal mixing parameters for a binary system. This function utilizes the nelder-mead
         algorithm to minimize the temperature deviation in the liquidus"""

        if self.mpds_liquidus is None:
            print("system missing liquidus data!\n")
            return

        # find invariant points
        if not self.invariants:
            self.invariants = self.find_invariant_points(verbose=verbose, t_tol=t_tol)

        # L0_a = a, L0_b = b, L1_a = c, L1_b = d, x = xb
        xb, t, a, b, c, d = sp.symbols('x t a b c d')
        L0 = a + b * t
        L1 = c + d * t
        xa = 1 - xb

        R = 8.314
        Ga = (self.component_data[self.components[0]][0] -
              t * self.component_data[self.components[0]][0] / self.component_data[self.components[0]][1])
        Gb = (self.component_data[self.components[1]][0] -
              t * self.component_data[self.components[1]][0] / self.component_data[self.components[1]][1])

        G_ideal = R * t * (xa * sp.log(xa) + xb * sp.log(xb))
        G_xs = L0 * (xa * xb) + L1 * (xa * xb * (xa - xb))
        G_liq = Ga * xa + Gb * xb + G_ideal + G_xs

        G_prime = sp.diff(G_liq, xb)
        G_double_prime = sp.diff(G_prime, xb)

        def find_nearest_phase(composition, tol=0.02):
            # sort phases by ascending distance from target composition, excluding the liquid phase
            sorted_phases = sorted(self.phases[:-1], key=lambda x: abs(x['comp'] - composition))
            nearest = sorted_phases[0]
            deviation = abs(nearest['comp'] - composition)
            if deviation > tol:
                return {}, deviation
            return nearest, deviation

        eqs = []

        # compare invariant points to self.phases to assess solving conditions
        for inv in self.invariants:
            if inv['type'] == 'mig':
                # x0 = 1  # component composition nearest to dome
                # mig_phase, dev = find_nearest_phase(inv['phase_comp'])

                # x0, g0 = inv['phase_comp'], mig_phase['energy']  # nearest phase (compound or component)
                x1, t1 = inv['cbounds'][0]  # bottom left of dome
                x2, t2 = inv['tbounds'][1]  # top of dome
                x3, t3 = inv['cbounds'][1]  # bottom right of dome

                eqn1 = sp.Eq(G_double_prime.subs({xb: x2, t: t2}), 0)
                # eqn2 = sp.Eq(G_liq.subs({xb: x1, t: t1}) + G_prime.subs({xb: x1, t: t1}) * (x0 - x1), g0)
                # eqn3 = sp.Eq(G_liq.subs({xb: x3, t: t3}) + G_prime.subs({xb: x3, t: t3}) * (x0 - x3), g0)
                eqn4 = sp.Eq(G_prime.subs({xb: x1, t: t1}), G_prime.subs({xb: x3, t: t3}))

                eqs.append(['mig', f'{round(x2, 2)} - 2nd order', t1, eqn1])
                eqs.append(['mig', f'{round(x1, 2)}-{round(x3, 2)} - 1st order', t1, eqn4])
                # eqs.append(['mig', f'{round(x1, 2)}-{round(x3, 2)} - Oth order lhs', t1, eqn2])
                # eqs.append(['mig', f'{round(x1, 2)}-{round(x3, 2)} - Oth order rhs', t1, eqn3])

            if inv['type'] == 'cmp':

                if '(' in inv['phases'][0]:  # if invariant phase is a component solid solution phase
                    if inv['comp'] < 0.5:
                        self.ignored_comp_ranges.append([0, inv['comp']])  # if left of peritectic
                    elif inv['comp'] > 0.5:
                        self.ignored_comp_ranges.append([inv['comp'], 1])  # if right of peritectic
                    continue

                # check to see if there is a matching phase on the DFT convex hull
                nearest_phase, dev = find_nearest_phase(inv['comp'])
                if not nearest_phase:
                    continue

                x1, t1 = nearest_phase['comp'], inv['temp']
                eqn = sp.Eq(G_liq.subs({xb: x1, t: t1}), nearest_phase['energy'])
                eqs.append(['cmp', f'{round(x1, 2)} - 0th order', t1, eqn])

            if inv['type'] == 'per':

                if '(' in inv['phases'][0]:  # if invariant phase is a component solid solution phase
                    if inv['phase_comps'][0] < inv['comp']:
                        self.ignored_comp_ranges.append([0, inv['comp']])  # if left of peritectic
                    elif inv['phase_comps'][0] > inv['comp']:
                        self.ignored_comp_ranges.append([inv['comp'], 1])  # if right of peritectic
                    continue

                # check to see if there is a matching phase on the DFT convex hull
                per_phase, dev = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                if not per_phase:
                    continue

                x1, t1 = inv['comp'], inv['temp']
                x2, g2 = per_phase['comp'], per_phase['energy']

                eqn1 = sp.Eq(G_liq.subs({xb: x1, t: t1}) + G_prime.subs({xb: x1, t: t1}) * (x2 - x1), g2)
                eqn2 = sp.Eq(G_liq.subs({xb: x1, t: t1}), g2)

                liq_point_at_phase = min(self.mpds_liquidus, key=lambda x: abs(x[0] - x2))
                temp_below_liq = liq_point_at_phase[1] - t1

                if temp_below_liq > t_tol:
                    eqs.append(['per', f'{round(x1, 2)} - 0th order', t1, eqn1])
                else:
                    eqs.append(['per', f'{round(x1, 2)} - pseudo CMP 0th order', t1, eqn2])

            if inv['type'] == 'eut':

                if None in inv['phase_comps']:
                    continue

                lhs_phase, dev = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                rhs_phase, dev = find_nearest_phase(inv['phase_comps'][1], tol=0.04)

                invalid_eut = False
                if not lhs_phase or lhs_phase['comp'] > inv['comp']:  # if no DFT phase or phase on wrong side of eut
                    self.ignored_comp_ranges.append([inv['phase_comps'][0], inv['comp']])
                    invalid_eut = True
                elif '(' in inv['phases'][0]:  # if left invariant phase is a component solid solution phase
                    self.ignored_comp_ranges.append([0, inv['comp']])
                    invalid_eut = True
                if not rhs_phase or rhs_phase['comp'] < inv['comp']:  # if no DFT phase or phase on wrong side of eut
                    self.ignored_comp_ranges.append([inv['comp'], inv['phase_comps'][1]])
                    invalid_eut = True
                elif '(' in inv['phases'][1]:  # if right invariant phase is a component solid solution phase
                    self.ignored_comp_ranges.append([inv['comp'], 1])
                    invalid_eut = True
                if invalid_eut:
                    continue

                x1, g1 = lhs_phase['comp'], lhs_phase['energy']
                x2, t2 = inv['comp'], inv['temp']
                x3, g3 = rhs_phase['comp'], rhs_phase['energy']

                eqn1 = sp.Eq(G_prime.subs({xb: x2, t: t2}), (g3 - g1) / (x3 - x1))  # slope only
                eqn2 = sp.Eq(G_liq.subs({xb: x2, t: t2}) + G_prime.subs({xb: x2, t: t2}) * (x1 - x2), g1)  # lhs + slope
                eqn3 = sp.Eq(G_liq.subs({xb: x2, t: t2}) + G_prime.subs({xb: x2, t: t2}) * (x3 - x2), g3)  # rhs + slope

                eqs.append(['eut', f'{round(x2, 2)} - 1st order', t2, eqn1])
                if g1 <= g3:
                    eqs.append(['eut', f'{round(x2, 2)} - Oth order lhs', t2, eqn2])
                else:
                    eqs.append(['eut', f'{round(x2, 2)} - Oth order rhs', t2, eqn3])

        max_liq_temp = max(self.mpds_liquidus, key=lambda x: x[1])[1]
        mean_liq_temp = (min(self.mpds_liquidus, key=lambda x: x[1])[1] + max_liq_temp) / 2
        eqs = [eq for eq in eqs if not eq[3] == False]
        # print(self.ignored_comp_ranges)
        # self.ignored_comp_ranges = [[0.32, 0.41], [0.50, 0.80]]
        initial_constrs = []
        # if len(eqs) >= 2:  # Perfectly constrained or overconstrained system: make a guess for which constraints work
        #     self.guess_symbols = [b, d]
        #     highest_tm_eq = max(eqs, key=lambda x: x[2])  # assume the highest tm invariant point is always included
        #     for eq in eqs:
        #         if eq != highest_tm_eq:
        #             self.constraints = sp.solve([eq[3], highest_tm_eq[3]], (a, c))
        #             try:
        #                 init_mae = self.f([0, 0])
        #                 if init_mae != float('inf'):
        #                     initial_constrs.append([eq, highest_tm_eq, init_mae])
        #             except RuntimeError:
        #                 continue
        #     initial_constrs.sort(key=lambda x: x[2])  # sort by initial MAE without temp-dependent param optimization
        if not initial_constrs or len(eqs) < 2:  # Underconstrained system: gen 2 fake constraints for L_b param fitting
            print("Underconstrained system detected! Fitting L_a parameters first")

            self.params = [0, 0, 0, 0]
            self.guess_symbols = [a, c]
            self.constraints = None
            try:
                self.nelder_mead(tol=10, verbose=verbose)
            except RuntimeError:
                return []

            eqn1 = sp.Eq(L0.subs({t: mean_liq_temp}),
                         self.L0_a())  # pseudo-constraint 1: L0 = L0a + L0b * mean_temp
            eqn2 = sp.Eq(L1.subs({t: mean_liq_temp}),
                         self.L1_a())  # pseudo-constraint 2: L1 = L1a + L1b * mean_temp
            initial_constrs = [[['pseudo', f'L0_a + L0_b * t = {round(self.L0_a(), 2)}', mean_liq_temp, eqn1],
                                ['pseudo', f'L1_a + L1_b * t = {round(self.L1_a(), 2)}', mean_liq_temp, eqn2], []]]
            self.guess_symbols = [b, d]

        # perform nelder-mead optimization and store results for the first n_opts constraint combos
        fitting_data = []
        for _ in range(n_opts):
            if not initial_constrs:
                break
            init_constr = initial_constrs.pop(0)[:-1]
            if verbose:
                print("--- initial constraints ---")
                for (eq_type, score, temp, eq) in init_constr:
                    print(f"invariant: {eq_type}, type: {score}, temperature: {round(temp, 1)}, equation: {eq}")
                print(f"maximum composition range fitted: {self.comp_range}")
                print(f"ignored composition ranges: {self.ignored_comp_ranges}")

            selected_eqs = [eq[3] for eq in init_constr]
            self.constraints = sp.solve(selected_eqs, (a, c))
            guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, [0, 0])}
            self.solve_params_from_constraints(guess_dict)
            try:
                mae, rmse, path = self.nelder_mead(verbose=verbose, tol=5E-1)
            except RuntimeError:
                continue
            norm_mae = mae / max_liq_temp
            norm_rmse = rmse / max_liq_temp
            l0 = self.L0_a() + mean_liq_temp * self.L0_b()
            l1 = self.L1_a() + mean_liq_temp * self.L1_b()
            fit_invs = self.hsx.liquidus_invariants()[0]
            fitting_data.append({'mae': mae, 'rmse': rmse, 'norm_mae': norm_mae, 'norm_rmse': norm_rmse, 'nmpath': path,
                                 'L0_a': self.L0_a(), 'L0_b': self.L0_b(), 'L1_a': self.L1_a(), 'L1_b': self.L1_b(),
                                 'L0': l0, 'L1': l1, 'euts': fit_invs['Eutectics'], 'pers': fit_invs['Peritectics'],
                                 'cmps': fit_invs['Congruent Melting'], 'migs': fit_invs['Misc Gaps']})

        # update the BinaryLiquid object points and parameters with the best fit produced by nelder-mead
        if fitting_data:
            best_fit = min(fitting_data, key=lambda x: x['mae'])
            self.params = [best_fit['L0_a'], best_fit['L0_b'], best_fit['L1_a'], best_fit['L1_b']]
            self.opt_path = best_fit['nmpath']
            self.update_phase_points()
        return fitting_data


class BLPlotter:
    """
    A plotting class for BinaryLiquid objects

    This class contains the functions used to create all subfigures in the interactive matrix
    """

    def __init__(self, binaryliquid: BinaryLiquid, **plotkwargs):
        """
        :param binaryliquid (BinaryLiquid): BinaryLiquid object containing data to generate plots from
        :param plotkwargs (dict): Keyword args passed to matplotlib.pyplot.plot
        """

        self._bl = binaryliquid
        self.plotkwargs = plotkwargs or {
            'axes': {'xmargin': 0.005, 'ymargin': 0}
        }

    def get_plot(self, plot_type: str, **kwargs) -> go.Figure | plt.Axes:
        if plot_type not in ['xt',
                             'pc',
                             'ch', 'ch+g', 'vch', 'vch+g',
                             'fit', 'fit+liq', 'pred', 'pred+fit', 'pred+liq', 'pred+fit+liq',
                             'ftable', 'ptable',
                             'nmp', 'paramconf',
                             'gliq']:
            raise ValueError(f"Invalid plot type '{plot_type}'")

        # # some fine-tuning of the plyplots so they look nicer
        # plt.rcParams['axes.xmargin'] = 0.005
        # plt.rcParams['axes.ymargin'] = 0

        fig = None

        if plot_type == 'xt':

            raise NotImplementedError()
            # fig, _ = plot_phase_diagram(self._bl.mpds_json)

        elif plot_type == 'pc':

            ([mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp],
             [mp_phases, mp_phases_ebelow, min_form_e]) = dm.get_low_temp_phase_data(self._bl.mpds_json, self._bl.ch)

            mpds_congruent_phases = {key: value for key, value in mpds_congruent_phases.items() if '(' not in key}
            mpds_incongruent_phases = {key: value for key, value in mpds_incongruent_phases.items() if '(' not in key}

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2), gridspec_kw={'hspace': 0})

            def plot_phases(ax, source, color, alpha=0.5):
                for phase, ((lb, ub), mag) in source.items():
                    # enforce a minimum width so there is space for a label
                    if ub - lb < 0.026:
                        ave = (ub + lb) / 2
                        lb = ave - 0.013
                        ub = ave + 0.013
                    ax.fill_betweenx([min(0, mag), max(0, mag)], lb, ub, color=color, alpha=alpha)
                    ax.set_xlim(0, 1)
                    ax.margins(x=0, y=0)

            plot_phases(ax1, mpds_congruent_phases, 'blue')
            plot_phases(ax1, mpds_incongruent_phases, 'purple')
            plot_phases(ax2, mp_phases, 'orange')
            plot_phases(ax2, mp_phases_ebelow, 'red')

            mpds_phases = bool(mpds_congruent_phases or mpds_incongruent_phases)

            if mpds_phases:
                tick_range = np.linspace(0, max_phase_temp, 4)[1:]
                ax1.set_yticks(tick_range)
                ax1.set_yticklabels([format(tick, '.1e') for tick in tick_range])
                ax1.set_ylim(0, 1.1 * max_phase_temp)
            else:
                ax1.set_yticks([])

            ax1.set_ylabel('MPDS', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
            ax1.yaxis.set_label_position('right')
            ax1.set_xticks([])

            if mp_phases:
                tick_range = np.linspace(0, min_form_e, 4)
                ax2.set_yticks(tick_range)
                ax2.set_yticklabels([format(tick, '.1e') for tick in tick_range])
                ax2.set_ylim(1.1 * min_form_e, 0)
            elif mpds_phases:
                ax2.set_yticks([0])
                ax2.set_yticklabels([format(0, '.1e')])
                ax2.set_ylim(-1, 0)
            else:
                ax2.set_yticks([])

            ax2.set_ylabel('MP', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
            ax2.yaxis.set_label_position('right')
            ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
            ax2.set_xticklabels([0, 20, 40, 60, 80, 100])

            fig.suptitle('Low Temperature Phase Comparison', fontweight='semibold')

        elif plot_type in ['ch', 'ch+g', 'vch', 'vch+g']:

            if plot_type in ['vch', 'vch+g']:
                ch, struct_props = dm.get_dft_convexhull(self._bl.sys_name, self._bl.dft_type, inc_structure_data=True)
                for entry in ch.stable_entries:
                    struct_props[entry.name]['composition'] = entry.composition

                new_entries = [PDEntry(composition=e['composition'],
                                       energy=e['specific_volume']*e['composition'].num_atoms)
                               for e in struct_props.values()]
                vch = PhaseDiagram(new_entries, elements=[Element(c) for c in self._bl.components])
                pdp = PDPlotter(vch)
            else:
                pdp = PDPlotter(self._bl.ch)

            if plot_type in ['ch', 'vch']:

                fig = pdp.get_plot()
                fig.update_xaxes(title={'text': 'Composition (fraction)'})

                if plot_type == 'vch':
                    fig.update_yaxes(title={'text': 'Referenced Specific Volume (Å^3/atom)'})

            elif (not self._bl.component_data or not self._bl.mpds_liquidus) and 't_vals' not in kwargs:

                print("BinaryLiquid object phase diagram not initialized! Returning plot without liquid energy")
                fig = pdp.get_plot()
                fig.update_xaxes(title={'text': 'Composition (fraction)'})

            else:

                params = self._bl.params
                t_vals = []
                max_phase_temp = 0
                mean_liq_temp = 0

                if 't_vals' in kwargs:
                    t_vals = kwargs['t_vals']
                else:
                    asc_temp = sorted(self._bl.mpds_liquidus, key=lambda x: x[1])
                    mean_liq_temp = (asc_temp[0][1] + asc_temp[-1][1]) / 2
                    mpds_phases = dm.identify_MPDS_phases(self._bl.mpds_json)

                    if mpds_phases:
                        max_phase_temp = max(mpds_phases, key=lambda x: x['tbounds'][1][1])['tbounds'][1][1]
                    else:
                        max_phase_temp = asc_temp[0][1]

                # add curves
                xb, t, a, b, c, d = sp.symbols('x t a b c d')
                L0 = a + b * t
                L1 = c + d * t
                xa = 1 - xb

                R = 8.314
                Ga = (self._bl.component_data[self._bl.components[0]][0] -
                      t * self._bl.component_data[self._bl.components[0]][0] /
                      self._bl.component_data[self._bl.components[0]][1])
                Gb = (self._bl.component_data[self._bl.components[1]][0] -
                      t * self._bl.component_data[self._bl.components[1]][0] /
                      self._bl.component_data[self._bl.components[1]][1])

                G_ideal = R * t * (xa * sp.log(xa) + xb * sp.log(xb))
                G_xs = L0 * (xa * xb) + L1 * (xa * xb * (xa - xb))
                G_liq = Ga * xa + Gb * xb + G_ideal + G_xs

                def get_g_curve(A=0, B=0, C=0, D=0, T=0):
                    gliq_fx = sp.lambdify(xb, G_liq.subs({t: T, a: A, b: B,
                                                          c: C, d: D}), 'numpy')
                    gliq_vals = gliq_fx(X_vals[1:-1])
                    ga = np.float64(Ga.subs({t: T}) / 96485)
                    gb = np.float64(Gb.subs({t: T}) / 96485)
                    name = f'Liquid T={int(T)}K'
                    if B == 0 and D == 0:
                        name += '- No Lb'
                    return go.Scatter(
                        x=X_vals,
                        y=[ga] + [g / 96485 for g in gliq_vals] + [gb],
                        mode='lines',
                        name=name
                    )

                if t_vals:
                    traces = []
                    for temp in reversed(t_vals):
                        traces.append(get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=temp))
                else:
                    # plot temp-independent terms at  max phase decomp temperature, full gliquid at same temp then 0K
                    traces = [get_g_curve(A=params[0] + params[1] * mean_liq_temp,
                                          C=params[2] + params[3] * mean_liq_temp,
                                          T=max_phase_temp),
                              get_g_curve(A=params[0],
                                          B=params[1],
                                          C=params[2],
                                          D=params[3],
                                          T=max_phase_temp),
                              get_g_curve(A=params[0],
                                          B=params[1],
                                          C=params[2],
                                          D=params[3])
                              ]

                # the PDPlotter source code was modified here to allow for the trace order to be shifted
                # if this causes an error, run with 'ch' instead of 'ch+g'
                fig = pdp.get_plot(data=traces)
                fig.update_xaxes(title={'text': 'Composition (fraction)'})

        elif plot_type in ['fit', 'fit+liq', 'pred', 'pred+fit', 'pred+liq', 'pred+fit+liq']:

            mpds_liq = []
            fit_liq = []
            gas_temp = None

            if plot_type in ['fit+liq', 'pred+liq', 'pred+fit+liq']:
                mpds_liq = self._bl.mpds_liquidus
                if not mpds_liq:
                    print("BinaryLiquid object liquidus not intialized! Returning plot without digitized liquidus")

            if plot_type in ['pred+fit', 'pred+fit+liq']:
                if 'fit_params' in kwargs:
                    fit_params = kwargs['fit_params']
                    if len(fit_params) == 4 and fit_params[0] != 0:
                        pred_params = self._bl.params.copy()
                        self._bl.params = fit_params
                        self._bl.update_phase_points()
                        fit_liq = self._bl.phases[-1]['points']
                        self._bl.params = pred_params
                        self._bl.update_phase_points()
                    else:
                        print("Fitted parameters are invalid! Returning plot without fitted liquidus")
                else:
                    print("Keyworded argument fit_params not specified! Returning plot without fitted liquidus")

            pred_pd = bool(plot_type in ['pred', 'pred+fit', 'pred+liq', 'pred+fit+liq'])

            if pred_pd:
                gas_temp = min([cd[3] for cd in self._bl.component_data.values()])

            if self._bl.hsx is None:
                self._bl.update_phase_points()
            fig = self._bl.hsx.plot_tx(mpds_liquidus=mpds_liq, fitted_liquidus=fit_liq, pred=pred_pd, gas_temp=gas_temp)

        elif plot_type in ['ftable', 'ptable']:

            if plot_type == 'ftable':
                pd_type = 'Fitted'
            else:
                pd_type = 'Predicted'

            # df contains parameter values, MAE, RMSE, PD entry, and counts of invariant points
            param_table_headers = ['L0_a', 'L0_b', 'L1_a', 'L1_b']
            parameters_for_table = [['N/A'] * 4]
            inv_table_headers = ['PD Source', 'Eutectics', 'Congruent Melting', 'Peritectics', 'Liquid Misc Gaps']
            inv_counts = [['MPDS', 'N/A', 'N/A', 'N/A', 'N/A'],
                          [pd_type, 'N/A', 'N/A', 'N/A', 'N/A']]
            mae = 'N/A'
            rmse = 'N/A'
            mpds_ref = 'none'

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 2))

            # MPDS data exists for analysis
            if 'reference' in self._bl.mpds_json and self._bl.mpds_json['reference'] is not None:
                # mpds_ref = self._bl.mpds_json['reference']['entry'].split('//')[1]
                mpds_ref = self._bl.mpds_json['reference']['citation']
                # if citation not found in json, run dm.get_ref_data and update cached file
                mpds_invs = self._bl.find_invariant_points()
                mpds_inv_types = ['eut', 'cmp', 'per', 'mig']
                mpds_inv_counts = [len([inv for inv in mpds_invs if inv['type'] == itype]) for itype in mpds_inv_types]
                inv_counts[0] = ['MPDS'] + mpds_inv_counts

            # Parameter data exists
            if self._bl.L0_a() != 0:

                parameter_values = self._bl.params
                for ind, param in enumerate(parameter_values):
                    formatted_param = "{:.4g}".format(param)
                    if abs(param) >= 1000:
                        formatted_param = "{:,.0f}".format(float(formatted_param))
                    parameters_for_table[0][ind] = formatted_param

                if not self._bl.hsx:
                    self._bl.update_phase_points()

                # System has parameter data and liquidus line to compare to
                if self._bl.mpds_liquidus:
                    mae, rmse = self._bl.calculate_deviation_metrics()
                    if mae >= 1000:
                        mae = "{:,.0f}".format(mae) + ' K'
                        rmse = "{:,.0f}".format(rmse) + ' K'
                    else:
                        mae = "{:,.1f}".format(mae) + ' K'
                        rmse = "{:,.1f}".format(rmse) + ' K'

                inv_count_dict = self._bl.hsx.liquidus_invariants()[2]
                hsx_inv_types = ['Eutectics', 'Congruent Melting', 'Peritectics', 'Misc Gaps']
                inv_counts[1] = [pd_type] + [inv_count_dict[itype] for itype in hsx_inv_types]

            # Plot the first table
            ax.table(colLabels=param_table_headers,
                     cellText=parameters_for_table,
                     loc='upper left',
                     colWidths=[0.2, 0.2, 0.2, 0.2],
                     colColours=['#A0CBE2'] * 4,
                     cellLoc='center',
                     bbox=[0, 0.68, 0.8, 0.3])

            # Plot the second table
            table_bottom = ax.table(cellText=inv_counts,
                                    colLabels=inv_table_headers,
                                    loc='lower left',
                                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2],
                                    colColours=['#A0CBE2'] * 5,
                                    cellLoc='center',
                                    bbox=[0, -0.01, 1.09, 0.5])

            ax.text(0, 1.05, f'{pd_type} Parameters - Liquidus MAE: {mae}, RMSE: {rmse}', fontweight='semibold',
                    horizontalalignment='left', verticalalignment='center', fontsize=12,
                    transform=ax.transAxes)

            ax.text(0, 0.56, 'Invariant Point Counts', fontsize=12,
                    horizontalalignment='left', verticalalignment='center', fontweight='semibold',
                    transform=ax.transAxes)

            ax.text(0.815, 0.97, 'Phase Diagram Source:', horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, fontweight='semibold')

            ax.text(0.815, 0.86, mpds_ref, horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes)

            ax.text(0.815, 0.69, 'DFT Energies:', horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes, fontweight='semibold')

            ax.text(0.815, 0.58, 'Materials Project GGA', horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes)

            table_bottom.auto_set_font_size(False)
            table_bottom.set_fontsize(10)
            ax.axis('off')
            plt.tight_layout()

        elif plot_type == 'nmp':

            fig, ax = plt.subplots(figsize=(8, 5))
            num_iters = self._bl.opt_path.shape[2]
            fig.suptitle(self._bl.sys_name + " 2-Parameter Nelder-Mead Path", fontweight='semibold', fontsize=14)

            tdev_range = [None, None]
            for i in range(num_iters):
                path_i = self._bl.opt_path[:, :, i]
                t_devs = [num for num in path_i[:, -1:] if num != float('inf')]
                if tdev_range[0] is None:
                    tdev_range[0] = t_devs[0]
                if tdev_range[1] is None:
                    tdev_range[1] = t_devs[0]
                tdev_range[0] = min(tdev_range[0], min(t_devs))
                tdev_range[1] = max(tdev_range[1], max(t_devs))

            # Triangle color mapping
            sm1 = cm.ScalarMappable(cmap=cm.get_cmap('winter'), norm=LogNorm(vmin=1, vmax=num_iters))
            triangle_colors = sm1.to_rgba(np.arange(1, num_iters + 1, 1))
            ticks = [2 ** exp for exp in np.arange(0, math.ceil(np.log2(num_iters)), 1)]
            cbar1 = fig.colorbar(sm1, ax=ax, aspect=14)
            cbar1.minorticks_off()
            cbar1.set_ticks(ticks)
            cbar1.set_ticklabels(ticks)
            cbar1.set_label('Nelder-Mead Iteration', style='italic', labelpad=8, fontsize=12)

            # Marker color mapping
            sm2 = cm.ScalarMappable(cmap=cm.get_cmap('autumn'), norm=plt.Normalize(tdev_range[0], tdev_range[1]))
            marker_colors = sm2.to_rgba(np.arange(tdev_range[0], tdev_range[1], 1))
            cbar2 = fig.colorbar(sm2, ax=ax, aspect=14)
            cbar2.set_label('MAE From MPDS Liquidus (' + chr(176) + 'C)', style='italic', labelpad=10, fontsize=12)

            plotted_points = []
            for i in range(num_iters):
                path_i = self._bl.opt_path[:, :, i]
                triangle = path_i[:, :-1]
                t_devs = path_i[:, -1:]

                # Plot triangles
                coordinates = [triangle[i, :] for i in range(triangle.shape[0])]
                pair_combinations = list(combinations(coordinates, 2))
                for combo in pair_combinations:
                    line = np.array(combo)
                    ax.plot(line[:, 0], line[:, 1],
                            color=triangle_colors[i], linewidth=(2 - 1.7 * (i / num_iters)), zorder=0)

                # Plot markers
                for point, t_dev in zip(triangle, t_devs):
                    if list(point) in plotted_points:
                        continue
                    if t_dev != float('inf'):
                        c_ind = int(t_dev - tdev_range[0])
                        marker_color = marker_colors[c_ind]
                        ax.scatter(point[0], point[1], s=(55 - 54.7 * (i / num_iters)),
                                   color=marker_color, marker='^', edgecolor='black', linewidth=0.3, zorder=1)
                    else:
                        ax.scatter(point[0], point[1], s=(45 - 44.7 * (i / num_iters)),
                                   color='black', label='Incalculable MAE', marker='^', zorder=1)
                    plotted_points.append(list(point))

            # Legend, Axis labels, Plot area scaling
            # handles, labels = plt.gca().get_legend_handles_labels()
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys())
            ax.autoscale()
            ly, uy = ax.get_ylim()
            ax.set_ylim((uy + ly) / 2 - (uy - ly) / 2 * 1.1,
                        (uy + ly) / 2 + (uy - ly) / 2 * 1.1)
            lx, ux = ax.get_xlim()
            ax.set_xlim((ux + lx) / 2 - (ux - lx) / 2 * 1.1,
                        (ux + lx) / 2 + (ux - lx) / 2 * 1.1)
            ax.set_xlabel('L0_b', fontweight='semibold', fontsize=12)
            ax.set_ylabel('L1_b', fontweight='semibold', fontsize=12)
            fig.tight_layout()

        elif plot_type == 'gliq':

            if 't_vals' not in kwargs:
                print("Missing keyworded argument for t_vals, plotting gliquid at 0K")
                t_vals = [0]
                units = 'K'
            else:
                t_vals = kwargs['t_vals']
                units = kwargs.get('units', 'C')

            if units == 'C':
                t_vals = [t + 273.15 for t in t_vals]
            elif units != 'K':
                raise SyntaxError("Keyworded argument 'units' must be set to either 'C' or 'K'!")

            xb, t, a, b, c, d = sp.symbols('x t a b c d')
            L0 = a + b * t
            L1 = c + d * t
            xa = 1 - xb

            R = 8.314
            Ga = (self._bl.component_data[self._bl.components[0]][0] -
                  t * self._bl.component_data[self._bl.components[0]][0] /
                  self._bl.component_data[self._bl.components[0]][1])
            Gb = (self._bl.component_data[self._bl.components[1]][0] -
                  t * self._bl.component_data[self._bl.components[1]][0] /
                  self._bl.component_data[self._bl.components[1]][1])

            G_ideal = R * t * (xa * sp.log(xa) + xb * sp.log(xb))
            G_xs = L0 * (xa * xb) + L1 * (xa * xb * (xa - xb))
            G_liq = Ga * xa + Gb * xb + G_ideal + G_xs

            n_plots = len(t_vals)
            if n_plots == 1:
                n_rows = 1
            else:
                n_rows = 2
            n_cols = math.ceil(n_plots / 2.0)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 4 * n_rows))
            fig.suptitle('Gliquid Curves & Convex Hulls', fontweight='semibold')

            for i in range(n_rows):
                for j in range(n_cols):
                    if n_rows == 1:
                        ax = axes
                    elif n_cols == 1:
                        ax = axes[i]
                    elif (i + j + 2) <= n_plots:
                        ax = axes[i][j]
                    else:
                        break
                    temp = t_vals.pop(0)
                    gliq_fx = sp.lambdify(xb, G_liq.subs({t: temp, a: self._bl.L0_a(), b: self._bl.L0_b(),
                                                          c: self._bl.L1_a(), d: self._bl.L1_b()}), 'numpy')
                    gliq_vals = gliq_fx(X_vals)
                    ax.plot(X_vals, gliq_vals)
                    if units == 'C':
                        ax.set_title(f"T = {temp - 273.15}C")
                    else:
                        ax.set_title(f"T = {temp}K")
                    last_phase = []
                    for phase in self._bl.phases:
                        if 'energy' in phase:
                            if last_phase:
                                ax.plot([last_phase[0], phase['comp']], [last_phase[1], phase['energy']],
                                        color='black', linestyle='--', zorder=-1)
                            ax.scatter(phase['comp'], phase['energy'], label=phase['name'])
                            last_phase = [phase['comp'], phase['energy']]
                    ax.legend()

        else:
            raise NotImplementedError(f"No implementation for plot type: {plot_type}")

        return fig

    def show(self, plot_type: str, **kwargs) -> None:

        fig = self.get_plot(plot_type, **kwargs)

        if plot_type in ['ch', 'ch+g', 'vch', 'vch+g', 'fit', 'fit+liq', 'pred', 'pred+liq', 'pred+fit', 'pred+fit+liq']:
            fig.show()
        else:
            plt.show()
            plt.close(fig)

    def write_image(self, plot_type: str, stream: str | StringIO, image_format: str = "svg", **kwargs) -> None:

        fig = self.get_plot(plot_type, **kwargs)
        if fig is None:
            return

        if plot_type in ['fit', 'fit+liq', 'pred', 'pred+liq', 'pred+fit', 'pred+fit+liq']:
            fig.write_image(stream, format=image_format)
        elif plot_type in ['ch', 'ch+g', 'vch', 'vch+g']:
            fig.write_image(stream, format=image_format, width=480 * 1.8, height=300 * 1.7)
        else:
            fig.savefig(stream, format=image_format)  # bbox_inches='tight'?
            plt.close(fig)


# bl = BinaryLiquid.from_cache('Pb-Tb', params=[-131021, -4.99, 71675, -8.49])
# bl.update_phase_points()
# points = sorted([[100 - 100*p[0], p[1]-273.15] for p in bl.phases[-1]['points']])
# x_points = [p[0] for p in points]
# y_points = [p[1] for p in points]
# df = pd.DataFrame({'Composition (atomic % Pb)': x_points, 'Liquidus Temperature (°C)': y_points})
# df.to_excel(f'data/8_12_24_Tb-Pb_prediction_coordinates.xlsx', index=False)
# blp = BLPlotter(bl)
# blp.write_image(plot_type='pred', stream='Pb-Tb_predicted.svg')
# blp.write_image(plot_type='ch+g', t_vals=[721.8, 1629, 1947.3], stream='Pb-Tb_ch+g.svg')

# bl = BinaryLiquid.from_cache('Cs-V', params=[-19200, -3, 5170, -1.2], reconstruction=True)
# print(bl.component_data)
# BLPlotter(bl).show(plot_type='pred')

# bl = BinaryLiquid.from_cache('Cu-Mg')
# blp = BLPlotter(bl)
# blp.show(plot_type='vch')
