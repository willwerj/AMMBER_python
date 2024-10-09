"""
@author: Joshua Willwerth

This script provides functions to load locally cached phase diagram and DFT entry data
"""
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram, CompoundPhaseDiagram, PDEntry
from pymatgen.core import Composition, Element, Structure

import os
import json
import numpy as np

data_dir = "data"
fusion_enthalpies_file = f"{data_dir}/fusion_enthalpies.json"
fusion_temps_file = f"{data_dir}/fusion_temperatures.json"
vaporization_enthalpies_file = f"{data_dir}/vaporization_enthalpies.json"
vaporization_temps_file = f"{data_dir}/vaporization_temperatures.json"

if not os.path.exists(fusion_enthalpies_file):
    raise FileNotFoundError("Invalid path provided for fusion enthalpies file!")
with open(fusion_enthalpies_file, "r") as file:
    melt_enthalpies = json.load(file)

if not os.path.exists(fusion_temps_file):
    raise FileNotFoundError("Invalid path provided for fusion temperatures file!")
with open(fusion_temps_file, "r") as file:
    melt_temps = json.load(file)

if not os.path.exists(vaporization_enthalpies_file):
    raise FileNotFoundError("Invalid path provided for vaporization enthalpies file!")
with open(vaporization_enthalpies_file, "r") as file:
    boiling_enthalpies = json.load(file)

if not os.path.exists(vaporization_temps_file):
    raise FileNotFoundError("Invalid path provided for vaporization temperatures file!")
with open(vaporization_temps_file, "r") as file:
    boiling_temps = json.load(file)


def t_at_boundary(t, boundary):
    return t <= boundary[0] + 2 or t >= boundary[1] - 2


def section_liquidus(points):
    sections = []
    current_section = []

    x1, y1 = points[0]
    x2, y2 = points[1]

    if x2 > x1:
        direction = "increasing"
        current_section.append(points[0])
    elif x2 < x1:
        direction = "decreasing"
        current_section.append(points[0])
    else:
        direction = None
        sections.append([points[0]])

    for i in range(1, len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if x2 > x1:
            new_direction = "increasing"
        elif x2 < x1:
            new_direction = "decreasing"
        else:
            new_direction = None
        # add x1, y1
        current_section.append(points[i])

        if new_direction != direction or new_direction is None:
            if current_section:
                sections.append(current_section)
                current_section = []
        direction = new_direction

    if current_section:
        current_section.append(points[-1])
        sections.append(current_section)
    return sections


def within_tol_from_line(p1, p2, p3, tol):
    try:
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    except ZeroDivisionError:
        return tol >= abs(p1[1] - p2[1])
    y_h = m * (p3[0] - p1[0]) + p1[1]
    return tol >= abs(p3[1] - y_h)


def fill_liquidus(p1, p2, max_interval):
    num_between = int(np.floor((p2[0] - p1[0]) / max_interval))
    filled_X = np.linspace(p1[0], p2[0], num_between + 2)
    filled_T = np.linspace(p1[1], p2[1], num_between + 2)
    filled_section = [[filled_X[i], filled_T[i]] for i in range(len(filled_X))]
    return filled_section[1:-1]


# load the MPDS data for system from the API or local cache
# returns dict, dict, 2D list
# ind is a testing feature used to select a phase diagram at a specific rank. highest ranked PD selected by default
def get_MPDS_data(components, pd_ind=0):
    """Retrive MPDS data from cache
    Required: List of components. Must provide only two components if querying binary phase diagrams.
    pd -> index of PD sorted by selection criteria. Returns a tuple of three fields:
    MPDS JSON (dict), Component data (dict), MPDS Liquidus (2D list)"""

    sys = '-'.join(sorted(components))
    component_data = {}

    for comp in components:
        component_data[comp] = [melt_enthalpies[comp], melt_temps[comp], boiling_enthalpies[comp], boiling_temps[comp]]

    # look for cached json for the system
    sys_file = os.path.join(f"{data_dir}\\{sys}", f"{sys}_MPDS_PD_{pd_ind}.json")
    if os.path.exists(sys_file):
        # get MPDS data from stored jsons in liquidus curves folder
        print("\nloading JSON from cache...")
        with open(sys_file, 'r') as f:
            mpds_json = json.load(f)
        # ------------------------------------------------------------------------
        return mpds_json, component_data, extract_MPDS_liquidus(mpds_json, components)

    else:
        raise FileNotFoundError(f"Missing or improperly named phase diagram file for {sys} system!")


def shape_to_list(svgpath):
    # split svgpath into tags, ordered pairs
    data = svgpath.split(' ')

    # remove 'L' and 'M' tags, so only ordered pairs remain
    data = [s for s in data if not (s == 'L' or s == 'M')]

    # convert string pairs into [X, T] float pairs and store as list
    X = [float(i.split(',')[0]) / 100.0 for i in data]
    T = [float(i.split(',')[1]) + 273.15 for i in data]
    return [[X[i], T[i]] for i in range(len(X))]


# pull liquidus curve data from MPDS json and convert to list of [X, T] coordinates in ascending composition order
# returns 2D list
def extract_MPDS_liquidus(MPDS_json, verbose=True):
    if MPDS_json['reference'] is None:
        if verbose:
            print("system JSON does not contain any data!\n")
        return None

    components = MPDS_json['chemical_elements']
    if verbose:
        print("reading MPDS liquidus from entry at " + MPDS_json['reference']['entry'] + "...\n")

    # extract liquidus curve svgpath from system JSON
    data = ""
    for boundary in MPDS_json['shapes']:
        if 'label' in boundary and boundary['label'] == 'L':
            data = boundary['svgpath']
            break
    if not data:
        if verbose:
            print("no liquidus data found in JSON!")
        return None

    MPDS_liquidus = shape_to_list(data)

    # remove points at the edge of the graph boundaries
    MPDS_liquidus = [coord for coord in MPDS_liquidus if not t_at_boundary(coord[1] - 273.15, MPDS_json['temp'])]

    if len(MPDS_liquidus) < 3:
        if verbose:
            print("MPDS liquidus does not span the entire composition range!")
        return None

    # split liquidus into segments of continuous points
    sections = section_liquidus(MPDS_liquidus)

    # sort sections by descending size
    sections.sort(key=len, reverse=True)

    # sort each section by ascending composition
    for section in sections:
        section.sort()

    # record endpoints of main section
    MPDS_liquidus = sections.pop(0)

    lhs = [0, melt_temps[components[0]]]
    rhs = [1, melt_temps[components[1]]]

    # append sections to the liquidus if not overlapping in range
    for section in sections:

        # if section upper bound is less than main section lower bound
        if section[-1][0] <= MPDS_liquidus[0][0] and within_tol_from_line(MPDS_liquidus[0], lhs, section[-1], 250):
            MPDS_liquidus = section + MPDS_liquidus

        # if section lower bound is greater than main section upper bound
        elif section[0][0] >= MPDS_liquidus[-1][0] and within_tol_from_line(MPDS_liquidus[-1], rhs, section[0], 250):
            MPDS_liquidus.extend(section)

        # i'll admit it at this point I am feeling pretty dumb about not using a svgpath parser because I have
        # do all of these strange exceptions to make this work. don't worry about this, it's just some edge case
        elif len(section) == 2:
            if section[0][0] < MPDS_liquidus[0][0] and within_tol_from_line(MPDS_liquidus[0], lhs, section[0], 170):
                MPDS_liquidus = [section[0]] + MPDS_liquidus

            elif section[-1][0] > MPDS_liquidus[-1][0] and within_tol_from_line(MPDS_liquidus[-1], rhs, section[-1],
                                                                                170):
                MPDS_liquidus.extend(section)

    # if the liquidus does not have endpoints near the ends of the composition range, melting temps won't be good
    if 100 * MPDS_liquidus[0][0] > 3 or 100 * MPDS_liquidus[-1][0] < 97:
        if verbose:
            print(f"MPDS liquidus does not span the entire composition range! "
                  f"({100 * MPDS_liquidus[0][0]}-{100 * MPDS_liquidus[-1][0]})")
        return None

    MPDS_liquidus.sort()

    # fill in ranges with missing points
    for i in reversed(range(len(MPDS_liquidus) - 1)):
        if MPDS_liquidus[i + 1][0] - MPDS_liquidus[i][0] > 0.06:
            filler = fill_liquidus(MPDS_liquidus[i], MPDS_liquidus[i + 1], 0.03)
            for point in reversed(filler):
                MPDS_liquidus.insert(i + 1, point)

    # filter out duplicate values in the liquidus curve; greatly improves runtime efficiency
    for i in reversed(range(len(MPDS_liquidus) - 1)):
        if MPDS_liquidus[i][0] == 0 or MPDS_liquidus[i][1] == 0:
            continue
        if abs(1 - MPDS_liquidus[i + 1][0] / MPDS_liquidus[i][0]) < 0.0005 and \
                abs(1 - MPDS_liquidus[i + 1][1] / MPDS_liquidus[i][1]) < 0.0005:
            del (MPDS_liquidus[i + 1])

    return MPDS_liquidus


# returns the DFT convex hull of a given system with specified functionals
def get_dft_convexhull(system, dft_type="GGA/GGA+U", inc_structure_data=False, verbose=False):
    if isinstance(system, str):
        components = sorted(system.split('-'))
        sys_name = '-'.join(components)
    elif isinstance(system, list):
        components = sorted(system)
        sys_name = '-'.join(components)
    else:
        print("Error: system must be a hyphenated string or a list")
        return None
    try:
        [Composition(c) for c in components]
    except ValueError as e:
        print(e)
        return None

    if 'Yb' in components:
        dft_type = "GGA"
    if verbose:
        print("using DFT entries solved with", dft_type, "functionals")
    sys_dir = f"{data_dir}\\{sys_name}"
    dft_entries_file = os.path.join(sys_dir, f"{sys_name}_ENTRIES_MP_GGA.json")
    use_compound_pd = bool([c for c in components if len(Composition(c).elements) > 1])

    if os.path.exists(dft_entries_file):
        with open(dft_entries_file, "r") as f:
            computed_entry_dicts = json.load(f)
    else:
        FileNotFoundError(f"Missing or improperly named DFT entries file for {sys_name} system!")

    stable_structure_props = {}

    try:
        if use_compound_pd:
            pd = CompoundPhaseDiagram(terminal_compositions=[Composition(c) for c in components],
                                      entries=[ComputedEntry.from_dict(e) for e in computed_entry_dicts])
        else:
            pd = PhaseDiagram(elements=[Element(c) for c in components],
                              entries=[ComputedEntry.from_dict(e) for e in computed_entry_dicts])
        if verbose:
            print(len(pd.stable_entries) - 2, "stable line compound(s) on the DFT convex hull\n")

        if not inc_structure_data:
            return pd

        for entry in pd.stable_entries:
            matching_composition = [e for e in computed_entry_dicts
                                    if Composition.from_dict(e['composition']) == entry.composition]
            lowest_energy_entry = min(matching_composition, key=lambda x: x['energy'])
            struct = Structure.from_dict(lowest_energy_entry['structure'])
            volume = struct.volume  # structure volume in cubic angstrom
            num_atoms = struct.num_sites  # number of atoms per structure
            specific_volume = volume / num_atoms  # specific volume is in units of cubic angstrom per atom
            stable_structure_props[entry.composition.reduced_formula] = {'specific_volume': specific_volume}

        return pd, stable_structure_props
    except ValueError as e:
        print(f"error loading DFT entries from cache: {e}")
        return None


def identify_MPDS_phases(MPDS_json, verbose=False):
    if MPDS_json['reference'] is None:
        if verbose:
            print("system JSON does not contain any data!\n")
        return []

    phases = []
    data = ""
    for shape in MPDS_json['shapes']:

        if 'nphases' in shape and 'is_solid' in shape:
            # indentify line compounds and single-phase solid solutions
            if shape['nphases'] == 1 and shape['is_solid'] and 'label' in shape:
                # if '(' in shape['label'].split(' ')[0]:
                #     continue

                # split svgpath into tags, ordered pairs
                data = shape_to_list(shape['svgpath'])

                if not data:
                    if verbose:
                        print(f"no point data found for phase {shape['label']} in JSON!")
                    continue

                # sort by ascending T value
                data.sort(key=lambda x: x[1])
                tbounds = [data[0], data[-1]]
                # comp = Composition(shape['label'].split(' ')[0]).fractional_composition.as_dict()[components[1]]
                if shape['kind'] == 'phase':
                    data.sort(key=lambda x: x[0])
                    cbounds = [data[0], data[-1]]
                    if cbounds[-1][0] < 0.03 or cbounds[0][0] > 0.97:
                        continue

                    phases.append({'type': 'ss', 'name': shape['label'].split(' ')[0], 'comp': tbounds[1][0],
                                   'cbounds': cbounds, 'tbounds': tbounds})
                else:  # kind == compound
                    phases.append({'type': 'lc', 'name': shape['label'].split(' ')[0], 'comp': tbounds[1][0],
                                   'tbounds': tbounds})

    if not data:
        if verbose:
            print("no phase data found in JSON!")
        return phases

    phases.sort(key=lambda x: x['comp'])
    return phases


def get_low_temp_phase_data(mpds_json, mp_ch):
    mpds_congruent_phases = {}
    mpds_incongruent_phases = {}
    max_phase_temp = 0

    identified_phases = identify_MPDS_phases(mpds_json)
    mpds_liquidus = extract_MPDS_liquidus(mpds_json, verbose=False)

    def phase_decomp_on_liq(phase, liq):
        if liq is None:
            return False
        for i in range(len(liq) - 1):
            if liq[i][0] == phase['tbounds'][1][0]:
                return abs(liq[i][1] - phase['tbounds'][1][1]) < 10
            # composition falls between two points:
            elif liq[i][0] < phase['tbounds'][1][0] < liq[i + 1][0]:
                return abs((liq[i][1] + liq[i + 1][1]) / 2 - phase['tbounds'][1][1]) < 10

    for phase in identified_phases:
        # Check to see if these are low temperature phases (phase lower bound must be within lower 10% of temp range)
        if (phase['type'] in ['lc', 'ss'] and phase['tbounds'][0][1] < (mpds_json['temp'][0] + 273.15) +
                (mpds_json['temp'][1] - mpds_json['temp'][0]) * 0.10):
            if phase_decomp_on_liq(phase, mpds_liquidus):
                if phase['type'] == 'ss':
                    mpds_congruent_phases[phase['name']] = (
                        (phase['cbounds'][0][0], phase['cbounds'][1][0]), phase['tbounds'][1][1])
                else:
                    mpds_congruent_phases[phase['name']] = ((phase['comp'], phase['comp']), phase['tbounds'][1][1])
            else:
                if phase['type'] == 'ss':
                    mpds_incongruent_phases[phase['name']] = (
                        (phase['cbounds'][0][0], phase['cbounds'][1][0]), phase['tbounds'][1][1])
                else:
                    mpds_incongruent_phases[phase['name']] = (
                        (phase['comp'], phase['comp']), phase['tbounds'][1][1])
            max_phase_temp = max(phase['tbounds'][1][1], max_phase_temp)

    if max_phase_temp == 0 and mpds_liquidus:
        asc_temp = sorted(mpds_liquidus, key=lambda x: x[1])
        max_phase_temp = asc_temp[0][1]

    mp_phases = {}
    mp_phases_ebelow = {}
    min_form_e = 0

    for entry in mp_ch.stable_entries:
        # skip pure components
        if len(entry.composition.fractional_composition.as_dict()) == 1:
            continue
        comp = entry.composition.fractional_composition.as_dict()[mp_ch.elements[1].symbol]

        form_e = mp_ch.get_form_energy_per_atom(entry)
        mp_phases[entry.name] = ((comp, comp), form_e)
        min_form_e = min(form_e, min_form_e)

        ch_copy = PhaseDiagram([e for e in mp_ch.stable_entries if e != entry])
        e_below_hull = -abs(mp_ch.get_hull_energy_per_atom(entry.composition) -
                            ch_copy.get_hull_energy_per_atom(entry.composition))
        mp_phases_ebelow[entry.name] = ((comp, comp), e_below_hull)

    return [mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp], [mp_phases, mp_phases_ebelow, min_form_e]
