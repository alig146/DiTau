#!/usr/bin/env python
# coding: utf-8

import glob
import time
import json
import numpy as np
import awkward as ak
import uproot
import vector
import pickle
import pandas as pd
import matplotlib.pyplot as plt


branches = [
    'ditau_obj_truth_leadTau_p4',
    'ditau_obj_truth_subleadTau_p4',
    # 'boson_0_classifierParticleOrigin',
    # 'boson_0_mother_pdgId',
    # 'boson_0_mother_status',
    # 'boson_0_pdgId',
    # 'boson_0_truth_pdgId',
    # 'boson_0_truth_q',
    # 'boson_0_truth_status',
    # 'boson_0_q',
    'ditau_obj_IsTruthMatched',
    'ditau_obj_IsTruthHadronic',
    'ditau_obj_truth_p4',
    'met_truth_p4',
    'NOMINAL_pileup_random_run_number',
    'ditau_obj_bdt_score',
    'ditau_obj_omni_score',
    'ditau_obj_leadsubjet_charge',
    'ditau_obj_leadsubjet_n_core_tracks',
    'ditau_obj_leadsubjet_p4',
    'ditau_obj_nSubjets',
    'ditau_obj_n_tracks',
    'ditau_obj_p4',
    'ditau_obj_subleadsubjet_charge',
    'ditau_obj_subleadsubjet_n_core_tracks',
    'ditau_obj_subleadsubjet_p4',
    'jets_p4',
    'event_number',
    'met_p4',
    'met_sumet',
    'met_hpto_p4',
    'event_is_bad_batman',
    'NOMINAL_pileup_combined_weight',
    'n_bjets_GN2v01_FixedCutBEff_70',
    'weight_mc',
    'n_jets'
]

signal_branches = branches + ['theory_weights_nominal']
background_branches = branches

data_branches = [
    'event_number',    
    'met_p4',
    'met_sumet',
    'met_hpto_p4',
    'event_is_bad_batman',
    'ditau_obj_bdt_score',
    'ditau_obj_omni_score',
    'ditau_obj_leadsubjet_charge',
    'ditau_obj_leadsubjet_n_core_tracks',
    'ditau_obj_leadsubjet_p4',
    'ditau_obj_nSubjets',
    'ditau_obj_n_tracks',
    'ditau_obj_p4',
    'jets_p4',
    'ditau_obj_subleadsubjet_charge',
    'ditau_obj_subleadsubjet_n_core_tracks',
    'ditau_obj_subleadsubjet_p4',
    'n_bjets_GN2v01_FixedCutBEff_70'
]

path_template_mc = '/pscratch/sd/a/agarabag/ditdau_samples/V02/mc/ditau_hh/{year}/nom/user.*.{dsid}.*/user.*.root'
path_template_data = '/pscratch/sd/a/agarabag/ditdau_samples/V02/data/ditau_hh/data{year}/user.*/user.*.root'

calc_vars = ['ditau_pt', 'leadsubjet_pt', 'subleadsubjet_pt', 'visible_ditau_m', 'met', 'collinear_mass', 'x1', 'x2', 'met_sig', 'met_phi', 'event_number', 'k_t', 'kappa', 'delta_R',
             'delta_phi', 'delta_eta', 'combined_weights', 'fake_factor', 'delta_R_lead', 'delta_eta_lead', 'delta_phi_lead', 'delta_R_sublead', 'delta_eta_sublead', 'delta_phi_sublead',
             'met_centrality', 'omni_score', 'leadsubjet_charge', 'subleadsubjet_charge', 'leadsubjet_n_core_tracks', 'subleadsubjet_n_core_tracks', 'e_ratio_lead', 'e_ratio_sublead',
             'higgs_pt', 'leadsubjet_eta', 'subleadsubjet_eta', 'ditau_eta', 'delta_phi_met_ditau']

ditau_id_cut = 0.9993
ff_scale = 1

# -----------------------------
# Constants and configuration (Run 2 defaults)
# -----------------------------

Ztt_inc = [
    700792, 700793, 700794,
    700901, 700902, 700903,
    700360
]

ttV = [
    410155,
    504330, 504334, 504338, 504342, 504346,
    304014,
]

VV = [
    700600, 700601, 700602, 700603, 700604,
    701085, 701095, 701105, 701110, 701115, 701125,
]

Top = [
    410470, 410471,
    410644, 410645, 410646, 410647, 410658, 410659
]

W = [
    700338, 700339, 700340,
    700341, 700342, 700343,
    700344, 700345, 700346,
    700347, 700348, 700349,
    700362, 700363, 700364,
]

Zll_inc = [
    700320, 700321, 700322, 700467, 700468, 700469,
    700323, 700324, 700325, 700470, 700471, 700472,
    700358,
    700359,
]

ggH = [345120, 345121, 345122, 345123, 345324, 600686]
VBFH = [346190, 346191, 346192, 346193, 345948]
WH = [345211, 345212]
ZH = [345217]
ttH = [346343, 346344, 346345]

LUMI = {
    '15': 3244.54,
    '16': 33402.2,
    '17': 44630.6,
    '18': 58791.6,
}

LUMI_SCALE = {
    '15': (3244.54 + 33402.2) / 3244.54,
    '16': (3244.54 + 33402.2) / 33402.2,
    '17': 1,
    '18': 1,
}

year_map = {
    'mc20e': '18',
    'mc20d': '17',
    'mc20a': '15'
}

# -----------------------------
# Constants and configuration (Run 3 defaults)
# -----------------------------

Ztt_inc_run3 = [
    700792, 700793, 700794, 700901, 700902, 700903, 700982
]

ttV_run3 = [
    522024, 522028, 522032, 522036, 522040, 700995, 700996, 700997
]

VV_run3 = [
    701040, 701045, 701050, 701060, 701085, 701095, 701105, 701110, 701115, 701125
]

Top_run3 = [
    601229, 601230, 601237, 601348, 601349, 601350, 601351
]

W_run3 = [
    700777, 700778, 700779, 700780, 700781, 700782, 700783, 700784, 700785
]

Zll_inc_run3 = [
    700786, 700787, 700788, 700895, 700896, 700897,
    700789, 700790, 700791, 700898, 700899, 700900,
    700981
]

ggH_run3 = [
    603414, 603415, 603416, 603417, 603418, 602632
]

VBFH_run3 = [
    603422, 603423, 603424, 603425, 601599
]

WH_run3 = [603426, 603427]
ZH_run3 = [603428]
ttH_run3 = [603419, 603420, 603421]

LUMI_run3 = {
    '22': 26071.4,
    '23': 25767.5,
    '24': 109400.0,
}

LUMI_SCALE_run3 = {
    '22': 1,
    '23': 1,
    '24': 1,
}

year_map_run3 = {
    'mc23a': '22',
    'mc23d': '23', 
    'mc23e': '24'
}

# -----------------------------
# Multi-run configuration (Run 2 and Run 3)
# -----------------------------

RUN2_CONFIG = {
    'name': 'run2',
    'year_map': year_map,
    'lumi': LUMI,
    'lumi_scale': LUMI_SCALE,
    'path_template_mc': path_template_mc,
    'path_template_data': path_template_data,
    'trigger': {
        '15': ['HLT_j360', 'HLT_xe70_mht', 'HLT_tau125_medium1_tracktwo'],
        '16': ['HLT_j380', 'HLT_j420_a10_lcw_L1J100', 'HLT_j420_a10r_L1J100', 'HLT_xe110_mht_L1XE50', 'HLT_tau160_medium1_tracktwo'],
        '17': ['HLT_j400', 'HLT_j440_a10_lcw_subjes_L1J100', 'HLT_xe110_pufit_L1XE55', 'HLT_tau160_medium1_tracktwo'],
        '18': ['HLT_j420', 'HLT_j420_a10t_lcw_jes_35smcINF_L1J100', 'HLT_j420_a10t_lcw_jes_35smcINF_L1SC111', 'HLT_xe110_pufit_xe70_L1XE50', 'HLT_tau160_medium1_tracktwoEF_L1TAU100']
    },
    'ff_hist_file': '../data/FF_hadhad_ratio_1d.root',
}

RUN3_CONFIG = {
    'name': 'run3',
    'year_map': year_map_run3,
    'lumi': LUMI_run3,
    'lumi_scale': LUMI_SCALE_run3,
    'path_template_mc': path_template_mc,
    'path_template_data': path_template_data,
    'trigger': {
        '22': ['HLT_j420_pf_ftf_preselj225_L1J100', 'HLT_j420_35smcINF_a10t_lcw_jes_L1J100', 'HLT_xe65_cell_xe90_pfopufit_L1XE50', 'HLT_tau160_mediumRNN_tracktwoMVA_L1TAU100'],
        '23': ['HLT_j420_pf_ftf_preselj225_L1J100', 'HLT_j420_35smcINF_a10t_lcw_jes_L1J100', 'HLT_xe65_cell_xe90_pfopufit_L1XE50', 'HLT_tau160_mediumRNN_tracktwoMVA_L1eTAU140'],
        # '24': ['HLT_j400_pf_ftf_preselj225_L1jJ180', 'HLT_j460_a10t_lcw_jes_L1jLJ140', 'HLT_xe65_cell_xe90_pfopufit_L1jXE100', 'HLT_tau160_mediumRNN_tracktwoMVA_L1eTAU140']
        '24': ['HLT_j400_pf_ftf_preselj225_L1jJ180', 'HLT_xe65_cell_xe90_pfopufit_L1jXE100', 'HLT_tau160_mediumRNN_tracktwoMVA_L1eTAU140']
    },
    'ff_hist_file': '../data/FF_hadhad_ratio_1d_run3.root',
}


def get_config(run: str = 'run2'):
    return RUN3_CONFIG if str(run).lower() in ['run3', 'r3', 'mc23'] else RUN2_CONFIG


# -----------------------------
# Weight helpers
# -----------------------------

def read_event_weights(event_id, data_year):
    # Choose correct cross-section/sum of weights file by run
    year_str = str(data_year)
    if year_str in ['22', '23', '24'] or year_str.startswith('mc23'):
        file_path = '../data/xsec_sumofweights_nom_run3.json'
    else:
        file_path = '../data/xsec_sumofweights_nom.json'
    print(f"Reading weights from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    events = data.get(data_year, {}).get('ditau_hh', [])
    for event in events:
        if event[0] == event_id:
            return (event[1], event[2])
    return (None, None)


def fetch_weights(id_list, data_year):
    results = {}
    for event_id in id_list:
        event_weight, sum_event_weights = read_event_weights(event_id, data_year)
        if (event_weight is None or sum_event_weights is None or sum_event_weights == 0.0):
            results[event_id] = 1.0
            print(f"DSID {event_id} not found for year {data_year} - using weight 1.0")
        else:
            print(f"DSID {event_id}, year: {data_year}, weight: {event_weight}, sum: {sum_event_weights}")
            results[event_id] = event_weight / sum_event_weights
    return results


# -----------------------------
# IO helpers
# -----------------------------

def read_root(dsid_list, mc_ws, year_id='mc20e', year='18', is_signal=False, config=None):
    cfg = get_config('run3') if (year in ['22', '23', '24'] or str(year_id).startswith('mc23')) else get_config('run2') if config is None else config
    out = []
    hlt_branches = cfg['trigger']
    branches_to_read = signal_branches if is_signal else background_branches
    for dsid in dsid_list:
        file_pattern = cfg['path_template_mc'].format(dsid=dsid, year=year_id)
        files = glob.glob(file_pattern)
        for file in files:
            with uproot.open(file + ':NOMINAL') as f_1:
                branches_read = branches_to_read + hlt_branches.get(year, [])
                events = f_1.arrays(branches_read, library='ak')
                mc_weight = events['theory_weights_nominal'] if is_signal else events['weight_mc']
                if year in ['15', '16']:
                    mask_15 = (events['NOMINAL_pileup_random_run_number'] <= 284484) & (events['NOMINAL_pileup_random_run_number'] > 0)
                    mask_16 = (events['NOMINAL_pileup_random_run_number'] > 284484) & (events['NOMINAL_pileup_random_run_number'] <= 311563)
                    events['weight'] = mc_ws[dsid] * ak.ones_like(events['ditau_obj_n_tracks']) * mc_weight * events['NOMINAL_pileup_combined_weight']
                    events['weight'] = ak.where(mask_15,
                                                events['weight'] * cfg['lumi_scale']['15'] * cfg['lumi']['15'],
                                                ak.where(mask_16,
                                                         events['weight'] * cfg['lumi_scale']['16'] * cfg['lumi']['16'],
                                                         events['weight']))
                elif year in ['17', '18', '22', '23', '24']:
                    events['weight'] = (mc_ws[dsid] * ak.ones_like(events['ditau_obj_n_tracks'])) * \
                                       mc_weight * events['NOMINAL_pileup_combined_weight'] * \
                                       cfg['lumi_scale'][year] * cfg['lumi'][year]
                out = ak.concatenate((out, events))
    return out


def read_data_root(year='18', config=None):
    cfg = get_config('run3') if year in ['22', '23', '24'] else get_config('run2') if config is None else config
    file_paths = cfg['path_template_data'].format(year=year)
    out = []
    l1 = glob.glob(file_paths)
    for i in range(len(l1)):
        f_1 = uproot.open(l1[i] + ':NOMINAL')
        if f_1.num_entries == 0:
            continue
        branches_read = []
        branches_read.extend(data_branches)
        branches_read.extend(cfg['trigger'].get(year, []))
        events = f_1.arrays(branches_read, library='ak')
        out = ak.concatenate((events, out))
    return out


def save_raw_data(data, data_type='MC'):
    if data_type not in ['MC', 'data']:
        raise ValueError("data_type must be either 'MC' or 'data'")

    # Determine run label from keys in the payload
    run_label = 'run2'
    try:
        if data_type == 'MC':
            # If any dataset key starts with mc23* it's Run 3
            if isinstance(data, dict) and any(str(k).startswith('mc23') for k in data.keys()):
                run_label = 'run3'
        else:
            # Data case: presence of data_22/23/24 implies Run 3
            if isinstance(data, dict) and any(k in data for k in ['data_22', 'data_23', 'data_24']):
                run_label = 'run3'
    except Exception:
        # Fall back to run2 if inspection fails
        run_label = 'run2'

    output_file = f"/pscratch/sd/a/agarabag/ditdau_samples/raw_{data_type.lower()}_data_test_{run_label}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    return data


# -----------------------------
# Data Selection helpers
# -----------------------------

def data_Cut(t, year, config=None):
    cfg = get_config('run3') if year in ['22', '23', '24'] else get_config('run2') if config is None else config
    # Build HLT mask from configured trigger names present in t
    hlt_list = cfg['trigger'].get(year, [])
    if not hlt_list:
        raise ValueError(f"Unsupported year: {year}")
    hlt_cut = None
    for trig in hlt_list:
        cur = t[trig]
        hlt_cut = cur if hlt_cut is None else (hlt_cut | cur)
    general_cut = ((t['ditau_obj_nSubjets'] >= 2) &
                   (((t['ditau_obj_leadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_leadsubjet_n_core_tracks'] == 3)) &
                    ((t['ditau_obj_subleadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_subleadsubjet_n_core_tracks'] == 3))) &
                   (t['ditau_obj_leadsubjet_charge'] * t['ditau_obj_subleadsubjet_charge'] == -1) &
                   (t['ditau_obj_omni_score'] < ditau_id_cut) &
                   (t['n_bjets_GN2v01_FixedCutBEff_70'] == 0) &
                   (t['ditau_obj_n_tracks'] - t['ditau_obj_leadsubjet_n_core_tracks'] - t['ditau_obj_subleadsubjet_n_core_tracks'] == 0))
    final_cut = ak.where(general_cut & (hlt_cut))
    return t[final_cut]


def mc_Cut(t, year, config=None):
    cfg = get_config('run3') if year in ['22', '23', '24'] else get_config('run2') if config is None else config
    if year in ['15', '16']:
        mask_15 = (t['NOMINAL_pileup_random_run_number'] <= 284484) & (t['NOMINAL_pileup_random_run_number'] > 0)
        mask_16 = (t['NOMINAL_pileup_random_run_number'] > 284484) & (t['NOMINAL_pileup_random_run_number'] <= 311563)
        hlt_list_15 = cfg['trigger']['15']
        hlt_list_16 = cfg['trigger']['16']
        hlt_cut_15 = None
        for trig in hlt_list_15:
            hlt_cut_15 = t[trig] if hlt_cut_15 is None else (hlt_cut_15 | t[trig])
        hlt_cut_16 = None
        for trig in hlt_list_16:
            hlt_cut_16 = t[trig] if hlt_cut_16 is None else (hlt_cut_16 | t[trig])
        hlt_cut = (mask_15 & hlt_cut_15) | (mask_16 & hlt_cut_16)
    else:
        hlt_list = cfg['trigger'].get(year, [])
        if not hlt_list:
            raise ValueError(f"Unsupported year: {year}")
        hlt_cut = None
        for trig in hlt_list:
            hlt_cut = t[trig] if hlt_cut is None else (hlt_cut | t[trig])
    general_cut = ((t['ditau_obj_nSubjets'] >= 2) &
                   (((t['ditau_obj_leadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_leadsubjet_n_core_tracks'] == 3)) &
                    ((t['ditau_obj_subleadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_subleadsubjet_n_core_tracks'] == 3))) &
                   (t['ditau_obj_IsTruthMatched'] == 1) &
                   (t['n_bjets_GN2v01_FixedCutBEff_70'] == 0) &
                   (t['ditau_obj_leadsubjet_charge'] * t['ditau_obj_subleadsubjet_charge'] == -1) &
                   (t['ditau_obj_omni_score'] >= ditau_id_cut) &
                   (t['ditau_obj_n_tracks'] - t['ditau_obj_leadsubjet_n_core_tracks'] - t['ditau_obj_subleadsubjet_n_core_tracks'] == 0))
    final_cut = ak.where(general_cut & (hlt_cut))
    return t[final_cut]


def apply_cuts(data, data_type='MC', config=None):
    if data_type not in ['MC', 'data']:
        raise ValueError("data_type must be either 'MC' or 'data'")
    cut_results = {}
    if data_type == 'MC':
        # Auto-detect run by dataset keys
        ds_keys = list(data.keys())
        auto_cfg = get_config('run3') if any(k.startswith('mc23') for k in ds_keys) else get_config('run2')
        cfg = auto_cfg if config is None else config
        for dataset, results in data.items():
            year = cfg['year_map'][dataset]
            cut_results[dataset] = {}
            for var_name, sample in results.items():
                try:
                    cut_results[dataset][var_name] = mc_Cut(sample, year, config=cfg)
                except Exception:
                    cut_results[dataset][var_name] = None
    else:
        # Auto-detect run by provided keys
        keys = list(data.keys())
        if any(k.startswith('data_') for k in keys):
            # Could be run2 (15-18) or run3 (22-24); inspect one key
            if any(k in keys for k in ['data_22', 'data_23', 'data_24']):
                cfg = get_config('run3') if config is None else config
                data_vars = {k: (k.split('_')[1], data[k]) for k in ['data_22', 'data_23', 'data_24'] if k in data}
            else:
                cfg = get_config('run2') if config is None else config
                data_vars = {k: (k.split('_')[1], data[k]) for k in ['data_18', 'data_17', 'data_16', 'data_15'] if k in data}
        else:
            cfg = get_config('run2') if config is None else config
            data_vars = {}
        for name, (year, sample) in data_vars.items():
            try:
                cut_results[name] = data_Cut(sample, year, config=cfg)
            except Exception:
                cut_results[name] = None
    return cut_results


def combine_mc_years(cut_mc):
    """Combine MC samples across available datasets (Run 2 or Run 3).

    Auto-detects dataset keys (e.g., mc20e/d/a or mc23a/d/e) and merges
    by base process name (before the final _ suffix).
    """
    combined = {}
    base_processes = set()
    # Collect base process names from whatever datasets are present
    for dataset_key, dataset in cut_mc.items():
        for proc in dataset.keys():
            base_name = proc.rsplit('_', 1)[0]
            base_processes.add(base_name)
    # Merge across all datasets we actually have
    for base_proc in base_processes:
        year_data = []
        for dataset_key, dataset in cut_mc.items():
            # dataset suffix is last char of dataset key (e, d, a) for mc20*, (a, d, e) for mc23*
            suffix = dataset_key[-1]
            proc_name = f"{base_proc}_{suffix}"
            if proc_name in dataset and dataset[proc_name] is not None:
                year_data.append(dataset[proc_name])
        if year_data:
            combined[base_proc] = ak.concatenate(year_data)
    return combined


def combine_data_years(cut_data):
    """Combine data across available years (Run 2 or Run 3).

    Accepts any subset of keys of the form 'data_XX' where XX in {15,16,17,18,22,23,24}.
    """
    # Pick up all data_* keys that actually exist and are non-None
    present_keys = [k for k in cut_data.keys() if k.startswith('data_') and cut_data[k] is not None]
    if not present_keys:
        return ak.Array([])
    # Sort by numeric suffix for deterministic order
    present_keys.sort(key=lambda k: int(k.split('_')[1]))
    data_arrays = [cut_data[k] for k in present_keys]
    combined_data = ak.concatenate(data_arrays)
    return combined_data


# -----------------------------
# Physics helpers and variable builders
# -----------------------------

def collinear_mass_calc(k1, k2, metetx, metety):
    k1_px = np.array(k1.px)
    k1_py = np.array(k1.py)
    k1_pz = np.array(k1.pz)
    k1_energy = np.array(k1.energy)
    k2_px = np.array(k2.px)
    k2_py = np.array(k2.py)
    k2_pz = np.array(k2.pz)
    k2_energy = np.array(k2.energy)
    metetx = np.array(metetx)
    metety = np.array(metety)
    collinear_mass = []
    x1_out = []
    x2_out = []
    for i in range(len(metetx)):
        K = np.array([[k1_px[i], k2_px[i]], [k1_py[i], k2_py[i]]])
        if np.linalg.det(K) == 0:
            return [0], [0], [0]
        M = np.array([[metetx[i]], [metety[i]]])
        Kinv = np.linalg.inv(K)
        X = np.dot(Kinv, M)
        X1 = X[0, 0]
        X2 = X[1, 0]
        x1 = 1. / (1. + X1)
        x2 = 1. / (1. + X2)
        p1 = vector.obj(px=k1_px[i], py=k1_py[i], pz=k1_pz[i], energy=k1_energy[i]) * (1 / x1)
        p2 = vector.obj(px=k2_px[i], py=k2_py[i], pz=k2_pz[i], energy=k2_energy[i]) * (1 / x2)
        collinear_mass.append((p1 + p2).mass)
        x1_out.append(x1)
        x2_out.append(x2)
    return collinear_mass, x1_out, x2_out


def determine_prong(subleadNTracks):
    if subleadNTracks == 1:
        return 1
    elif subleadNTracks == 3:
        return 2
    else:
        return -1


def load_histograms(root_file):
    file = uproot.open(root_file)
    histograms = {
        "h_ff_13p1p": file["FF_13p1p_ditau_obj_subleadsubjet_pt_ff"].to_hist(),
        "h_ff_13p3p": file["FF_13p3p_ditau_obj_subleadsubjet_pt_ff"].to_hist(),
    }
    return histograms


def fake_factor_calc(subleadNTracks, sublead_pt, histograms):
    weights = []
    for sublead, pt in zip(subleadNTracks, sublead_pt):
        prongness = determine_prong(sublead)
        if prongness == 1:
            hist = histograms["h_ff_13p1p"]
        elif prongness == 2:
            hist = histograms["h_ff_13p3p"]
        else:
            weights.append(-1)
            continue
        bin_edges = hist.axes[0].edges
        bin_idx = np.digitize(pt, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(hist.values()) - 1)
        weight = hist.values()[bin_idx]
        weights.append(weight)
    return [w * ff_scale for w in weights]


def met_centrality(lead_jet_phi, sublead_jet_phi, met_phi):
    lead_jet_phi_np = ak.to_numpy(lead_jet_phi)
    sublead_jet_phi_np = ak.to_numpy(sublead_jet_phi)
    met_phi_np = ak.to_numpy(met_phi)
    d = np.sin(sublead_jet_phi_np - lead_jet_phi_np)
    centrality = np.full_like(d, -1234.0)
    valid_indices = np.where(d != 0)
    A = np.sin(met_phi_np[valid_indices] - lead_jet_phi_np[valid_indices]) / d[valid_indices]
    B = np.sin(sublead_jet_phi_np[valid_indices] - met_phi_np[valid_indices]) / d[valid_indices]
    centrality[valid_indices] = (A + B) / np.sqrt(A * A + B * B)
    return centrality


def compute_vbf_features(t, leadsubjet_p4, subleadsubjet_p4, ditau_p4, met_2d):
    """Compute VBF-like jet features with ΔR(jet, ditau) > 0.4 selection.

    Returns a dict of arrays. Missing values are filled with -999.
    """
    # Select up to first 10 jets for stability
    jets_vec = vector.zip({
        'px': ak.Array([obj.fP.fX for obj in t['jets_p4'][:, :10]]),
        'py': ak.Array([obj.fP.fY for obj in t['jets_p4'][:, :10]]),
        'pz': ak.Array([obj.fP.fZ for obj in t['jets_p4'][:, :10]]),
        'energy': t['jets_p4'][:, :10].fE
    })
    # Keep jets well separated from the ditau system
    delta_r_jets = jets_vec.deltaR(ditau_p4)
    selected_mask = delta_r_jets > 0.4
    selected_jets = jets_vec[selected_mask]

    leading_jet = ak.firsts(selected_jets)
    subleading_jet = ak.firsts(selected_jets[:, 1:])

    leading_jet_pt = ak.fill_none(leading_jet.pt, -999)
    subleading_jet_pt = ak.fill_none(subleading_jet.pt, -999)
    valid_pair = (leading_jet_pt != -999) & (subleading_jet_pt != -999)
    valid_leading = (leading_jet_pt != -999)

    Ht = ak.fill_none(ak.sum(selected_jets.pt, axis=1), -999) + ditau_p4.pt
    eta_product = ak.where(valid_pair, leading_jet.eta * subleading_jet.eta, -999)
    delta_eta_jj = ak.where(valid_pair, np.abs(leading_jet.eta - subleading_jet.eta), -999)
    Mjj = ak.where(valid_pair, ak.fill_none((leading_jet + subleading_jet).mass, -999), -999)
    pt_jj = ak.where(valid_pair, ak.fill_none((leading_jet + subleading_jet).pt, -999), -999)
    delta_phi_jj = ak.where(valid_pair, ak.fill_none(leading_jet.deltaphi(subleading_jet), -999), -999)
    pt_jj_higgs = ak.where(valid_pair, ak.fill_none((leading_jet + subleading_jet + (met_2d + (leadsubjet_p4 + subleadsubjet_p4))).pt, -999), -999)
    delta_r_leadjet_ditau = ak.where(valid_leading, ak.fill_none(leading_jet.deltaR((met_2d + (leadsubjet_p4 + subleadsubjet_p4))), -999), -999)

    return {
        'Ht': Ht,
        'eta_product': eta_product,
        'delta_eta_jj': delta_eta_jj,
        'Mjj': Mjj,
        'pt_jj': pt_jj,
        'delta_phi_jj': delta_phi_jj,
        'pt_jj_higgs': pt_jj_higgs,
        'delta_r_leadjet_ditau': delta_r_leadjet_ditau,
        'leading_jet_pt': leading_jet_pt,
        'subleading_jet_pt': subleading_jet_pt,
    }


def Var(t, include_vbf=False):
    leadsubjet_p4 = vector.zip({'px': t['ditau_obj_leadsubjet_p4'].fP.fX,
                                'py': t['ditau_obj_leadsubjet_p4'].fP.fY,
                                'pz': t['ditau_obj_leadsubjet_p4'].fP.fZ,
                                'energy': t['ditau_obj_leadsubjet_p4'].fE})
    subleadsubjet_p4 = vector.zip({'px': t['ditau_obj_subleadsubjet_p4'].fP.fX,
                                   'py': t['ditau_obj_subleadsubjet_p4'].fP.fY,
                                   'pz': t['ditau_obj_subleadsubjet_p4'].fP.fZ,
                                   'energy': t['ditau_obj_subleadsubjet_p4'].fE})
    ditau_p4 = vector.zip({'px': t['ditau_obj_p4'].fP.fX,
                           'py': t['ditau_obj_p4'].fP.fY,
                           'pz': t['ditau_obj_p4'].fP.fZ,
                           'energy': t['ditau_obj_p4'].fE})
    delta_phi = leadsubjet_p4.deltaphi(subleadsubjet_p4)
    delta_eta = leadsubjet_p4.deltaeta(subleadsubjet_p4)
    delta_R = leadsubjet_p4.deltaR(subleadsubjet_p4)
    k_t = delta_R * subleadsubjet_p4.pt
    kappa = delta_R * (subleadsubjet_p4.pt / (subleadsubjet_p4.pt + leadsubjet_p4.pt))
    delta_R_lead = leadsubjet_p4.deltaR(ditau_p4)
    delta_eta_lead = leadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_lead = leadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_lead = leadsubjet_p4.energy / ditau_p4.energy
    delta_R_sublead = subleadsubjet_p4.deltaR(ditau_p4)
    delta_eta_sublead = subleadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_sublead = subleadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_sublead = subleadsubjet_p4.energy / ditau_p4.energy
    event_id = t['event_number']
    combined_weights = t['weight']
    visible_ditau_m = (leadsubjet_p4 + subleadsubjet_p4).mass
    met_2d = vector.zip({'px': t['met_p4'].fP.fX,
                         'py': t['met_p4'].fP.fY,
                         'pz': t['met_p4'].fP.fZ,
                         'energy': t['met_p4'].fE})
    met_pt = met_2d.pt
    met_phi = met_2d.phi
    k1 = leadsubjet_p4
    k2 = subleadsubjet_p4
    metetx = met_2d.px
    metety = met_2d.py
    collinear_mass, x1, x2 = collinear_mass_calc(k1, k2, metetx, metety)
    delta_phi_met_ditau = met_2d.deltaphi(leadsubjet_p4 + subleadsubjet_p4)
    met_sig = met_pt / 1000.0 / 0.5 / np.sqrt(t['met_sumet'] / 1000.0)
    fake_factor = np.ones(len(t['ditau_obj_leadsubjet_p4'].fP.fX))
    met_centrality_val = met_centrality(leadsubjet_p4.phi, subleadsubjet_p4.phi, met_phi)
    higgs_pt = (met_2d + (leadsubjet_p4 + subleadsubjet_p4)).pt
    base_out = [ditau_p4.pt, leadsubjet_p4.pt, subleadsubjet_p4.pt, visible_ditau_m, met_pt, collinear_mass, x1, x2,
                met_sig, met_phi, event_id, k_t, kappa, delta_R, delta_phi, delta_eta, combined_weights, fake_factor,
                delta_R_lead, delta_eta_lead, delta_phi_lead, delta_R_sublead, delta_eta_sublead, delta_phi_sublead,
                met_centrality_val, t.ditau_obj_omni_score, t.ditau_obj_leadsubjet_charge, t.ditau_obj_subleadsubjet_charge,
                t.ditau_obj_leadsubjet_n_core_tracks, t.ditau_obj_subleadsubjet_n_core_tracks, e_ratio_lead, e_ratio_sublead,
                higgs_pt, leadsubjet_p4.eta, subleadsubjet_p4.eta, ditau_p4.eta, delta_phi_met_ditau]
    if not include_vbf:
        return base_out
    vbf = compute_vbf_features(t, leadsubjet_p4, subleadsubjet_p4, ditau_p4, met_2d)
    return base_out + [
        vbf['Ht'], vbf['eta_product'], vbf['delta_eta_jj'], vbf['Mjj'], vbf['pt_jj'], vbf['delta_phi_jj'],
        vbf['pt_jj_higgs'], vbf['delta_r_leadjet_ditau'], vbf['leading_jet_pt'], vbf['subleading_jet_pt']
    ]


def Data_Var(t, config=None, include_vbf=False):
    cfg = get_config('run2') if config is None else config
    leadsubjet_p4 = vector.zip({'px': t['ditau_obj_leadsubjet_p4'].fP.fX,
                                'py': t['ditau_obj_leadsubjet_p4'].fP.fY,
                                'pz': t['ditau_obj_leadsubjet_p4'].fP.fZ,
                                'energy': t['ditau_obj_leadsubjet_p4'].fE})
    subleadsubjet_p4 = vector.zip({'px': t['ditau_obj_subleadsubjet_p4'].fP.fX,
                                   'py': t['ditau_obj_subleadsubjet_p4'].fP.fY,
                                   'pz': t['ditau_obj_subleadsubjet_p4'].fP.fZ,
                                   'energy': t['ditau_obj_subleadsubjet_p4'].fE})
    ditau_p4 = vector.zip({'px': t['ditau_obj_p4'].fP.fX,
                           'py': t['ditau_obj_p4'].fP.fY,
                           'pz': t['ditau_obj_p4'].fP.fZ,
                           'energy': t['ditau_obj_p4'].fE})
    delta_phi = leadsubjet_p4.deltaphi(subleadsubjet_p4)
    delta_eta = leadsubjet_p4.deltaeta(subleadsubjet_p4)
    delta_R = leadsubjet_p4.deltaR(subleadsubjet_p4)
    k_t = delta_R * subleadsubjet_p4.pt
    kappa = delta_R * (subleadsubjet_p4.pt / (subleadsubjet_p4.pt + leadsubjet_p4.pt))
    delta_R_lead = leadsubjet_p4.deltaR(ditau_p4)
    delta_eta_lead = leadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_lead = leadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_lead = leadsubjet_p4.energy / ditau_p4.energy
    delta_R_sublead = subleadsubjet_p4.deltaR(ditau_p4)
    delta_eta_sublead = subleadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_sublead = subleadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_sublead = subleadsubjet_p4.energy / ditau_p4.energy
    visible_ditau_m = (leadsubjet_p4 + subleadsubjet_p4).mass
    event_id = t['event_number']
    histograms = load_histograms(cfg['ff_hist_file'])
    leadNTracks = np.array(t.ditau_obj_subleadsubjet_n_core_tracks)
    subleadNTracks = np.array(t.ditau_obj_leadsubjet_n_core_tracks)
    lead_pt = np.array(leadsubjet_p4.pt)
    sublead_pt = np.array(subleadsubjet_p4.pt)
    delta_r = np.array(delta_R)
    fake_factor = fake_factor_calc(subleadNTracks, sublead_pt, histograms)
    met_2d = vector.zip({'px': t['met_p4'].fP.fX,
                         'py': t['met_p4'].fP.fY,
                         'pz': t['met_p4'].fP.fZ,
                         'energy': t['met_p4'].fE})
    met_pt = met_2d.pt
    met_phi = met_2d.phi
    k1 = leadsubjet_p4
    k2 = subleadsubjet_p4
    metetx = met_2d.px
    metety = met_2d.py
    collinear_mass, x1, x2 = collinear_mass_calc(k1, k2, metetx, metety)
    delta_phi_met_ditau = met_2d.deltaphi(leadsubjet_p4 + subleadsubjet_p4)
    met_sig = met_pt / 1000.0 / 0.5 / np.sqrt(t['met_sumet'] / 1000.0)
    combined_weights = np.ones(len(t['ditau_obj_leadsubjet_p4'].fP.fX))
    met_centrality_val = met_centrality(leadsubjet_p4.phi, subleadsubjet_p4.phi, met_phi)
    higgs_pt = (met_2d + (leadsubjet_p4 + subleadsubjet_p4)).pt
    base_out = [ditau_p4.pt, leadsubjet_p4.pt, subleadsubjet_p4.pt, visible_ditau_m, met_pt, collinear_mass, x1, x2, met_sig, met_phi, event_id, k_t, kappa, delta_R, delta_phi, delta_eta, combined_weights, fake_factor,
                delta_R_lead, delta_eta_lead, delta_phi_lead, delta_R_sublead, delta_eta_sublead, delta_phi_sublead,
                met_centrality_val, t.ditau_obj_omni_score, t.ditau_obj_leadsubjet_charge, t.ditau_obj_subleadsubjet_charge, t.ditau_obj_leadsubjet_n_core_tracks,
                t.ditau_obj_subleadsubjet_n_core_tracks, e_ratio_lead, e_ratio_sublead,
                higgs_pt, leadsubjet_p4.eta, subleadsubjet_p4.eta, ditau_p4.eta, delta_phi_met_ditau]
    if not include_vbf:
        return base_out
    vbf = compute_vbf_features(t, leadsubjet_p4, subleadsubjet_p4, ditau_p4, met_2d)
    return base_out + [
        vbf['Ht'], vbf['eta_product'], vbf['delta_eta_jj'], vbf['Mjj'], vbf['pt_jj'], vbf['delta_phi_jj'],
        vbf['pt_jj_higgs'], vbf['delta_r_leadjet_ditau'], vbf['leading_jet_pt'], vbf['subleading_jet_pt']
    ]


def event_cleaning(t):
    cut_mask = np.where((np.array(t[6]) > 0) & (np.array(t[7]) > 0) & (np.array(t[6]) < 2) & (np.array(t[7]) < 2) &
                        (np.array(t[1]) > 50) & (np.array(t[2]) > 15) & (np.array(t[4]) > 60) & (np.array(t[16]) > 0))[0]
    filtered_t = [np.array(arr)[cut_mask] for arr in t]
    return filtered_t


# -----------------------------
# Plot helpers
# -----------------------------

def plot_variable_distributions(plot_configs, signal_processes, background_processes):
    for var, (xmin, xmax, nbins) in plot_configs.items():
        plt.figure(figsize=(8, 6))
        plt_bins = np.linspace(xmin, xmax, nbins)
        if var == 'delta_phi_met_ditau_normalized':
            combined_signal = np.concatenate([(process['delta_phi_met_ditau'] + np.pi) / 2*np.pi for process in signal_processes])
            combined_background = np.concatenate([(process['delta_phi_met_ditau'] + np.pi) / 2*np.pi for process in background_processes])
        else:
            combined_signal = np.concatenate([process[var] for process in signal_processes])
            combined_background = np.concatenate([process[var] for process in background_processes])
        plt.hist(combined_signal, bins=plt_bins, histtype='stepfilled', alpha=0.3, label='Combined Signal', color='red', density=True)
        plt.hist(combined_background, bins=plt_bins, histtype='stepfilled', alpha=0.3, label='Combined Background', color='blue', density=True)
        plt.legend(fontsize=14)
        plt.grid(True)
        if var == 'delta_phi_met_ditau_normalized':
            plt.xlabel('(Δφ(MET, DiTau) + π) / π', fontsize=22)
        else:
            plt.xlabel(var, fontsize=22)
        plt.ylabel('Events', fontsize=22)
        plt.show()


