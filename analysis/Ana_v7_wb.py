import glob, os, sys
sys.path.append("..")
from utils.utils import *
import uproot, ROOT, random, time, vector
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import json
from sklearn.metrics import roc_curve, roc_auc_score, auc, log_loss
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from xgboost import XGBClassifier 
import wandb
import pickle



ditau_id_cut = 0.999973

def data_Cut(t, year):
    # Define HLT conditions based on the year
    if year == '15':
        hlt_cut = t['HLT_j360']
    elif year == '16':
        hlt_cut = t['HLT_j380'] | t['HLT_j420_a10_lcw_L1J100'] | t['HLT_j420_a10r_L1J100']
    elif year == '17':
        hlt_cut = t['HLT_j400'] | t['HLT_j440_a10_lcw_subjes_L1J100']
    elif year == '18':
        hlt_cut = t['HLT_j420'] | t['HLT_j420_a10t_lcw_jes_35smcINF_L1J100'] | t['HLT_j420_a10t_lcw_jes_35smcINF_L1SC111']
    else:
        raise ValueError(f"Unsupported year: {year}")
    
    general_cut = ((t['ditau_obj_nSubjets'] >= 2) & \
               (((t['ditau_obj_leadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_leadsubjet_n_core_tracks'] == 3)) & 
                ((t['ditau_obj_subleadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_subleadsubjet_n_core_tracks'] == 3))) & \
               (t['ditau_obj_leadsubjet_charge'] * t['ditau_obj_subleadsubjet_charge'] == -1) & \
               (t['ditau_obj_omni_score'] < ditau_id_cut) & \
               (t['n_bjets_DL1dv01_FixedCutBEff_70'] == 0) & \
               (t['ditau_obj_n_tracks'] - t['ditau_obj_leadsubjet_n_core_tracks'] - t['ditau_obj_subleadsubjet_n_core_tracks'] == 0))

    final_cut = ak.where(general_cut & (hlt_cut))
    # final_cut = ak.where(general_cut)
    return t[final_cut]

def mc_Cut(t, year):
    # Define HLT conditions based on the year and run number for 2015 and 2016
    if year == '15' or year == '16':
        mask_15 = (t['NOMINAL_pileup_random_run_number'] <= 284484) & (t['NOMINAL_pileup_random_run_number'] > 0)
        mask_16 = (t['NOMINAL_pileup_random_run_number'] > 284484) & (t['NOMINAL_pileup_random_run_number'] <= 311563)

        hlt_cut_15 = t['HLT_j360']
        hlt_cut_16 = t['HLT_j380'] | t['HLT_j420_a10_lcw_L1J100'] | t['HLT_j420_a10r_L1J100']

        hlt_cut = (mask_15 & hlt_cut_15) | (mask_16 & hlt_cut_16)

    elif year == '17':
        hlt_cut = t['HLT_j400'] | t['HLT_j440_a10_lcw_subjes_L1J100']

    elif year == '18':
        hlt_cut = t['HLT_j420'] | t['HLT_j420_a10t_lcw_jes_35smcINF_L1J100'] | t['HLT_j420_a10t_lcw_jes_35smcINF_L1SC111']
    else:
        raise ValueError(f"Unsupported year: {year}")

    general_cut = ((t['ditau_obj_nSubjets'] >= 2) & \
               (((t['ditau_obj_leadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_leadsubjet_n_core_tracks'] == 3)) & 
                ((t['ditau_obj_subleadsubjet_n_core_tracks'] == 1) | (t['ditau_obj_subleadsubjet_n_core_tracks'] == 3))) & \
               (t['ditau_obj_IsTruthMatched']==1) & \
               (t['n_bjets_DL1dv01_FixedCutBEff_70'] == 0) & \
               (t['ditau_obj_leadsubjet_charge'] * t['ditau_obj_subleadsubjet_charge'] == -1) & \
               (t['ditau_obj_omni_score'] >= ditau_id_cut) & \
               (t['ditau_obj_n_tracks'] - t['ditau_obj_leadsubjet_n_core_tracks'] - t['ditau_obj_subleadsubjet_n_core_tracks'] == 0))

    # Combine HLT-specific cuts with general cuts
    final_cut = ak.where(general_cut & (hlt_cut))
    # final_cut = ak.where(general_cut)
    return t[final_cut]

def apply_cuts(data, data_type='MC'):
    """
    Apply cuts to both MC and real data
    data_type: 'MC' or 'data'
    """
    if data_type not in ['MC', 'data']:
        raise ValueError("data_type must be either 'MC' or 'data'")
    
    cut_results = {}
    
    if data_type == 'MC':
        year_map = {
        'mc20e': '18',
        'mc20d': '17', 
        'mc20a': '15'
        }
        # Handle MC data structure (nested dictionary)
        for dataset, results in data.items():
            year = year_map[dataset]
            cut_results[dataset] = {}
            
            print(f"Applying MC cuts to {dataset} (year {year})...")
            for var_name, sample in results.items():
                try:
                    cut_results[dataset][var_name] = mc_Cut(sample, year)
                    print(f"Processed {var_name}")
                except Exception as e:
                    print(f"Error processing {var_name}: {str(e)}")
                    cut_results[dataset][var_name] = None
    
    else:
        # Handle real data (direct samples)
        data_vars = {
            'data_18': ('18', data['data_18']),
            'data_17': ('17', data['data_17']),
            'data_16': ('16', data['data_16']),
            'data_15': ('15', data['data_15'])
        }
        
        for name, (year, sample) in data_vars.items():
            print(name, year, sample)
            try:
                cut_results[name] = data_Cut(sample, year)
                print(f"Processed {name}")
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                cut_results[name] = None

    return cut_results

with open('raw_mc_data.pkl', 'rb') as f:
    uncut_mc = pickle.load(f)

with open('raw_data_data.pkl', 'rb') as f:
    uncut_data = pickle.load(f)

cut_data = apply_cuts({'data_18': uncut_data['data_18'], 'data_17': uncut_data['data_17'], 
                       'data_16': uncut_data['data_16'], 'data_15': uncut_data['data_15']}, data_type='data')
cut_mc = apply_cuts(uncut_mc, data_type='MC')

def combine_mc_years(cut_mc):    
    combined = {}
    # Get base process names (without year suffix)
    base_processes = set()
    for dataset in cut_mc.values():
        for proc in dataset.keys():
            base_name = proc.rsplit('_', 1)[0]  # Remove _e, _d, _a suffix
            base_processes.add(base_name)
    
    # Combine each process across years
    for base_proc in base_processes:
        year_data = []
        for dataset in ['mc20e', 'mc20d', 'mc20a']:
            proc_name = f"{base_proc}_{dataset[-1]}"
            if proc_name in cut_mc[dataset]:
                year_data.append(cut_mc[dataset][proc_name])
        
        # Combine using awkward array concatenation
        if year_data:
            combined[base_proc] = ak.concatenate(year_data)
            print(f"Combined {base_proc} across years")
    
    return combined

def combine_data_years(cut_data):    
    # Extract arrays from all years
    data_arrays = [
        cut_data['data_18'],
        cut_data['data_17'],
        cut_data['data_16'],
        cut_data['data_15']
    ]
    
    # Combine using awkward array concatenation
    combined_data = ak.concatenate(data_arrays)
    print(f"Combined data from all years: {len(combined_data)} events")
    
    return combined_data

combined_mc = combine_mc_years(cut_mc)
combines_data = combine_data_years(cut_data)

#combine all mc the samples
vbfh_cut = combined_mc['VBFH']
ggh_cut = combined_mc['ggH']
wh_cut = combined_mc['WH']
zh_cut = combined_mc['ZH']
tth_cut = combined_mc['ttH']
ztt_inc_cut = combined_mc['Ztt_inc']
ttv_cut = combined_mc['ttV']
vv_cut = combined_mc['VV']
top_cut = combined_mc['Top']
w_cut = combined_mc['W']
zll_inc_cut = combined_mc['Zll_inc']

#combine all data samples
data_cut = combines_data

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
        K = np.array([[k1_px[i], k2_px[i]],
                      [k1_py[i], k2_py[i]]])
        if np.linalg.det(K) == 0:
            print("WARNING: Singular matrix")
            return 0
        M = np.array([[metetx[i]],
                      [metety[i]]])
        
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

ff_scale = 1

def determine_prong(leadNTracks, subleadNTracks):
    if leadNTracks == 1 and subleadNTracks == 1:
        return 1
    elif leadNTracks == 3 and subleadNTracks == 1:
        return 2
    elif leadNTracks == 1 and subleadNTracks == 3:
        return 2
    elif leadNTracks == 3 and subleadNTracks == 3:
        return 3
    else:
        return -1

def load_histograms(root_file):
    """Load 3D fake factor histograms"""
    file = uproot.open(root_file)
    histograms = {
        "h_ff_1p1p": file["2D_FF_1p1p"].to_hist(),  # Assuming these are the 3D histogram names
        "h_ff_1p3p": file["2D_FF_1p3p"].to_hist(),
        "h_ff_3p3p": file["2D_FF_3p3p"].to_hist()
    }
    return histograms

def fake_factor_calc(leadNTracks, subleadNTracks, sublead_pt, delta_r, histograms):
    """Calculate fake factors using 3D histograms with axes: (delta_r, lead_pt, sublead_pt)"""
    weights = []
    
    for lead, sublead, s_pt, dr in zip(leadNTracks, subleadNTracks, sublead_pt, delta_r):
        prongness = determine_prong(lead, sublead)
        weight = 0
        
        try:
            if prongness == 1:
                hist = histograms["h_ff_1p1p"]
                # Reordered indices to match histogram structure
                delta_r_idx = hist.axes[0].index(dr)
                sublead_idx = hist.axes[1].index(s_pt)
                weight = hist.values()[delta_r_idx, sublead_idx]
                
            elif prongness == 2:
                hist = histograms["h_ff_1p3p"]
                delta_r_idx = hist.axes[0].index(dr)
                sublead_idx = hist.axes[1].index(s_pt)
                weight = hist.values()[delta_r_idx, sublead_idx]
                
            elif prongness == 3:
                hist = histograms["h_ff_3p3p"]
                delta_r_idx = hist.axes[0].index(dr)
                sublead_idx = hist.axes[1].index(s_pt)
                weight = hist.values()[delta_r_idx, sublead_idx]
                
            else:
                print(f"Warning: Invalid prongness value: {prongness}")
                weight = -888
                
        except IndexError:
            print(f"Warning: Value out of histogram bounds: delta_r={dr}, sublead_pt={s_pt}")
            weight = -999

        if weight < 0:
            print(f"Warning: Negative weight: {weight}")
            weight = 1
            
        weights.append(weight)
    
    return [w * ff_scale for w in weights]

def met_centrality(lead_jet_phi, sublead_jet_phi, met_phi):
    # Convert Awkward Arrays to NumPy for calculation
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

def Var(t):
    leadsubjet_p4 = vector.zip(
        {
        'px': t['ditau_obj_leadsubjet_p4'].fP.fX,
        'py':t['ditau_obj_leadsubjet_p4'].fP.fY,
        'pz':t['ditau_obj_leadsubjet_p4'].fP.fZ,
        'energy':t['ditau_obj_leadsubjet_p4'].fE
        }
    )
    subleadsubjet_p4 = vector.zip(
        {
        'px':t['ditau_obj_subleadsubjet_p4'].fP.fX,
        'py':t['ditau_obj_subleadsubjet_p4'].fP.fY,
        'pz':t['ditau_obj_subleadsubjet_p4'].fP.fZ,
        'energy':t['ditau_obj_subleadsubjet_p4'].fE
        }
    )
    ditau_p4 = vector.zip(
        {
        'px':t['ditau_obj_p4'].fP.fX,
        'py':t['ditau_obj_p4'].fP.fY,
        'pz':t['ditau_obj_p4'].fP.fZ,
        'energy':t['ditau_obj_p4'].fE
        }
    )
    delta_phi = leadsubjet_p4.deltaphi(subleadsubjet_p4)
    delta_eta = leadsubjet_p4.deltaeta(subleadsubjet_p4)
    delta_R = leadsubjet_p4.deltaR(subleadsubjet_p4)
    k_t = delta_R*subleadsubjet_p4.pt
    kappa = delta_R*(subleadsubjet_p4.pt/(subleadsubjet_p4.pt+leadsubjet_p4.pt))
    
    delta_R_lead = leadsubjet_p4.deltaR(ditau_p4)
    delta_eta_lead = leadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_lead = leadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_lead = leadsubjet_p4.energy/ditau_p4.energy

    delta_R_sublead = subleadsubjet_p4.deltaR(ditau_p4)
    delta_eta_sublead = subleadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_sublead = subleadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_sublead = subleadsubjet_p4.energy/ditau_p4.energy

    event_id = t['event_number']

    combined_weights = t['weight']
    
    # visible_ditau_m = t['ditau_obj_mvis_recalc']
    visible_ditau_m = (leadsubjet_p4 + subleadsubjet_p4).mass    

    #caulate missing pt
    met_2d = vector.zip(
        {
        'px': t['met_hpto_p4'].fP.fX, 
        'py': t['met_hpto_p4'].fP.fY,
        'pz': t['met_hpto_p4'].fP.fZ,
        'energy': t['met_hpto_p4'].fE
        }
    ) 
    met_pt = met_2d.pt
    met_phi = met_2d.phi
    ######
    k1 = leadsubjet_p4
    k2 = subleadsubjet_p4
    metetx = met_2d.px
    metety = met_2d.py
    collinear_mass, x1, x2 = collinear_mass_calc(k1, k2, metetx, metety)
    ######

    #delta phi between met and ditau
    delta_phi_met_ditau = met_2d.deltaphi(ditau_p4)

    met_sig = met_pt / 1000.0 / 0.5 / np.sqrt(t['met_sumet'] / 1000.0)
    
    fake_factor = np.ones(len(t['ditau_obj_leadsubjet_p4'].fP.fX))

    met_centrality_val = met_centrality(leadsubjet_p4.phi, subleadsubjet_p4.phi, met_phi)

    higgs_pt = (met_2d + ditau_p4).pt

    return [ditau_p4.pt, leadsubjet_p4.pt, subleadsubjet_p4.pt, visible_ditau_m, met_pt, collinear_mass, x1, x2,
            met_sig, met_phi, event_id, k_t, kappa, delta_R, delta_phi, delta_eta, combined_weights, fake_factor, 
            delta_R_lead, delta_eta_lead, delta_phi_lead, delta_R_sublead, delta_eta_sublead, delta_phi_sublead,
            met_centrality_val, t.ditau_obj_omni_score, t.ditau_obj_leadsubjet_charge, t.ditau_obj_subleadsubjet_charge, 
            t.ditau_obj_leadsubjet_n_core_tracks, t.ditau_obj_subleadsubjet_n_core_tracks, e_ratio_lead, e_ratio_sublead,
            higgs_pt, leadsubjet_p4.eta, subleadsubjet_p4.eta, ditau_p4.eta, delta_phi_met_ditau]

def Data_Var(t):
    leadsubjet_p4 = vector.zip(
        {
        'px': t['ditau_obj_leadsubjet_p4'].fP.fX,
        'py':t['ditau_obj_leadsubjet_p4'].fP.fY,
        'pz':t['ditau_obj_leadsubjet_p4'].fP.fZ,
        'energy':t['ditau_obj_leadsubjet_p4'].fE
        }
    )
    subleadsubjet_p4 = vector.zip(
        {
        'px':t['ditau_obj_subleadsubjet_p4'].fP.fX,
        'py':t['ditau_obj_subleadsubjet_p4'].fP.fY,
        'pz':t['ditau_obj_subleadsubjet_p4'].fP.fZ,
        'energy':t['ditau_obj_subleadsubjet_p4'].fE
        }
    )
    ditau_p4 = vector.zip(
        {
        'px':t['ditau_obj_p4'].fP.fX,
        'py':t['ditau_obj_p4'].fP.fY,
        'pz':t['ditau_obj_p4'].fP.fZ,
        'energy':t['ditau_obj_p4'].fE
        }
    )
    delta_phi = leadsubjet_p4.deltaphi(subleadsubjet_p4)
    delta_eta = leadsubjet_p4.deltaeta(subleadsubjet_p4)
    delta_R = leadsubjet_p4.deltaR(subleadsubjet_p4)
    k_t = delta_R*subleadsubjet_p4.pt
    kappa = delta_R*(subleadsubjet_p4.pt/(subleadsubjet_p4.pt+leadsubjet_p4.pt))
    
    delta_R_lead = leadsubjet_p4.deltaR(ditau_p4)
    delta_eta_lead = leadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_lead = leadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_lead = leadsubjet_p4.energy/ditau_p4.energy

    delta_R_sublead = subleadsubjet_p4.deltaR(ditau_p4)
    delta_eta_sublead = subleadsubjet_p4.deltaeta(ditau_p4)
    delta_phi_sublead = subleadsubjet_p4.deltaphi(ditau_p4)
    e_ratio_sublead = subleadsubjet_p4.energy/ditau_p4.energy

    visible_ditau_m = (leadsubjet_p4 + subleadsubjet_p4).mass    

    event_id = t['event_number']
    
    ######
    histograms = load_histograms("/home/agarabag/ditau_analysis/boom/data/fake_factors/FF_hadhad_ratio_3d.root")

    leadNTracks = np.array(t.ditau_obj_subleadsubjet_n_core_tracks)
    subleadNTracks = np.array(t.ditau_obj_leadsubjet_n_core_tracks)
    lead_pt = np.array(leadsubjet_p4.pt)
    sublead_pt = np.array(subleadsubjet_p4.pt)
    delta_r = np.array(delta_R)
    fake_factor = fake_factor_calc(leadNTracks, subleadNTracks, lead_pt, sublead_pt, delta_r, histograms) ###3d
    # fake_factor = fake_factor_calc(leadNTracks, subleadNTracks, sublead_pt, delta_r, histograms) ###2d

    ######
    ######
    met_2d = vector.zip(
        {
        'px': t['met_hpto_p4'].fP.fX, 
        'py': t['met_hpto_p4'].fP.fY,
        'pz': t['met_hpto_p4'].fP.fZ,
        'energy': t['met_hpto_p4'].fE
        }
    ) 
    met_pt = met_2d.pt
    met_phi = met_2d.phi
    k1 = leadsubjet_p4
    k2 = subleadsubjet_p4
    metetx = met_2d.px
    metety = met_2d.py
    collinear_mass, x1, x2 = collinear_mass_calc(k1, k2, metetx, metety)
    ######

    #delta phi between met and ditau
    delta_phi_met_ditau = met_2d.deltaphi(ditau_p4)

    met_sig = met_pt / 1000.0 / 0.5 / np.sqrt(t['met_sumet'] / 1000.0)

    combined_weights = np.ones(len(t['ditau_obj_leadsubjet_p4'].fP.fX))

    met_centrality_val = met_centrality(leadsubjet_p4.phi, subleadsubjet_p4.phi, met_phi)

    higgs_pt = (met_2d + ditau_p4).pt

    return [ditau_p4.pt, leadsubjet_p4.pt, subleadsubjet_p4.pt, visible_ditau_m, met_pt, collinear_mass, x1, x2, met_sig, met_phi, event_id, k_t, kappa, delta_R, delta_phi, delta_eta, combined_weights, fake_factor, 
            delta_R_lead, delta_eta_lead, delta_phi_lead, delta_R_sublead, delta_eta_sublead, delta_phi_sublead,
            met_centrality_val, t.ditau_obj_omni_score, t.ditau_obj_leadsubjet_charge, t.ditau_obj_subleadsubjet_charge, t.ditau_obj_leadsubjet_n_core_tracks, 
            t.ditau_obj_subleadsubjet_n_core_tracks, e_ratio_lead, e_ratio_sublead,
            higgs_pt, leadsubjet_p4.eta, subleadsubjet_p4.eta, ditau_p4.eta, delta_phi_met_ditau]

def Data_Var(t):
    leadsubjet_p4 = vector.obj(px=t['ditau_obj_leadsubjet_p4'].fP.fX,
                           py=t['ditau_obj_leadsubjet_p4'].fP.fY,
                           pz=t['ditau_obj_leadsubjet_p4'].fP.fZ,
                           energy=t['ditau_obj_leadsubjet_p4'].fE)                           
    subleadsubjet_p4 = vector.obj(px=t['ditau_obj_subleadsubjet_p4'].fP.fX,
                           py=t['ditau_obj_subleadsubjet_p4'].fP.fY,
                           pz=t['ditau_obj_subleadsubjet_p4'].fP.fZ,
                           energy=t['ditau_obj_subleadsubjet_p4'].fE)
    ditau_p4 = vector.obj(px=t['ditau_obj_p4'].fP.fX,
                          py=t['ditau_obj_p4'].fP.fY,
                          pz=t['ditau_obj_p4'].fP.fZ,
                          energy=t['ditau_obj_p4'].fE)
                          
    delta_R = vector.obj(pt=leadsubjet_p4.pt, phi=leadsubjet_p4.phi, eta=leadsubjet_p4.eta).deltaR(vector.obj(pt=subleadsubjet_p4.pt, phi=subleadsubjet_p4.phi, eta=subleadsubjet_p4.eta))
    delta_phi = vector.obj(pt=leadsubjet_p4.pt, phi=leadsubjet_p4.phi, eta=leadsubjet_p4.eta).deltaphi(vector.obj(pt=subleadsubjet_p4.pt, phi=subleadsubjet_p4.phi, eta=subleadsubjet_p4.eta))
    delta_eta = vector.obj(pt=leadsubjet_p4.pt, phi=leadsubjet_p4.phi, eta=leadsubjet_p4.eta).deltaeta(vector.obj(pt=subleadsubjet_p4.pt, phi=subleadsubjet_p4.phi, eta=subleadsubjet_p4.eta))
    k_t = delta_R*subleadsubjet_p4.pt
    kappa = delta_R*(subleadsubjet_p4.pt/(subleadsubjet_p4.pt+leadsubjet_p4.pt))
    # visible_ditau_m = t['ditau_obj_mvis_recalc']  
    visible_ditau_m = (leadsubjet_p4 + subleadsubjet_p4).mass 

    delta_R_lead = vector.obj(pt=leadsubjet_p4.pt, phi=leadsubjet_p4.phi, eta=leadsubjet_p4.eta).deltaR(vector.obj(pt=ditau_p4.pt, phi=ditau_p4.phi, eta=ditau_p4.eta))
    delta_eta_lead = vector.obj(pt=leadsubjet_p4.pt, phi=leadsubjet_p4.phi, eta=leadsubjet_p4.eta).deltaeta(vector.obj(pt=ditau_p4.pt, phi=ditau_p4.phi, eta=ditau_p4.eta))
    delta_phi_lead = vector.obj(pt=leadsubjet_p4.pt, phi=leadsubjet_p4.phi, eta=leadsubjet_p4.eta).deltaphi(vector.obj(pt=ditau_p4.pt, phi=ditau_p4.phi, eta=ditau_p4.eta))
    e_ratio_lead = leadsubjet_p4.energy/ditau_p4.energy

    delta_R_sublead = vector.obj(pt=subleadsubjet_p4.pt, phi=subleadsubjet_p4.phi, eta=subleadsubjet_p4.eta).deltaR(vector.obj(pt=ditau_p4.pt, phi=ditau_p4.phi, eta=ditau_p4.eta))
    delta_eta_sublead = vector.obj(pt=subleadsubjet_p4.pt, phi=subleadsubjet_p4.phi, eta=subleadsubjet_p4.eta).deltaeta(vector.obj(pt=ditau_p4.pt, phi=ditau_p4.phi, eta=ditau_p4.eta))
    delta_phi_sublead = vector.obj(pt=subleadsubjet_p4.pt, phi=subleadsubjet_p4.phi, eta=subleadsubjet_p4.eta).deltaphi(vector.obj(pt=ditau_p4.pt, phi=ditau_p4.phi, eta=ditau_p4.eta))
    e_ratio_sublead = subleadsubjet_p4.energy/ditau_p4.energy

    met_2d = vector.obj(px=t['met_hpto_p4'].fP.fX, py=t['met_hpto_p4'].fP.fY)  
    met_pt = np.sqrt(met_2d.px**2 + met_2d.py**2)
    met_phi = met_2d.phi

    event_id = t['event_number']
    ######
    histograms = load_histograms("FF_hadhad_ratio_2d_wp999973.root")
    leadNTracks = np.array(t.ditau_obj_subleadsubjet_n_core_tracks)
    subleadNTracks = np.array(t.ditau_obj_leadsubjet_n_core_tracks)
    lead_pt = np.array(leadsubjet_p4.pt)
    sublead_pt = np.array(subleadsubjet_p4.pt)
    delta_r = np.array(delta_R)
    #fake_factor = fake_factor_calc(leadNTracks, subleadNTracks, lead_pt, sublead_pt, delta_r, histograms) ###3d
    fake_factor = fake_factor_calc(leadNTracks, subleadNTracks, sublead_pt, delta_r, histograms) ###2d

    ######
    ######
    k1 = leadsubjet_p4
    k2 = subleadsubjet_p4
    metetx = met_2d.px
    metety = met_2d.py
    collinear_mass, x1, x2 = collinear_mass_calc(k1, k2, metetx, metety)
    ######

    #delta phi between met and ditau
    delta_phi_met_ditau = vector.obj(pt=met_pt, phi=met_phi, eta=0).deltaphi(vector.obj(pt=ditau_p4.pt, phi=ditau_p4.phi, eta=ditau_p4.eta))

    met_sig = met_pt / 1000.0 / 0.5 / np.sqrt(t['met_sumet'] / 1000.0)

    combined_weights = np.ones(len(t['ditau_obj_leadsubjet_p4'].fP.fX))

    met_centrality_val = met_centrality(leadsubjet_p4.phi, subleadsubjet_p4.phi, met_phi)

    higgs_pt = (met_2d + ditau_p4).pt

    return [ditau_p4.pt, leadsubjet_p4.pt, subleadsubjet_p4.pt, visible_ditau_m, met_pt, collinear_mass, x1, x2, met_sig, met_phi, event_id, k_t, kappa, delta_R, delta_phi, delta_eta, combined_weights, fake_factor, 
            delta_R_lead, delta_eta_lead, delta_phi_lead, delta_R_sublead, delta_eta_sublead, delta_phi_sublead,
            met_centrality_val, t.ditau_obj_omni_score, t.ditau_obj_leadsubjet_charge, t.ditau_obj_subleadsubjet_charge, t.ditau_obj_leadsubjet_n_core_tracks, 
            t.ditau_obj_subleadsubjet_n_core_tracks, e_ratio_lead, e_ratio_sublead,
            higgs_pt, leadsubjet_p4.eta, subleadsubjet_p4.eta, ditau_p4.eta, delta_phi_met_ditau]

def cut_x1_x2(t):
    cut_mask = np.where((np.array(t[6]) > 0) & (np.array(t[7]) > 0) & (np.array(t[6]) < 2) & (np.array(t[7]) < 2) & (np.array(t[16]) > 0) & (np.array(t[1]) > 50) & (np.array(t[2]) > 15))[0]
    filtered_t = [np.array(arr)[cut_mask] for arr in t]
    return filtered_t

calc_vars = ['ditau_pt', 'leadsubjet_pt', 'subleadsubjet_pt', 'visible_ditau_m', 'met', 'collinear_mass', 'x1', 'x2', 'met_sig', 'met_phi', 'event_number', 'k_t', 'kappa', 'delta_R', 
             'delta_phi', 'delta_eta', 'combined_weights', 'fake_factor', 'delta_R_lead', 'delta_eta_lead', 'delta_phi_lead', 'delta_R_sublead', 'delta_eta_sublead', 'delta_phi_sublead',
             'met_centrality', 'omni_score', 'leadsubjet_charge', 'subleadsubjet_charge', 'leadsubjet_n_core_tracks', 'subleadsubjet_n_core_tracks', 'e_ratio_lead', 'e_ratio_sublead',
             'higgs_pt', 'leadsubjet_eta', 'subleadsubjet_eta', 'ditau_eta', 'delta_phi_met_ditau']

ggh_plot = Var(ggh_cut)
vbfh_plot = Var(vbfh_cut)
wh_plot = Var(wh_cut)
zh_plot = Var(zh_cut)
tth_plot = Var(tth_cut)

vv_plot = Var(vv_cut)
top_plot = Var(top_cut)
ztt_plot = Var(ztt_inc_cut)
ttv_plot = Var(ttv_cut)
w_plot = Var(w_cut)
zll_plot = Var(zll_inc_cut)

ggh_plot = cut_x1_x2(ggh_plot)
vbfh_plot = cut_x1_x2(vbfh_plot)
wh_plot = cut_x1_x2(wh_plot)
zh_plot = cut_x1_x2(zh_plot)
tth_plot = cut_x1_x2(tth_plot)

vv_plot = cut_x1_x2(vv_plot)
top_plot = cut_x1_x2(top_plot)
ztt_plot = cut_x1_x2(ztt_plot)
ttv_plot = cut_x1_x2(ttv_plot)
w_plot = cut_x1_x2(w_plot)
zll_plot = cut_x1_x2(zll_plot)

#convert signal and background to pandas dataframe
ggh_plot = pd.DataFrame(np.array(ggh_plot).T, columns=calc_vars)
vbfh_plot = pd.DataFrame(np.array(vbfh_plot).T, columns=calc_vars)
wh_plot = pd.DataFrame(np.array(wh_plot).T, columns=calc_vars)
zh_plot = pd.DataFrame(np.array(zh_plot).T, columns=calc_vars)
tth_plot = pd.DataFrame(np.array(tth_plot).T, columns=calc_vars)

vv_plot = pd.DataFrame(np.array(vv_plot).T, columns=calc_vars)
top_plot = pd.DataFrame(np.array(top_plot).T, columns=calc_vars)
ztt_plot = pd.DataFrame(np.array(ztt_plot).T, columns=calc_vars)
ttv_plot = pd.DataFrame(np.array(ttv_plot).T, columns=calc_vars)
w_plot = pd.DataFrame(np.array(w_plot).T, columns=calc_vars)
zll_plot = pd.DataFrame(np.array(zll_plot).T, columns=calc_vars)


data_plot = Data_Var(data_cut)
data_plot = cut_x1_x2(data_plot)
data_s = np.array(data_plot).T
data_plot = pd.DataFrame(data_s, columns=calc_vars)




vbfh_plot['label'] = 1
ggh_plot['label'] = 1
wh_plot['label'] = 1
zh_plot['label'] = 1
tth_plot['label'] = 1

vv_plot['label'] = 0
top_plot['label'] = 0
ztt_plot['label'] = 0
ttv_plot['label'] = 0
w_plot['label'] = 0
zll_plot['label'] = 0

data_plot['label'] = 0

# Add a 'sample_type' column to each DataFrame
vbfh_plot['sample_type'] = 'vbfh'
ggh_plot['sample_type'] = 'ggh'
wh_plot['sample_type'] = 'wh'
zh_plot['sample_type'] = 'zh'
tth_plot['sample_type'] = 'tth'

vv_plot['sample_type'] = 'vv'
top_plot['sample_type'] = 'top'
ztt_plot['sample_type'] = 'ztt'
ttv_plot['sample_type'] = 'ttv'
w_plot['sample_type'] = 'w'
zll_plot['sample_type'] = 'zll'
data_plot['sample_type'] = 'data'


df = pd.concat([data_plot, ggh_plot, vbfh_plot, wh_plot, zh_plot, tth_plot, vv_plot, top_plot, ztt_plot, ttv_plot, w_plot, zll_plot])

training_var = [
    'ditau_pt', 'leadsubjet_pt', 'subleadsubjet_pt', 'visible_ditau_m',
    'collinear_mass', 'delta_R', 'delta_phi', 'delta_eta', 'label', 'met', 'met_sig', 'met_centrality',
    'event_number', 'fake_factor', 'combined_weights', 'k_t', 'x1', 'x2', 'sample_type', 
    'omni_score', 'leadsubjet_charge', 'subleadsubjet_charge', 'leadsubjet_n_core_tracks', 'subleadsubjet_n_core_tracks',
    'delta_R_lead', 'delta_eta_lead', 'delta_phi_lead', 'delta_R_sublead', 'delta_eta_sublead', 'delta_phi_sublead', 'e_ratio_lead', 'e_ratio_sublead',
    'higgs_pt', 'leadsubjet_eta', 'subleadsubjet_eta', 'ditau_eta', 'delta_phi_met_ditau'
]

df = df[training_var]

def split_data(df):
    # Split data based on event number
    ids = df['event_number'] % 3
    #append ids to the dataframe
    df['ids'] = ids
    print("All sample Split:", len(df[ids == 0]), len(df[ids == 1]), len(df[ids == 2]), len(df[ids == 3]), len(df[ids == 4]))
    #print how many signal and background events are in each set
    print("Signal Split:", len(df[(df['label'] == 1) & (df['ids'] == 0)]), len(df[(df['label'] == 1) & (df['ids'] == 1)]), len(df[(df['label'] == 1) & (df['ids'] == 2)]), len(df[(df['label'] == 1) & (df['ids'] == 3)]), len(df[(df['label'] == 1) & (df['ids'] == 4)]))
    print("Background Split:", len(df[(df['label'] == 0) & (df['ids'] == 0)]), len(df[(df['label'] == 0) & (df['ids'] == 1)]), len(df[(df['label'] == 0) & (df['ids'] == 2)]), len(df[(df['label'] == 0) & (df['ids'] == 3)]), len(df[(df['label'] == 0) & (df['ids'] == 4)]))
    # return [df[ids == 0], df[ids == 1], df[ids == 2], df[ids == 3], df[ids == 4]]
    return [df[ids == 0], df[ids == 1], df[ids == 2]]

df_split = split_data(df)

sweep_config = {
    'method': 'random', 
    'name': 'NewditauBDTSW2',
    'metric': {'name': 'significance', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.004, 'max': 0.08},
        'max_depth': {'min': 2, 'max': 6},
        'n_estimators': {'min': 50, 'max': 300},
    }
}

def train():
    wandb.init()
    config = wandb.config

    bdt_training_var = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
    feature_name_mapping = {
        'leadsubjet_pt': 'f0',
        'subleadsubjet_pt': 'f1',
        'visible_ditau_m': 'f2',
        'collinear_mass': 'f3',
        'delta_R': 'f4',
        'met': 'f5',
        'met_sig': 'f6',
        'x1': 'f7',
        'x2': 'f8',
        'met_centrality': 'f9',
    }

    # Map df_split column names
    for i in range(len(df_split)):
        df_split[i] = df_split[i].rename(columns=feature_name_mapping)

    log_losses = []
    signal_scores = []
    background_scores = []

    for i in range(len(df_split)):
        X_test = df_split[i][bdt_training_var]
        y_test = df_split[i]['label']
        evnt_w_test = df_split[i]['combined_weights']
        ff_test = df_split[i]['fake_factor']

        X_train = pd.concat([df_split[j] for j in range(len(df_split)) if j != i])[bdt_training_var]
        y_train = pd.concat([df_split[j] for j in range(len(df_split)) if j != i])['label']
        evnt_w_train = pd.concat([df_split[j] for j in range(len(df_split)) if j != i])['combined_weights']
        ff_train = pd.concat([df_split[j] for j in range(len(df_split)) if j != i])['fake_factor']
        
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        params = {
            'learning_rate': config.learning_rate,
            'max_depth': config.max_depth,
            'n_estimators': config.n_estimators,
            'eval_metric': 'logloss',
            'random_state': 2,
            'scale_pos_weight': scale_pos_weight,
            'base_score': 0.5,
            'objective': 'binary:logistic',
            'gamma': 0.001,
            'verbosity': 1,
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=ff_train * evnt_w_train)

        y_pred_proba = model.predict_proba(X_test)
        log_loss_val = log_loss(y_test, y_pred_proba, sample_weight=ff_test * evnt_w_test)
        log_losses.append(log_loss_val)

        signal_scores.extend(y_pred_proba[:, 1][y_test == 1])
        background_scores.extend(y_pred_proba[:, 1][y_test == 0])

    mean_log_loss = sum(log_losses) / len(log_losses)

    sig_hist = plt_to_root_hist_w(signal_scores, 10, 0., 1., None, False)
    bkg_hist = plt_to_root_hist_w(background_scores, 10, 0., 1., None, False)
    sig_hist.Scale(1/sig_hist.Integral())
    bkg_hist.Scale(1/bkg_hist.Integral())
    sigma = significance_bin_by_bin(sig_hist, bkg_hist, s_much_less_than_b=False)

    wandb.log({"validation_log_loss": mean_log_loss, "significance": sigma})

    

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="DiTau_ID")

# Start the sweep
wandb.agent(sweep_id, train, count=40)


