import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, '../')
from utils import mva_utils


def _arrays_to_df(arr_list, label):
    var_names = [
        'ditau_pt','leadsubjet_pt','subleadsubjet_pt','visible_ditau_m','met','collinear_mass','x1','x2',
        'met_sig','met_phi','event_number','k_t','kappa','delta_R','delta_phi','delta_eta','combined_weights','fake_factor',
        'delta_R_lead','delta_eta_lead','delta_phi_lead','delta_R_sublead','delta_eta_sublead','delta_phi_sublead',
        'met_centrality','omni_score','leadsubjet_charge','subleadsubjet_charge','leadsubjet_n_core_tracks','subleadsubjet_n_core_tracks',
        'e_ratio_lead','e_ratio_sublead','higgs_pt','leadsubjet_eta','subleadsubjet_eta','ditau_eta','delta_phi_met_ditau',
        'Ht','eta_product','delta_eta_jj','Mjj','pt_jj','delta_phi_jj','pt_jj_higgs','delta_r_leadjet_ditau','leading_jet_pt','subleading_jet_pt'
    ]
    data = {name: np.array(arr) for name, arr in zip(var_names, arr_list)}
    df = pd.DataFrame(data)
    df['label'] = label
    return df


def load_run2_df_from_pickles(mc_pickle_path: str, data_pickle_path: str) -> pd.DataFrame:
    """Load uncut MC/data pickles (Run 2), apply cuts, combine years, build labeled DataFrame.

    MC signal: ggH (label 2), VBFH (label 1)
    MC backgrounds (explicit): Ztt_inc, VV, Top, W, Zll_inc, ttV (label 0)
    Data (all years combined) is also treated as background (label 0).
    """
    cfg_run2 = mva_utils.get_config('run2')

    with open(mc_pickle_path, 'rb') as f:
        uncut_mc = pickle.load(f)
    with open(data_pickle_path, 'rb') as f:
        uncut_data = pickle.load(f)

    cut_mc = mva_utils.apply_cuts(uncut_mc, data_type='MC', config=cfg_run2)
    combined_mc = mva_utils.combine_mc_years(cut_mc)

    cut_data = mva_utils.apply_cuts(
        {
            'data_15': uncut_data.get('data_15'),
            'data_16': uncut_data.get('data_16'),
            'data_17': uncut_data.get('data_17'),
            'data_18': uncut_data.get('data_18'),
        },
        data_type='data',
        config=cfg_run2,
    )
    combined_data = mva_utils.combine_data_years(cut_data)

    include_vbf = True
    dfs = []
    if 'ggH' in combined_mc:
        var_ggh = mva_utils.Var(combined_mc['ggH'], include_vbf=include_vbf)
        dfs.append(_arrays_to_df(var_ggh, label=2))
    if 'VBFH' in combined_mc:
        var_vbf = mva_utils.Var(combined_mc['VBFH'], include_vbf=include_vbf)
        dfs.append(_arrays_to_df(var_vbf, label=1))
    # Explicit MC backgrounds to include
    background_keys = ['Ztt_inc', 'VV', 'Top', 'W', 'Zll_inc', 'ttV']
    for key in background_keys:
        if key in combined_mc:
            arr = combined_mc[key]
            var_bkg = mva_utils.Var(arr, include_vbf=include_vbf)
            dfs.append(_arrays_to_df(var_bkg, label=0))
    if len(combined_data) > 0:
        var_data = mva_utils.Data_Var(combined_data, include_vbf=include_vbf)
        dfs.append(_arrays_to_df(var_data, label=0))

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df


