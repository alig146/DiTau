import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, '../')
from utils import mva_utils
import argparse

# Shared feature mapping for STXS training
FEATURE_MAPPING = {
    'leadsubjet_pt': 'f0',
    'subleadsubjet_pt': 'f1', 
    'visible_ditau_m': 'f2',
    'collinear_mass': 'f3',
    'delta_R': 'f4',
    'met': 'f5',
    'delta_phi_met_ditau': 'f6',
    'eta_product': 'f7',
    'delta_eta_jj': 'f8',
    'Mjj': 'f9',
    'pt_jj': 'f10',
    'pt_jj_higgs': 'f11'
}

def _arrays_to_df(arr_list, label):
    var_names = [
        'ditau_pt','leadsubjet_pt','subleadsubjet_pt','visible_ditau_m','met','collinear_mass','x1','x2',
        'met_sig','met_phi','event_number','k_t','kappa','delta_R','delta_phi','delta_eta','combined_weights','fake_factor',
        'delta_R_lead','delta_eta_lead','delta_phi_lead','delta_R_sublead','delta_eta_sublead','delta_phi_sublead',
        'met_centrality','omni_score','leadsubjet_charge','subleadsubjet_charge','leadsubjet_n_core_tracks','subleadsubjet_n_core_tracks',
        'e_ratio_lead','e_ratio_sublead','higgs_pt','leadsubjet_eta','subleadsubjet_eta','ditau_eta','delta_phi_met_ditau',
        'Ht','eta_product','delta_eta_jj','Mjj','pt_jj','delta_phi_jj','pt_jj_higgs','delta_r_leadjet_ditau',
        'vbf_leading_jet_pt','vbf_subleading_jet_pt',
        'mean_pT_subjets','rms_pT_subjets','skewness_pT_subjets','kurtosis_pT_subjets',
        'mean_dR_subjet_ditau','rms_dR_subjet_ditau','skewness_dR_subjet_ditau','kurtosis_dR_subjet_ditau',
        'mean_dEta_subjet_ditau','rms_dEta_subjet_ditau','skewness_dEta_subjet_ditau','kurtosis_dEta_subjet_ditau',
        'mean_dEta_subjet_MET','rms_dEta_subjet_MET','skewness_dEta_subjet_MET','kurtosis_dEta_subjet_MET',
        'delta_R_lead_boost','delta_R_sublead_boost','delta_R_subjets_boost',
        'delta_R_met_leadsubjet_boost','delta_R_met_subleadsubjet_boost',
        'leadsubjet_pt_boost','subleadsubjet_pt_boost','met_pt_boost','subjet_vismass_boost',
        'delta_R_met_leaduniquejet','delta_R_leadsubjet_leaduniquejet','delta_R_subleadsubjet_leaduniquejet',
        'delta_R_met_leaduniquejet_boost','delta_R_leadsubjet_leaduniquejet_boost','delta_R_subleadsubjet_leaduniquejet_boost'
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



def build_stxs_training_df_run2(mc_pickle_path: str, data_pickle_path: str, save_path: str | None = None) -> pd.DataFrame:
    """Return a preprocessed DataFrame for STXS Run 2 training and optionally save it.

    Steps:
    - Load uncut MC/data pickles
    - Apply cuts and combine years
    - Build variables and assemble labeled DataFrame
    - Create model-ready feature columns (f0..)
    - Filter out non-positive combined weights
    - Optionally save to a pickle at save_path
    """
    df = load_run2_df_from_pickles(mc_pickle_path, data_pickle_path)

    # Map human-readable feature names to model columns (f0..)
    for human_name, feat_name in FEATURE_MAPPING.items():
        if human_name in df.columns:
            df[feat_name] = df[human_name]
        else:
            df[feat_name] = np.nan

    # Drop events with non-positive combined weights
    df = df[df['combined_weights'] > 0].reset_index(drop=True)

    if save_path:
        df.to_pickle(save_path)
    return df


def load_stxs_training_df(path: str) -> pd.DataFrame:
    """Load previously built STXS training DataFrame from pickle."""
    return pd.read_pickle(path)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build and save the preprocessed STXS Run 2 training DataFrame."
    )
    parser.add_argument(
        "--mc", dest="mc_pickle", required=False,
        default="/pscratch/sd/a/agarabag/ditdau_samples/raw_mc_run2.pkl",
        help="Path to uncut MC pickle (Run 2)"
    )
    parser.add_argument(
        "--data", dest="data_pickle", required=False,
        default="/pscratch/sd/a/agarabag/ditdau_samples/raw_data_run2.pkl",
        help="Path to uncut data pickle (Run 2)"
    )
    parser.add_argument(
        "--out", dest="out_pickle", required=False,
        default="/pscratch/sd/a/agarabag/ditdau_samples/stxs_training_run2.pkl",
        help="Output path for preprocessed STXS training pickle"
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    print("[STXS] Building preprocessed training DataFrame (Run 2)...")
    print(f"  MC   : {args.mc_pickle}")
    print(f"  Data : {args.data_pickle}")
    print(f"  Out  : {args.out_pickle}")

    df = build_stxs_training_df_run2(
        mc_pickle_path=args.mc_pickle,
        data_pickle_path=args.data_pickle,
        save_path=args.out_pickle,
    )
    print(f"[STXS] Done. Rows: {len(df):,}. Saved to: {args.out_pickle}")


if __name__ == "__main__":
    main()

