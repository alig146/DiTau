import glob, os, sys
import uproot, ROOT
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from sklearn.metrics import auc
sys.path.append("..")
from utils.utils import *
import pandas as pd

def plotter(): 
    ####
    # combined_bkg = pd.read_csv('/global/u2/a/agarabag/pscratch/ditdau_samples/samples_for_gnn/combined_jz_ntuple_inc_omni_weights.csv')
    # # combined_signal = pd.read_csv(path+'inc_bdt_signal.csv')
    # combined_signal = pd.read_csv('/global/u2/a/agarabag/pscratch/ditdau_samples/samples_for_gnn/vhtautau_ntuple_inc_omni_weights.csv')

    path = "/global/u2/a/agarabag/pscratch/ditdau_samples/"
    combined_bkg = pd.read_csv(path+'combined_bkg_inc.csv')
    combined_signal = pd.read_csv(path+'combined_signal_inc.csv')

    class DataFrameCuts:
        def __init__(self, df_bkg, df_signal):
            self.df_bkg = df_bkg
            self.df_signal = df_signal

        def apply_cut(self, df, cut_condition):
            return df[cut_condition]

        def cut4(self, df):
            return self.apply_cut(df, ((df['n_tracks_lead'] == 1) & (df['n_tracks_subl'] == 3)) | ((df['n_tracks_lead'] == 3) & (df['n_tracks_subl'] == 1)))

        def cut5(self, df):
            return self.apply_cut(df, (df['n_tracks_lead'] == 1) & (df['n_tracks_subl'] == 1))

        def cut6(self, df):
            return self.apply_cut(df, (df['n_tracks_lead'] == 3) & (df['n_tracks_subl'] == 3))

        def process(self):
            self.combined_bkg_1p3p = self.cut4(self.df_bkg)
            self.combined_bkg_1p1p = self.cut5(self.df_bkg)
            self.combined_bkg_3p3p = self.cut6(self.df_bkg)

            self.combined_signal_1p3p = self.cut4(self.df_signal)
            self.combined_signal_1p1p = self.cut5(self.df_signal)
            self.combined_signal_3p3p = self.cut6(self.df_signal)


            return {
                'bkg': {
                    '1p3p': self.combined_bkg_1p3p,
                    '1p1p': self.combined_bkg_1p1p,
                    '3p3p': self.combined_bkg_3p3p,
                    'inc': self.df_bkg
                },
                'signal': {
                    '1p3p': self.combined_signal_1p3p,
                    '1p1p': self.combined_signal_1p1p,
                    '3p3p': self.combined_signal_3p3p,
                    'inc': self.df_signal
                }
            }

    channel = ['1p3p', '1p1p', '3p3p', 'inc']
    color = ['red', 'darkorange', 'green', 'steelblue']
    


    combined_bkg = combined_bkg[combined_bkg['bdt_score_new']>0]
    combined_signal = combined_signal[combined_signal['bdt_score_new']>0]
    combined_bkg_bdt = combined_bkg[combined_bkg['bdt_score_new'] > 0.045]
    combined_signal_bdt = combined_signal[combined_signal['bdt_score_new'] > 0.96]

    cuts_processor = DataFrameCuts(combined_bkg, combined_signal)
    combined = cuts_processor.process()

    cuts_processor_bdt = DataFrameCuts(combined_bkg_bdt, combined_signal_bdt)
    combined_bdt = cuts_processor_bdt.process()

    combined_bkg_1p3p = combined['bkg']['1p3p']
    combined_bkg_1p1p = combined['bkg']['1p1p']
    combined_bkg_3p3p = combined['bkg']['3p3p']
    combined_bkg_inc = combined['bkg']['inc']
    combined_signal_1p3p = combined['signal']['1p3p']
    combined_signal_1p1p = combined['signal']['1p1p']
    combined_signal_3p3p = combined['signal']['3p3p']
    combined_signal_inc = combined['signal']['inc']
    combined_bkg_1p3p_bdt = combined_bdt['bkg']['1p3p']
    combined_bkg_1p1p_bdt = combined_bdt['bkg']['1p1p']
    combined_bkg_3p3p_bdt = combined_bdt['bkg']['3p3p']
    combined_bkg_inc_bdt = combined_bdt['bkg']['inc']
    combined_signal_1p3p_bdt = combined_bdt['signal']['1p3p']
    combined_signal_1p1p_bdt = combined_bdt['signal']['1p1p']
    combined_signal_3p3p_bdt = combined_bdt['signal']['3p3p']
    combined_signal_inc_bdt = combined_bdt['signal']['inc']

    #for bdt
    combined_bkg_mva = combined_bkg[combined_bkg['bdt_score']>0]
    combined_signal_mva = combined_signal[combined_signal['bdt_score']>0]
    combined_bkg_bdt_mva = combined_bkg[combined_bkg['bdt_score'] > 0.045]
    combined_signal_bdt_mva = combined_signal[combined_signal['bdt_score'] > 0.72]

    cuts_processor_mva = DataFrameCuts(combined_bkg_mva, combined_signal_mva)
    combined_mva = cuts_processor_mva.process()

    cuts_processor_bdt_mva = DataFrameCuts(combined_bkg_bdt_mva, combined_signal_bdt_mva)
    combined_bdt_mva = cuts_processor_bdt_mva.process()

    combined_bkg_1p3p_mva = combined_mva['bkg']['1p3p']
    combined_bkg_1p1p_mva = combined_mva['bkg']['1p1p']
    combined_bkg_3p3p_mva = combined_mva['bkg']['3p3p']
    combined_bkg_inc_mva = combined_mva['bkg']['inc']
    combined_signal_1p3p_mva = combined_mva['signal']['1p3p']
    combined_signal_1p1p_mva = combined_mva['signal']['1p1p']
    combined_signal_3p3p_mva = combined_mva['signal']['3p3p']
    combined_signal_inc_mva = combined_mva['signal']['inc']
    combined_bkg_1p3p_bdt_mva = combined_bdt_mva['bkg']['1p3p']
    combined_bkg_1p1p_bdt_mva = combined_bdt_mva['bkg']['1p1p']
    combined_bkg_3p3p_bdt_mva = combined_bdt_mva['bkg']['3p3p']
    combined_bkg_inc_bdt_mva = combined_bdt_mva['bkg']['inc']
    combined_signal_1p3p_bdt_mva = combined_bdt_mva['signal']['1p3p']
    combined_signal_1p1p_bdt_mva = combined_bdt_mva['signal']['1p1p']
    combined_signal_3p3p_bdt_mva = combined_bdt_mva['signal']['3p3p']
    combined_signal_inc_bdt_mva = combined_bdt_mva['signal']['inc']

    combined_bkg_1p3p = combined_bkg_1p3p[(combined_bkg_1p3p['event_id']%100) >= 80]
    combined_bkg_1p3p_bdt = combined_bkg_1p3p_bdt[(combined_bkg_1p3p_bdt['event_id']%100) >= 80]
    combined_bkg_1p1p = combined_bkg_1p1p[(combined_bkg_1p1p['event_id']%100) >= 80]
    combined_bkg_1p1p_bdt = combined_bkg_1p1p_bdt[(combined_bkg_1p1p_bdt['event_id']%100) >= 80]
    combined_bkg_3p3p = combined_bkg_3p3p[(combined_bkg_3p3p['event_id']%100) >= 80]
    combined_bkg_3p3p_bdt = combined_bkg_3p3p_bdt[(combined_bkg_3p3p_bdt['event_id']%100) >= 80]
    combined_bkg_inc = combined_bkg_inc[(combined_bkg_inc['event_id']%100) >= 80]
    combined_bkg_inc_bdt = combined_bkg_inc_bdt[(combined_bkg_inc_bdt['event_id']%100) >= 80]

    combined_signal_1p3p = combined_signal_1p3p[(combined_signal_1p3p['event_id']%100) >= 80]
    combined_signal_1p3p_bdt = combined_signal_1p3p_bdt[(combined_signal_1p3p_bdt['event_id']%100) >= 80]
    combined_signal_1p1p = combined_signal_1p1p[(combined_signal_1p1p['event_id']%100) >= 80]
    combined_signal_1p1p_bdt = combined_signal_1p1p_bdt[(combined_signal_1p1p_bdt['event_id']%100) >= 80]
    combined_signal_3p3p = combined_signal_3p3p[(combined_signal_3p3p['event_id']%100) >= 80]
    combined_signal_3p3p_bdt = combined_signal_3p3p_bdt[(combined_signal_3p3p_bdt['event_id']%100) >= 80]
    combined_signal_inc = combined_signal_inc[(combined_signal_inc['event_id']%100) >= 80]
    combined_signal_inc_bdt = combined_signal_inc_bdt[(combined_signal_inc_bdt['event_id']%100) >= 80]
    # p = PdfPages("roc_curves.pdf") 

    # fig7 = plt.figure(figsize=(6, 6))
    # for i in range(4):
    #     fpr_bdt, tpr_bdt, thresholds_bdt = calc_roc(combined['signal'][channel[i]]['bdt_score_new'], combined['bkg'][channel[i]]['bdt_score_new'], combined['signal'][channel[i]]['event_weight'], combined['bkg'][channel[i]]['event_weight'])
    #     roc_auc = auc(fpr_bdt, tpr_bdt)
    #     tpr_bdt_idx = np.argmin(np.abs(tpr_bdt - 0.4))
    #     selected_bdt_wp = thresholds_bdt[tpr_bdt_idx]
    #     print("selected_wp_bdt ", channel[i], " ", selected_bdt_wp)
    #     plt.plot(tpr_bdt, 1/fpr_bdt, color=color[i], linewidth=1, alpha=0.7, label=f'{channel[i]} R22 BDT')

    # for i in range(4):
    #     fpr, tpr, thresholds = calc_roc(combined['signal'][channel[i]]['omni_probs'], combined['bkg'][channel[i]]['omni_probs'], combined['signal'][channel[i]]['event_weight'], combined['bkg'][channel[i]]['event_weight'])
    #     roc_auc = auc(fpr, tpr)
    #     tpr_idx = np.argmin(np.abs(tpr - 0.4))
    #     selected_wp = thresholds[tpr_idx]
    #     print("selected_wp_omni ", channel[i], " ", selected_wp)
    #     # print("fpr_omni: ", fpr[tpr_idx])
    #     plt.plot(tpr, 1/fpr, color=color[i], linestyle='dashed', label=f'{channel[i]} omni')



    # for i in range(4):
    #     fpr_bdt, tpr_bdt, thresholds_bdt = calc_roc(combined['signal'][channel[i]]['bdt_score_new'], 
    #                                                 combined['bkg'][channel[i]]['bdt_score_new'], 
    #                                                 combined['signal'][channel[i]]['event_weight'], 
    #                                                 combined['bkg'][channel[i]]['event_weight'])        
    #     # Find FPR value for BDT at threshold 0.96
    #     threshold_bdt_96 = 0.96
    #     fpr_idx_bdt = np.argmin(np.abs(thresholds_bdt - threshold_bdt_96))
    #     fpr_value_bdt = fpr_bdt[fpr_idx_bdt]
    #     print(f"FPR value for BDT at threshold 0.96 for {channel[i]}: {fpr_value_bdt}")

    #     # Calculate ROC for OMNI
    #     fpr_omni, tpr_omni, thresholds_omni = calc_roc(combined['signal'][channel[i]]['omni_probs'], 
    #                                                 combined['bkg'][channel[i]]['omni_probs'], 
    #                                                 combined['signal'][channel[i]]['event_weight'], 
    #                                                 combined['bkg'][channel[i]]['event_weight'])
        
    #     # Find threshold value for OMNI at the same FPR value as BDT
    #     fpr_idx_omni = np.argmin(np.abs(fpr_omni - fpr_value_bdt))
    #     threshold_omni = thresholds_omni[fpr_idx_omni]
    #     print(f"Threshold value for OMNI at FPR value of BDT for {channel[i]}: {threshold_omni}")


    # plt.xlabel('True Positive Rate')
    # plt.ylabel('1 / False Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # plt.show()

    # plt.xlabel("Signal Efficiency", fontsize=12)
    # plt.ylabel("Background Rejection", fontsize=12)
    # plt.yscale('log')
    # plt.legend()
    # plt.legend(prop={'size': 10})
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.grid(True, which="both", ls="--")
    # p.savefig(fig7)
    # plt.close(fig7)

    # ####### plot the score distribution
    # fig_score = plt.figure()
    # plt.hist(combined_bkg_1p3p['bdt_score_new'], bins=60, histtype="step", label="1p3p bkg", color='black', weights=combined_bkg_1p3p['weight'])
    # plt.hist(combined_bkg_1p3p['omni_probs'], bins=60, histtype="step", label="1p3p bkg new", linestyle='dashed', color='black', weights=combined_bkg_1p3p['weight'])
    # plt.hist(combined_signal_1p3p['bdt_score_new'], bins=60, histtype="step", label="1p3p sig", color='red', weights=combined_signal_1p3p['weight'])
    # plt.hist(combined_signal_1p3p['omni_probs'], bins=60, histtype="step", label="1p3p sig new", linestyle='dashed', color='red', weights=combined_signal_1p3p['weight'])
    # plt.xlabel("BDT score")
    # plt.ylabel("Counts")
    # plt.yscale('log')
    # plt.legend()
    # p.savefig(fig_score)
    # plt.close(fig_score)

    # fig_score2 = plt.figure()
    # plt.hist(combined_bkg_1p1p['bdt_score_new'], bins=60, histtype="step", label="1p1p bkg", color='black', weights=combined_bkg_1p1p['weight'])
    # plt.hist(combined_bkg_1p1p['omni_probs'], bins=60, histtype="step", label="1p1p bkg new", linestyle='dashed', color='black', weights=combined_bkg_1p1p['weight'])
    # plt.hist(combined_signal_1p1p['bdt_score_new'], bins=60, histtype="step", label="1p1p sig", color='red', weights=combined_signal_1p1p['weight'])
    # plt.hist(combined_signal_1p1p['omni_probs'], bins=60, histtype="step", label="1p1p sig new", linestyle='dashed', color='red', weights=combined_signal_1p1p['weight'])
    # plt.xlabel("BDT score")
    # plt.ylabel("Counts")
    # plt.yscale('log')
    # plt.legend()
    # p.savefig(fig_score2)
    # plt.close(fig_score2)

    # fig_score3 = plt.figure()
    # plt.hist(combined_bkg_3p3p['bdt_score_new'], bins=60, histtype="step", label="3p3p bkg", color='black', weights=combined_bkg_3p3p['weight'])
    # plt.hist(combined_bkg_3p3p['omni_probs'], bins=60, histtype="step", label="3p3p bkg new", linestyle='dashed', color='black', weights=combined_bkg_3p3p['weight'])
    # plt.hist(combined_signal_3p3p['bdt_score_new'], bins=60, histtype="step", label="3p3p sig", color='red', weights=combined_signal_3p3p['weight'])
    # plt.hist(combined_signal_3p3p['omni_probs'], bins=60, histtype="step", label="3p3p sig new", linestyle='dashed', color='red', weights=combined_signal_3p3p['weight'])
    # plt.xlabel("BDT score")
    # plt.ylabel("Counts")
    # plt.yscale('log')
    # plt.legend()
    # p.savefig(fig_score3)
    # plt.close(fig_score3)

    # fig_score4 = plt.figure()
    # plt.hist(combined_bkg_inc['bdt_score_new'], bins=60, histtype="step", label="inc bkg", color='black', weights=combined_bkg_inc['weight'])
    # plt.hist(combined_bkg_inc['omni_probs'], bins=60, histtype="step", label="inc bkg new", linestyle='dashed', color='black', weights=combined_bkg_inc['weight'])
    # plt.hist(combined_signal_inc['bdt_score_new'], bins=60, histtype="step", label="inc sig", color='red', weights=combined_signal_inc['weight'])
    # plt.hist(combined_signal_inc['omni_probs'], bins=60, histtype="step", label="inc sig new", linestyle='dashed', color='red', weights=combined_signal_inc['weight'])
    # plt.xlabel("BDT score")
    # plt.ylabel("Counts")
    # plt.yscale('log')
    # plt.legend()
    # p.savefig(fig_score4)
    # plt.close(fig_score4)

    # p.close() #end of plt plots

    # sig_1p3p_denom, sig_1p3p_denom_edge, sig_1p3p_num, sig_1p3p_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts, cuts_bdt)
    # sig_1p1p_denom, sig_1p1p_denom_edge, sig_1p1p_num, sig_1p1p_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts_1p1p, cuts_bdt_1p1p)
    # sig_3p3p_denom, sig_3p3p_denom_edge, sig_3p3p_num, sig_3p3p_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts_3p3p, cuts_bdt_3p3p)
    # sig_inc_denom, sig_inc_denom_edge, sig_inc_num, sig_inc_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts_inc, cuts_bdt_inc)

    # sig_1p3p_denom, sig_1p3p_denom_edge, sig_1p3p_num, sig_1p3p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts, cuts_bdt)
    # sig_1p1p_denom, sig_1p1p_denom_edge, sig_1p1p_num, sig_1p1p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts_1p1p, cuts_bdt_1p1p)
    # sig_3p3p_denom, sig_3p3p_denom_edge, sig_3p3p_num, sig_3p3p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts_3p3p, cuts_bdt_3p3p)
    # sig_inc_denom, sig_inc_denom_edge, sig_inc_num, sig_inc_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts_inc, cuts_bdt_inc)



    ####### root effeciency plots 
    ROOT.gROOT.SetStyle("ATLAS")
    ROOT.gStyle.SetLabelSize(0.03, "x")
    ROOT.gStyle.SetLabelSize(0.03, "y")
    # ROOT.gStyle.SetOptStat(0)
    # ROOT.gStyle.SetOptStat(False)
    # ROOT.gROOT.SetBatch(True)
    canvas = ROOT.TCanvas("canvas", "eff_plots", 500, 500)
    canvas.cd()
    # canvas.SetLeftMargin(0.1)
    # canvas.SetBottomMargin(0.1)
    canvas.Print("eff_plots.pdf[")

    sig_pt_list = [combined_signal_1p3p['ditau_pt']/1e6, combined_signal_1p3p_bdt['ditau_pt']/1e6, combined_signal_1p1p['ditau_pt']/1e6, combined_signal_1p1p_bdt['ditau_pt']/1e6, combined_signal_3p3p['ditau_pt']/1e6, combined_signal_3p3p_bdt['ditau_pt']/1e6, combined_signal_inc['ditau_pt']/1e6, combined_signal_inc_bdt['ditau_pt']/1e6]
    sig_eta_list = [combined_signal_1p3p['eta'], combined_signal_1p3p_bdt['eta'], combined_signal_1p1p['eta'], combined_signal_1p1p_bdt['eta'], combined_signal_3p3p['eta'], combined_signal_3p3p_bdt['eta'], combined_signal_inc['eta'], combined_signal_inc_bdt['eta']]
    sig_mu_list = [combined_signal_1p3p['average_mu'], combined_signal_1p3p_bdt['average_mu'], combined_signal_1p1p['average_mu'], combined_signal_1p1p_bdt['average_mu'], combined_signal_3p3p['average_mu'], combined_signal_3p3p_bdt['average_mu'], combined_signal_inc['average_mu'], combined_signal_inc_bdt['average_mu']]
    sig_w_list = [combined_signal_1p3p['event_weight'], combined_signal_1p3p_bdt['event_weight'], combined_signal_1p1p['event_weight'], combined_signal_1p1p_bdt['event_weight'], combined_signal_3p3p['event_weight'], combined_signal_3p3p_bdt['event_weight'], combined_signal_inc['event_weight'], combined_signal_inc_bdt['event_weight']]

    bkg_pt_list = [combined_bkg_1p3p_bdt['ditau_pt']/1e6, combined_bkg_1p3p['ditau_pt']/1e6, combined_bkg_1p1p_bdt['ditau_pt']/1e6, combined_bkg_1p1p['ditau_pt']/1e6, combined_bkg_3p3p_bdt['ditau_pt']/1e6, combined_bkg_3p3p['ditau_pt']/1e6, combined_bkg_inc_bdt['ditau_pt']/1e6, combined_bkg_inc['ditau_pt']/1e6]
    bkg_eta_list = [combined_bkg_1p3p_bdt['eta'], combined_bkg_1p3p['eta'], combined_bkg_1p1p_bdt['eta'], combined_bkg_1p1p['eta'], combined_bkg_3p3p_bdt['eta'], combined_bkg_3p3p['eta'], combined_bkg_inc_bdt['eta'], combined_bkg_inc['eta']]
    bkg_mu_list = [combined_bkg_1p3p_bdt['average_mu'], combined_bkg_1p3p['average_mu'], combined_bkg_1p1p_bdt['average_mu'], combined_bkg_1p1p['average_mu'], combined_bkg_3p3p_bdt['average_mu'], combined_bkg_3p3p['average_mu'], combined_bkg_inc_bdt['average_mu'], combined_bkg_inc['average_mu']]
    bkg_w_list = [combined_bkg_1p3p_bdt['event_weight'], combined_bkg_1p3p['event_weight'], combined_bkg_1p1p_bdt['event_weight'], combined_bkg_1p1p['event_weight'], combined_bkg_3p3p_bdt['event_weight'], combined_bkg_3p3p['event_weight'], combined_bkg_inc_bdt['event_weight'], combined_bkg_inc['event_weight']]

    sig_pt_list_mva = [combined_signal_1p3p_mva['ditau_pt']/1e6, combined_signal_1p3p_bdt_mva['ditau_pt']/1e6, combined_signal_1p1p_mva['ditau_pt']/1e6, combined_signal_1p1p_bdt_mva['ditau_pt']/1e6, combined_signal_3p3p_mva['ditau_pt']/1e6, combined_signal_3p3p_bdt_mva['ditau_pt']/1e6, combined_signal_inc_mva['ditau_pt']/1e6, combined_signal_inc_bdt_mva['ditau_pt']/1e6]
    sig_eta_list_mva = [combined_signal_1p3p_mva['eta'], combined_signal_1p3p_bdt_mva['eta'], combined_signal_1p1p_mva['eta'], combined_signal_1p1p_bdt_mva['eta'], combined_signal_3p3p_mva['eta'], combined_signal_3p3p_bdt_mva['eta'], combined_signal_inc_mva['eta'], combined_signal_inc_bdt_mva['eta']]
    sig_mu_list_mva = [combined_signal_1p3p_mva['average_mu'], combined_signal_1p3p_bdt_mva['average_mu'], combined_signal_1p1p_mva['average_mu'], combined_signal_1p1p_bdt_mva['average_mu'], combined_signal_3p3p_mva['average_mu'], combined_signal_3p3p_bdt_mva['average_mu'], combined_signal_inc_mva['average_mu'], combined_signal_inc_bdt_mva['average_mu']]
    sig_w_list_mva = [combined_signal_1p3p_mva['event_weight'], combined_signal_1p3p_bdt_mva['event_weight'], combined_signal_1p1p_mva['event_weight'], combined_signal_1p1p_bdt_mva['event_weight'], combined_signal_3p3p_mva['event_weight'], combined_signal_3p3p_bdt_mva['event_weight'], combined_signal_inc_mva['event_weight'], combined_signal_inc_bdt_mva['event_weight']]

    bkg_pt_list_mva = [combined_bkg_1p3p_bdt_mva['ditau_pt']/1e6, combined_bkg_1p3p_mva['ditau_pt']/1e6, combined_bkg_1p1p_bdt_mva['ditau_pt']/1e6, combined_bkg_1p1p_mva['ditau_pt']/1e6, combined_bkg_3p3p_bdt_mva['ditau_pt']/1e6, combined_bkg_3p3p_mva['ditau_pt']/1e6, combined_bkg_inc_bdt_mva['ditau_pt']/1e6, combined_bkg_inc_mva['ditau_pt']/1e6]
    bkg_eta_list_mva = [combined_bkg_1p3p_bdt_mva['eta'], combined_bkg_1p3p_mva['eta'], combined_bkg_1p1p_bdt_mva['eta'], combined_bkg_1p1p_mva['eta'], combined_bkg_3p3p_bdt_mva['eta'], combined_bkg_3p3p_mva['eta'], combined_bkg_inc_bdt_mva['eta'], combined_bkg_inc_mva['eta']]
    bkg_mu_list_mva = [combined_bkg_1p3p_bdt_mva['average_mu'], combined_bkg_1p3p_mva['average_mu'], combined_bkg_1p1p_bdt_mva['average_mu'], combined_bkg_1p1p_mva['average_mu'], combined_bkg_3p3p_bdt_mva['average_mu'], combined_bkg_3p3p_mva['average_mu'], combined_bkg_inc_bdt_mva['average_mu'], combined_bkg_inc_mva['average_mu']]
    bkg_w_list_mva = [combined_bkg_1p3p_bdt_mva['event_weight'], combined_bkg_1p3p_mva['event_weight'], combined_bkg_1p1p_bdt_mva['event_weight'], combined_bkg_1p1p_mva['event_weight'], combined_bkg_3p3p_bdt_mva['event_weight'], combined_bkg_3p3p_mva['event_weight'], combined_bkg_inc_bdt_mva['event_weight'], combined_bkg_inc_mva['event_weight']]
    
    #pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(bkg_pt_list, bkg_w_list, "DiJet P_{T} [TeV]", 15, 0.2, 1., eta=False, bkg=True)
    pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(bkg_pt_list, None, "DiJet P_{T} [TeV]", 15, 0.2, 1., eta=False, bkg=True)

    # pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(bkg_pt_list, bkg_w_list, "Leading Subjet P_{T} [TeV]", 20, 0., 0.5, eta=False, bkg=True)
    # pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(bkg_pt_list, bkg_w_list, "SubLeading Subjet P_{T} [TeV]", 20, 0., 0.5, eta=False, bkg=True)

    #pt_1p3p_eff_w_mva, pt_1p1p_eff_w_mva, pt_3p3p_eff_w_mva, pt_inc_eff_w_mva = plot_eff(bkg_pt_list_mva, bkg_w_list_mva, "DiJet P_{T} [TeV]", 15, 0.2, 1., eta=False, bkg=True)
    pt_1p3p_eff_w_mva, pt_1p1p_eff_w_mva, pt_3p3p_eff_w_mva, pt_inc_eff_w_mva = plot_eff(bkg_pt_list_mva, None, "DiJet P_{T} [TeV]", 15, 0.2, 1., eta=False, bkg=True)


    # pt_1p3p_eff_w.SetMarkerStyle(20)
    # pt_1p1p_eff_w.SetMarkerStyle(20)
    # pt_3p3p_eff_w.SetMarkerStyle(20)
    pt_inc_eff_w.SetMarkerStyle(20)
    pt_inc_eff_w_mva.SetMarkerStyle(20)
    # pt_1p3p_eff_w.SetMarkerSize(0.5)
    # pt_1p1p_eff_w.SetMarkerSize(0.5)
    # pt_3p3p_eff_w.SetMarkerSize(0.5)
    pt_inc_eff_w.SetMarkerSize(0.8)
    pt_inc_eff_w_mva.SetMarkerSize(0.8)
    # pt_1p3p_eff_w.SetMarkerColor(ROOT.kBlue+1)
    # pt_1p1p_eff_w.SetMarkerColor(ROOT.kOrange+8)
    # pt_3p3p_eff_w.SetMarkerColor(ROOT.kAzure+8)
    pt_inc_eff_w.SetMarkerColor(ROOT.kSpring-5)
    pt_inc_eff_w_mva.SetMarkerColor(ROOT.kOrange+8)
    pt_inc_eff_w.SetLineColor(ROOT.kSpring-5)
    pt_inc_eff_w_mva.SetLineColor(ROOT.kOrange+8)
    # pt_inc_eff_w.SetTitle("Dijet")
    pt_inc_eff_w.Draw("")
    pt_inc_eff_w_mva.Draw("same")
    # pt_1p3p_eff_w.Draw(" e")
    # pt_1p1p_eff_w.Draw("same e")
    # pt_3p3p_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.65, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    # legend.AddEntry(pt_1p3p_eff_w, "1p3p")
    # legend.AddEntry(pt_1p1p_eff_w, "1p1p")
    # legend.AddEntry(pt_3p3p_eff_w, "3p3p")
    legend.AddEntry(pt_inc_eff_w, "Omni inclusive",)
    legend.AddEntry(pt_inc_eff_w_mva, "R22 BDT inclusive",)
    legend.SetFillStyle(0)
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.035)
    tex.DrawLatex(0.2+0.02,0.88, "#bf{#it{ATLAS}} Internal")
    tex.DrawLatex(0.2+0.02,0.85, "DiJet Samples")
    pt_inc_eff_w.GetXaxis().SetLabelOffset(0.03)
    pt_inc_eff_w.GetYaxis().SetLabelOffset(0.03)
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_1p3p_eff_w, eta_1p1p_eff_w, eta_3p3p_eff_w, eta_inc_eff_w = plot_eff(bkg_eta_list, bkg_w_list, "#eta", 30, -2.5, 2.5, eta=True, bkg=True)
    eta_1p3p_eff_w_mva, eta_1p1p_eff_w_mva, eta_3p3p_eff_w_mva, eta_inc_eff_w_mva = plot_eff(bkg_eta_list_mva, bkg_w_list_mva, "#eta", 30, -2.5, 2.5, eta=True, bkg=True)

    # eta_1p3p_eff_w.SetMarkerStyle(20)
    # eta_1p1p_eff_w.SetMarkerStyle(20)
    # eta_3p3p_eff_w.SetMarkerStyle(20)
    eta_inc_eff_w.SetMarkerStyle(20)
    eta_inc_eff_w_mva.SetMarkerStyle(20)
    # eta_1p3p_eff_w.SetMarkerSize(0.5)
    # eta_1p1p_eff_w.SetMarkerSize(0.5)
    # eta_3p3p_eff_w.SetMarkerSize(0.5)
    eta_inc_eff_w.SetMarkerSize(0.8)
    eta_inc_eff_w_mva.SetMarkerSize(0.8)
    # eta_1p3p_eff_w.SetMarkerColor(ROOT.kBlue+1)
    # eta_1p1p_eff_w.SetMarkerColor(ROOT.kOrange+8)
    # eta_3p3p_eff_w.SetMarkerColor(ROOT.kAzure+8)
    eta_inc_eff_w.SetMarkerColor(ROOT.kSpring-5)
    eta_inc_eff_w_mva.SetMarkerColor(ROOT.kOrange+8)
    eta_inc_eff_w.SetLineColor(ROOT.kSpring-5)
    eta_inc_eff_w_mva.SetLineColor(ROOT.kOrange+8)
    # eta_1p3p_eff_w.Draw(" e")
    # eta_1p1p_eff_w.Draw("same e")
    # eta_3p3p_eff_w.Draw("same e")
    eta_inc_eff_w.Draw("")
    eta_inc_eff_w_mva.Draw("same")
    legend = ROOT.TLegend(0.65, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    # legend.AddEntry(eta_1p3p_eff_w, "1p3p")
    # legend.AddEntry(eta_1p1p_eff_w, "1p1p")
    # legend.AddEntry(eta_3p3p_eff_w, "3p3p")
    legend.AddEntry(eta_inc_eff_w, "Omni inclusive")
    legend.AddEntry(eta_inc_eff_w_mva, "R22 BDT inclusive")
    legend.Draw()
    # tex = ROOT.TLatex()
    # tex.SetNDC()
    # tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.88, "#bf{#it{ATLAS}} Internal")
    tex.DrawLatex(0.2+0.02,0.85, "DiJet Samples")
    eta_inc_eff_w.GetXaxis().SetLabelOffset(0.03)
    eta_inc_eff_w.GetYaxis().SetLabelOffset(0.03)
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    mu_1p3p_eff_w, mu_1p1p_eff_w, mu_3p3p_eff_w, mu_inc_eff_w = plot_eff(bkg_mu_list, bkg_w_list, "<#mu>", 15, 18, 74, eta=False, bkg=True)
    mu_1p3p_eff_w_mva, mu_1p1p_eff_w_mva, mu_3p3p_eff_w_mva, mu_inc_eff_w_mva = plot_eff(bkg_mu_list_mva, bkg_w_list_mva, "<#mu>", 15, 18, 74, eta=False, bkg=True)
    # mu_1p3p_eff_w.SetMarkerStyle(20)
    # mu_1p1p_eff_w.SetMarkerStyle(20)
    # mu_3p3p_eff_w.SetMarkerStyle(20)
    mu_inc_eff_w.SetMarkerStyle(20)
    mu_inc_eff_w_mva.SetMarkerStyle(20)
    # mu_1p3p_eff_w.SetMarkerSize(0.5)
    # mu_1p1p_eff_w.SetMarkerSize(0.5)
    # mu_3p3p_eff_w.SetMarkerSize(0.5)
    mu_inc_eff_w.SetMarkerSize(0.8)
    mu_inc_eff_w_mva.SetMarkerSize(0.8)
    # mu_1p3p_eff_w.SetMarkerColor(ROOT.kBlue+1)
    # mu_1p1p_eff_w.SetMarkerColor(ROOT.kOrange+8)
    # mu_3p3p_eff_w.SetMarkerColor(ROOT.kAzure+8)
    mu_inc_eff_w.SetMarkerColor(ROOT.kSpring-5)
    mu_inc_eff_w_mva.SetMarkerColor(ROOT.kOrange+8)
    mu_inc_eff_w.SetLineColor(ROOT.kSpring-5)
    mu_inc_eff_w_mva.SetLineColor(ROOT.kOrange+8)
    # mu_1p3p_eff_w.Draw(" e")
    # mu_1p1p_eff_w.Draw("same e")
    # mu_3p3p_eff_w.Draw("same e")
    mu_inc_eff_w.Draw("")
    mu_inc_eff_w_mva.Draw("same")
    legend = ROOT.TLegend(0.65, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    # legend.AddEntry(mu_1p3p_eff_w, "1p3p")
    # legend.AddEntry(mu_1p1p_eff_w, "1p1p")
    # legend.AddEntry(mu_3p3p_eff_w, "3p3p")
    legend.AddEntry(mu_inc_eff_w, "Omni inclusive")
    legend.AddEntry(mu_inc_eff_w_mva, "R22 BDT inclusive")
    legend.Draw()
    # tex = ROOT.TLatex()
    # tex.SetNDC()
    # tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.88, "#bf{#it{ATLAS}} Internal")
    tex.DrawLatex(0.2+0.02,0.85, "DiJet Samples")
    mu_inc_eff_w.GetXaxis().SetLabelOffset(0.03)
    mu_inc_eff_w.GetYaxis().SetLabelOffset(0.03)
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

 
    ##### signal eff plots
    pt_sig_1p3p_eff_w, pt_sig_1p1p_eff_w, pt_sig_3p3p_eff_w, pt_sig_inc_eff_w = plot_eff(sig_pt_list, sig_w_list, "P_{T} [TeV]", 15, 0.2, 1., eta=False, bkg=False)
    pt_sig_1p3p_eff_w_mva, pt_sig_1p1p_eff_w_mva, pt_sig_3p3p_eff_w_mva, pt_sig_inc_eff_w_mva = plot_eff(sig_pt_list_mva, sig_w_list_mva, "P_{T} [TeV]", 15, 0.2, 1., eta=False, bkg=False)

    # pt_sig_1p3p_eff_w.SetMarkerStyle(20)
    # pt_sig_1p1p_eff_w.SetMarkerStyle(20)
    # pt_sig_3p3p_eff_w.SetMarkerStyle(20)
    pt_sig_inc_eff_w.SetMarkerStyle(20)
    pt_sig_inc_eff_w_mva.SetMarkerStyle(20)
    # pt_sig_1p3p_eff_w.SetMarkerSize(0.5)
    # pt_sig_1p1p_eff_w.SetMarkerSize(0.5)
    # pt_sig_3p3p_eff_w.SetMarkerSize(0.5)
    pt_sig_inc_eff_w.SetMarkerSize(0.8)
    pt_sig_inc_eff_w_mva.SetMarkerSize(0.8)
    # pt_sig_1p3p_eff_w.SetMarkerColor(ROOT.kBlue+1)
    # pt_sig_1p1p_eff_w.SetMarkerColor(ROOT.kOrange+8)
    # pt_sig_3p3p_eff_w.SetMarkerColor(ROOT.kAzure+8)
    pt_sig_inc_eff_w.SetMarkerColor(ROOT.kSpring-5)
    pt_sig_inc_eff_w_mva.SetMarkerColor(ROOT.kOrange+8)
    pt_sig_inc_eff_w.SetLineColor(ROOT.kSpring-5)
    pt_sig_inc_eff_w_mva.SetLineColor(ROOT.kOrange+8)
    # pt_sig_1p3p_eff_w.Draw(" e")
    # pt_sig_1p1p_eff_w.Draw("same e")
    # pt_sig_3p3p_eff_w.Draw("same e")
    pt_sig_inc_eff_w.Draw("")
    pt_sig_inc_eff_w_mva.Draw("same ")
    legend = ROOT.TLegend(0.65, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    # legend.AddEntry(pt_sig_1p3p_eff_w, "1p3p")
    # legend.AddEntry(pt_sig_1p1p_eff_w, "1p1p")
    # legend.AddEntry(pt_sig_3p3p_eff_w, "3p3p")
    legend.AddEntry(pt_sig_inc_eff_w, "Omni inclusive")
    legend.AddEntry(pt_sig_inc_eff_w_mva, "R22 BDT inclusive")
    legend.Draw()
    # tex = ROOT.TLatex()
    # tex.SetNDC()
    # tex.SetTextSize(0.04)
    # tex.DrawLatex(0.2+0.02,0.25, "#bf{#it{ATLAS}} Internal")
    tex.DrawLatex(0.2+0.02,0.88, "#bf{#it{ATLAS}} Internal")
    # tex.DrawLatex(0.2+0.02,0.85, "DiJet Samples")
    pt_sig_inc_eff_w.GetXaxis().SetLabelOffset(0.03)
    pt_sig_inc_eff_w.GetYaxis().SetLabelOffset(0.03)
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_sig_1p3p_eff_w, eta_sig_1p1p_eff_w, eta_sig_3p3p_eff_w, eta_sig_inc_eff_w = plot_eff(sig_eta_list, sig_w_list, "#eta", 30, -2.5, 2.5, eta=True, bkg=False)
    eta_sig_1p3p_eff_w_mva, eta_sig_1p1p_eff_w_mva, eta_sig_3p3p_eff_w_mva, eta_sig_inc_eff_w_mva = plot_eff(sig_eta_list_mva, sig_w_list_mva, "#eta", 30, -2.5, 2.5, eta=True, bkg=False)

    # eta_sig_1p3p_eff_w.SetMarkerStyle(20)
    # eta_sig_1p1p_eff_w.SetMarkerStyle(20)
    # eta_sig_3p3p_eff_w.SetMarkerStyle(20)
    eta_sig_inc_eff_w.SetMarkerStyle(20)
    eta_sig_inc_eff_w_mva.SetMarkerStyle(20)
    # eta_sig_1p3p_eff_w.SetMarkerSize(0.5)
    # eta_sig_1p1p_eff_w.SetMarkerSize(0.5)
    # eta_sig_3p3p_eff_w.SetMarkerSize(0.5)
    eta_sig_inc_eff_w.SetMarkerSize(0.8)
    eta_sig_inc_eff_w_mva.SetMarkerSize(0.8)
    # eta_sig_1p3p_eff_w.SetMarkerColor(ROOT.kBlue+1)
    # eta_sig_1p1p_eff_w.SetMarkerColor(ROOT.kOrange+8)
    # eta_sig_3p3p_eff_w.SetMarkerColor(ROOT.kAzure+8)
    eta_sig_inc_eff_w.SetMarkerColor(ROOT.kSpring-5)
    eta_sig_inc_eff_w_mva.SetMarkerColor(ROOT.kOrange+8)
    eta_sig_inc_eff_w.SetLineColor(ROOT.kSpring-5)
    eta_sig_inc_eff_w_mva.SetLineColor(ROOT.kOrange+8)
    # eta_sig_1p3p_eff_w.Draw(" e")
    # eta_sig_1p1p_eff_w.Draw("same e")
    # eta_sig_3p3p_eff_w.Draw("same e")
    eta_sig_inc_eff_w.Draw("")
    eta_sig_inc_eff_w_mva.Draw("same")
    legend = ROOT.TLegend(0.65, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    # legend.AddEntry(eta_sig_1p3p_eff_w, "1p3p")
    # legend.AddEntry(eta_sig_1p1p_eff_w, "1p1p")
    # legend.AddEntry(eta_sig_3p3p_eff_w, "3p3p")
    legend.AddEntry(eta_sig_inc_eff_w, "Omni inclusive")
    legend.AddEntry(eta_sig_inc_eff_w_mva, "R22 BDT inclusive")
    legend.Draw()
    # tex = ROOT.TLatex()
    # tex.SetNDC()
    # tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.88, "#bf{#it{ATLAS}} Internal")
    eta_sig_inc_eff_w.GetXaxis().SetLabelOffset(0.03)
    eta_sig_inc_eff_w.GetYaxis().SetLabelOffset(0.03)
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    mu_sig_1p3p_eff_w, mu_sig_1p1p_eff_w, mu_sig_3p3p_eff_w, mu_sig_inc_eff_w = plot_eff(sig_mu_list, sig_w_list, "<#mu>", 15, 18, 74, eta=False, bkg=False)
    mu_sig_1p3p_eff_w_mva, mu_sig_1p1p_eff_w_mva, mu_sig_3p3p_eff_w_mva, mu_sig_inc_eff_w_mva = plot_eff(sig_mu_list_mva, sig_w_list_mva, "<#mu>", 15, 18, 74, eta=False, bkg=False)
    # mu_sig_1p3p_eff_w.SetMarkerStyle(20)
    # mu_sig_1p1p_eff_w.SetMarkerStyle(20)
    # mu_sig_3p3p_eff_w.SetMarkerStyle(20)
    mu_sig_inc_eff_w.SetMarkerStyle(20)
    mu_sig_inc_eff_w_mva.SetMarkerStyle(20)
    # mu_sig_1p3p_eff_w.SetMarkerSize(0.5)
    # mu_sig_1p1p_eff_w.SetMarkerSize(0.5)
    # mu_sig_3p3p_eff_w.SetMarkerSize(0.5)
    mu_sig_inc_eff_w.SetMarkerSize(0.8)
    mu_sig_inc_eff_w_mva.SetMarkerSize(0.8)
    # mu_sig_1p3p_eff_w.SetMarkerColor(ROOT.kBlue+1)
    # mu_sig_1p1p_eff_w.SetMarkerColor(ROOT.kOrange+8)
    # mu_sig_3p3p_eff_w.SetMarkerColor(ROOT.kAzure+8)
    mu_sig_inc_eff_w.SetMarkerColor(ROOT.kSpring-5)
    mu_sig_inc_eff_w_mva.SetMarkerColor(ROOT.kOrange+8)
    mu_sig_inc_eff_w.SetLineColor(ROOT.kSpring-5)
    mu_sig_inc_eff_w_mva.SetLineColor(ROOT.kOrange+8)
    # mu_sig_1p3p_eff_w.Draw(" e")
    # mu_sig_1p1p_eff_w.Draw("same e")
    # mu_sig_3p3p_eff_w.Draw("same e")
    mu_sig_inc_eff_w.Draw("")
    mu_sig_inc_eff_w_mva.Draw("same")
    legend = ROOT.TLegend(0.65, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    # legend.SetFillStyle(3005)
    # legend.AddEntry(mu_sig_1p3p_eff_w, "1p3p")
    # legend.AddEntry(mu_sig_1p1p_eff_w, "1p1p")
    # legend.AddEntry(mu_sig_3p3p_eff_w, "3p3p")
    legend.AddEntry(mu_sig_inc_eff_w, "Omni inclusive")
    legend.AddEntry(mu_sig_inc_eff_w_mva, "R22 BDT inclusive")
    legend.Draw()
    # tex = ROOT.TLatex()
    # tex.SetNDC()
    # tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.88, "#bf{#it{ATLAS}} Internal")
    mu_sig_inc_eff_w.GetXaxis().SetLabelOffset(0.03)
    mu_sig_inc_eff_w.GetYaxis().SetLabelOffset(0.03)
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    bk_pt_plt = plt_to_root_hist_w(combined_bkg_inc['ditau_pt'], 50, 200000, 1000000, combined_bkg_inc['event_weight'], False)
    # bk_pt_plt = plt_to_root_hist_w(combined_bkg_inc['lead_subjet_pt']/1e6, 20, 0, 0.5, None, False)
    bk_pt_plt.SetMarkerStyle(20)
    bk_pt_plt.SetMarkerSize(0.1)
    bk_pt_plt.Draw("hist e")
    # ROOT.gPad.SetLogy()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    canvas.Print("eff_plots.pdf]")







if __name__ == "__main__":
    plotter()
