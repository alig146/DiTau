import glob, os, sys
import uproot, ROOT
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.append("..")
from utils.utils import *


def signal_cut(df_chunk, bdt=False, pp13=False, pp11=False, pp33=False, ppinc=False):
    cut1 = (df_chunk['IsTruthHadronic']==1)
    cut2 = (df_chunk['n_subjets'] >=2)
    cut3 = ((df_chunk['ditau_pt'] >= 2e5) & (df_chunk['ditau_pt'] <= 1e6))

    cut4 = (((df_chunk['n_tracks_lead'] == 1) & (df_chunk['n_tracks_subl'] == 3)) | ((df_chunk['n_tracks_lead'] == 3) & (df_chunk['n_tracks_subl'] == 1)))
    cut5 = ((df_chunk['n_tracks_lead'] == 1) & (df_chunk['n_tracks_subl'] == 1))
    cut6 = ((df_chunk['n_tracks_lead'] == 3) & (df_chunk['n_tracks_subl'] == 3))
    cut7 = (((df_chunk['n_tracks_lead'] == 1) | (df_chunk['n_tracks_lead'] == 3)) & ((df_chunk['n_tracks_subl'] == 1) | (df_chunk['n_tracks_subl'] == 3)))

    cut8 = (df_chunk['bdt_score'] >= 0.72) 

    if bdt and pp13:
        return (cut1 & cut2 & cut3 & cut4 & cut8)
    elif bdt and pp11:
        return (cut1 & cut2 & cut3 & cut5 & cut8)
    elif bdt and pp33:
        return (cut1 & cut2 & cut3 & cut6 & cut8)
    elif bdt and ppinc:
        return (cut1 & cut2 & cut3 & cut7 & cut8)
    elif pp13:
        return (cut1 & cut2 & cut3 & cut4)
    elif pp11:
        return (cut1 & cut2 & cut3 & cut5)
    elif pp33:
        return (cut1 & cut2 & cut3 & cut6)
    elif ppinc:
        return (cut1 & cut2 & cut3 & cut7)
    else:
        return (cut1 & cut2 & cut3)
    
def bkg_cut(df_chunk, bdt=False, pp13=False, pp11=False, pp33=False, ppinc=False):
    cut1 = (df_chunk['n_subjets'] >=2)
    cut2 = ((df_chunk['ditau_pt'] >= 2e5) & (df_chunk['ditau_pt'] <= 1e6))

    cut4 = (((df_chunk['n_tracks_lead'] == 1) & (df_chunk['n_tracks_subl'] == 3)) | ((df_chunk['n_tracks_lead'] == 3) & (df_chunk['n_tracks_subl'] == 1)))
    cut5 = ((df_chunk['n_tracks_lead'] == 1) & (df_chunk['n_tracks_subl'] == 1))
    cut6 = ((df_chunk['n_tracks_lead'] == 3) & (df_chunk['n_tracks_subl'] == 3))
    cut7 = (((df_chunk['n_tracks_lead'] == 1) | (df_chunk['n_tracks_lead'] == 3)) & ((df_chunk['n_tracks_subl'] == 1) | (df_chunk['n_tracks_subl'] == 3)))

    cut8 = (df_chunk['bdt_score'] > 0.55)

    if bdt and pp13:
        return (cut1 & cut2 & cut4 & cut8)
    elif bdt and pp11:
        return (cut1 & cut2 & cut5 & cut8)
    elif bdt and pp33:
        return (cut1 & cut2 & cut6 & cut8)
    elif bdt and ppinc:
        return (cut1 & cut2 & cut7 & cut8)
    elif pp13:
        return (cut1 & cut2 & cut4)
    elif pp11:
        return (cut1 & cut2 & cut5)
    elif pp33:
        return (cut1 & cut2 & cut6)
    elif ppinc:
        return (cut1 & cut2 & cut7)
    else:
        return (cut1 & cut2)



def plotter():

    path = "/global/u2/a/agarabag/pscratch/ditdau_samples/"


    bkg_xs = [364701, 364702, 364703, 364704, 364705, 364706, 364707, 364708, 364709, 364710, 364711, 364712]
    graviton_xs = [425108, 425100, 425101, 425102, 425103, 425104, 425105, 425106, 425107]
    gamma_xs = [425200]

    # File Location. order maatch XS. 
    bkg_filelist = []
    for index in range(12):
        bkg_filelist.append(path+f"jz_w_newbdt/dijet_flattened_jz{index+1}.h5")

    # graviton_filelist = ["graviton_flattened_M1000.h5",
    #                 "graviton_flattened_M1500.h5",
    #                 "graviton_flattened_M1750.h5",
    #                 "graviton_flattened_M2000.h5",
    #                 "graviton_flattened_M2250.h5",
    #                 "graviton_flattened_M2500.h5",
    #                 "graviton_flattened_M3000.h5",
    #                 "graviton_flattened_M4000.h5"
    #                 "graviton_flattened_M5000.h5"]

    # gamma_filelist = ["gamma_flattened_0.h5"]   


    # # Define pT bins for pt weight
    # pt_bins = np.linspace(200000, 1000000, 41)


    # combined_gamma = h52panda(gamma_filelist, gamma_xs, signal_cut)
    # combined_graviton = h52panda(graviton_filelist, graviton_xs, signal_cut)
    combined_bkg_1p3p = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp13=True)
    combined_bkg_1p3p_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp13=True, bdt=True)
    combined_bkg_1p1p = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp11=True)
    combined_bkg_1p1p_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp11=True, bdt=True)
    combined_bkg_3p3p = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp33=True)
    combined_bkg_3p3p_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp33=True, bdt=True)
    combined_bkg_inc = h52panda(bkg_filelist, bkg_xs, bkg_cut, ppinc=True)
    combined_bkg_inc_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, ppinc=True, bdt=True)
    combined_bkg_1p3p['label'] = 0
    combined_bkg_1p3p_bdt['label'] = 0
    combined_bkg_1p1p['label'] = 0
    combined_bkg_1p1p_bdt['label'] = 0
    combined_bkg_3p3p['label'] = 0
    combined_bkg_3p3p_bdt['label'] = 0
    combined_bkg_inc['label'] = 0
    combined_bkg_inc_bdt['label'] = 0
    combined_bkg_1p3p['weight'] = combined_bkg_1p3p['event_weight'] * combined_bkg_1p3p['pT_weight']
    combined_bkg_1p3p_bdt['weight'] = combined_bkg_1p3p_bdt['event_weight'] * combined_bkg_1p3p_bdt['pT_weight']
    combined_bkg_1p1p['weight'] = combined_bkg_1p1p['event_weight'] * combined_bkg_1p1p['pT_weight']
    combined_bkg_1p1p_bdt['weight'] = combined_bkg_1p1p_bdt['event_weight'] * combined_bkg_1p1p_bdt['pT_weight']
    combined_bkg_3p3p['weight'] = combined_bkg_3p3p['event_weight'] * combined_bkg_3p3p['pT_weight']
    combined_bkg_3p3p_bdt['weight'] = combined_bkg_3p3p_bdt['event_weight'] * combined_bkg_3p3p_bdt['pT_weight']
    combined_bkg_inc['weight'] = combined_bkg_inc['event_weight'] * combined_bkg_inc['pT_weight']
    combined_bkg_inc_bdt['weight'] = combined_bkg_inc_bdt['event_weight'] * combined_bkg_inc_bdt['pT_weight']
    
    combined_bkg_1p3p.to_csv(path+'combined_bkg_1p3p.csv', index=False)
    combined_bkg_1p3p_bdt.to_csv(path+'combined_bkg_1p3p_bdt.csv', index=False)
    combined_bkg_1p1p.to_csv(path+'combined_bkg_1p1p.csv', index=False)
    combined_bkg_1p1p_bdt.to_csv(path+'combined_bkg_1p1p_bdt.csv', index=False)
    combined_bkg_3p3p.to_csv(path+'combined_bkg_3p3p.csv', index=False)
    combined_bkg_3p3p_bdt.to_csv(path+'combined_bkg_3p3p_bdt.csv', index=False)
    combined_bkg_inc.to_csv(path+'combined_bkg_inc.csv', index=False)
    combined_bkg_inc_bdt.to_csv(path+'combined_bkg_inc_bdt.csv', index=False)


    # combined_signal = pd.concat([combined_graviton, combined_gamma])
    # combined_signal['label'] = 1
    # combined_bkg['label'] = 0
    # df = combined_bkg
    # df = pd.concat([combined_bkg, combined_signal])
    # df['weight'] = df['event_weight'] * df['pT_weight']
    # df.to_csv(path+'combined_data.csv', index=False)
    # df = pd.read_csv(path+'combined_data.csv')


    # bkg_pt = df[df['label'] == 0]['ditau_pt']
    # bkg_weights = df[df['label'] == 0]['event_weight']
    # bkf_full_weight = df[df['label'] == 0]['weight']
    # background_scores = df[df['label'] == 0]['bdt_score']
    # background_scores_new = df[df['label'] == 0]['bdt_score_new']

    p = PdfPages("histogram.pdf") 

    # ###### roc curve with scikit 
    # fpr_1p3p, tpr_1p3p = calc_roc(signal_scores, background_scores, signal_weight, background_weight)
    # fpr_1p1p, tpr_1p1p = calc_roc(signal_scores_1p1p, background_scores_1p1p, signal_weight_1p1p, background_weight_1p1p)
    # fpr_3p3p, tpr_3p3p = calc_roc(signal_scores_3p3p, background_scores_3p3p, signal_weight_3p3p, background_weight_3p3p)
    # fpr_inc, tpr_inc = calc_roc(signal_scores_inc, background_scores_inc, signal_weight_inc, background_weight_inc)

    # fpr_1p3p_w, tpr_1p3p_w = calc_roc(signal_scores, background_scores, signal_weight*weights[cuts], background_weight*bkg_weights[bkg_cuts])
    # fpr_1p1p_w, tpr_1p1p_w = calc_roc(signal_scores_1p1p, background_scores_1p1p, signal_weight_1p1p*weights[cuts_1p1p], background_weight_1p1p*bkg_weights[bkg_cuts_1p1p])
    # fpr_3p3p_w, tpr_3p3p_w = calc_roc(signal_scores_3p3p, background_scores_3p3p, signal_weight_3p3p*weights[cuts_3p3p], background_weight_3p3p*bkg_weights[bkg_cuts_3p3p])
    # fpr_inc_w, tpr_inc_w = calc_roc(signal_scores_inc, background_scores_inc, signal_weight_inc*weights[cuts_inc], background_weight_inc*bkg_weights[bkg_cuts_inc])

    # fig7 = plt.figure()
    # plt.plot(tpr_1p3p, 1/fpr_1p3p, label="1p3p", color='black')
    # plt.plot(tpr_1p1p, 1/fpr_1p1p, label="1p1p", color='orange')
    # plt.plot(tpr_3p3p, 1/fpr_3p3p, label="3p3p", color='red')
    # plt.plot(tpr_inc, 1/fpr_inc, label="inclusive", color='green')
    # plt.plot(tpr_1p3p_w, 1/fpr_1p3p_w, label="1p3p weighted", linestyle='dashed', color='black')
    # plt.plot(tpr_1p1p_w, 1/fpr_1p1p_w, label="1p1p weighted", linestyle='dashed', color='orange')
    # plt.plot(tpr_3p3p_w, 1/fpr_3p3p_w, label="3p3p weighted", linestyle='dashed', color='red')
    # plt.plot(tpr_inc_w, 1/fpr_inc_w, label="incl weighted", linestyle='dashed', color='green')
    # plt.legend(loc='upper right')
    # plt.xlabel("TPR")
    # plt.ylabel("1/FPR")
    # plt.yscale('log')
    # p.savefig(fig7)
    # plt.close(fig7)


    ####### plot the score distribution
    fig_score = plt.figure()
    # plt.hist(signal_scores, bins=60, histtype="step", label="1p3p sig", color='black', weights=signal_weight)
    plt.hist(combined_bkg_1p3p['bdt_score'], bins=60, histtype="step", label="1p3p bkg", linestyle='dashed', color='black', weights=combined_bkg_1p3p['weight'])
    plt.hist(combined_bkg_1p3p['bdt_score_new'], bins=60, histtype="step", label="1p3p bkg new", linestyle='dashed', color='blue', weights=combined_bkg_1p3p['weight'])
    # plt.hist(signal_scores_1p1p, bins=60, histtype="step", label="1p1p sig", color='orange', weights=signal_weight_1p1p)
    # plt.hist(background_scores_1p1p, bins=60, histtype="step", label="1p1p bkg", linestyle='dashed', color='orange', weights=background_weight_1p1p)
    # plt.hist(signal_scores_3p3p, bins=60, histtype="step", label="3p3p sig", color='red', weights=signal_weight_3p3p)
    # plt.hist(background_scores_3p3p, bins=60, histtype="step", label="3p3p bkg", linestyle='dashed', color='red', weights=background_weight_3p3p)
    # plt.hist(signal_scores_inc, bins=60, histtype="step", label="inc sig", color='green', weights=signal_weight_inc)
    # plt.hist(background_scores_inc, bins=60, histtype="step", label="inc bkg", linestyle='dashed', color='green', weights=background_weight_inc)
    plt.xlabel("BDT score")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_score)
    plt.close(fig_score)


    p.close() #end of plt plots


    # sig_1p3p_denom, sig_1p3p_denom_edge, sig_1p3p_num, sig_1p3p_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts, cuts_bdt)
    # sig_1p1p_denom, sig_1p1p_denom_edge, sig_1p1p_num, sig_1p1p_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts_1p1p, cuts_bdt_1p1p)
    # sig_3p3p_denom, sig_3p3p_denom_edge, sig_3p3p_num, sig_3p3p_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts_3p3p, cuts_bdt_3p3p)
    # sig_inc_denom, sig_inc_denom_edge, sig_inc_num, sig_inc_num_edge = calculate_efficiency_hists(f1['ditau_pt'], bins, cuts_inc, cuts_bdt_inc)

    # sig_1p3p_denom, sig_1p3p_denom_edge, sig_1p3p_num, sig_1p3p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts, cuts_bdt)
    # sig_1p1p_denom, sig_1p1p_denom_edge, sig_1p1p_num, sig_1p1p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts_1p1p, cuts_bdt_1p1p)
    # sig_3p3p_denom, sig_3p3p_denom_edge, sig_3p3p_num, sig_3p3p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts_3p3p, cuts_bdt_3p3p)
    # sig_inc_denom, sig_inc_denom_edge, sig_inc_num, sig_inc_num_edge = calculate_efficiency_hists(f1['DiTauJetsAux.eta'], eta_bins, cuts_inc, cuts_bdt_inc)



    ####### root effeciency plots 
    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetBatch(True)
    canvas = ROOT.TCanvas("canvas", "eff_plots", 800, 500)
    canvas.cd()
    canvas.Print("eff_plots.pdf[")

    bkg_pt_list = [combined_bkg_1p3p['ditau_pt'], combined_bkg_1p3p_bdt['ditau_pt'], combined_bkg_1p1p['ditau_pt'], combined_bkg_1p1p_bdt['ditau_pt'], combined_bkg_3p3p['ditau_pt'], combined_bkg_3p3p_bdt['ditau_pt'], combined_bkg_inc['ditau_pt'], combined_bkg_inc_bdt['ditau_pt']]
    bkg_eta_list = [combined_bkg_1p3p['eta'], combined_bkg_1p3p_bdt['eta'], combined_bkg_1p1p['eta'], combined_bkg_1p1p_bdt['eta'], combined_bkg_3p3p['eta'], combined_bkg_3p3p_bdt['eta'], combined_bkg_inc['eta'], combined_bkg_inc_bdt['eta']]
    bkg_mu_list = [combined_bkg_1p3p['average_mu'], combined_bkg_1p3p_bdt['average_mu'], combined_bkg_1p1p['average_mu'], combined_bkg_1p1p_bdt['average_mu'], combined_bkg_3p3p['average_mu'], combined_bkg_3p3p_bdt['average_mu'], combined_bkg_inc['average_mu'], combined_bkg_inc_bdt['average_mu']]
    bkg_w_list = [combined_bkg_1p3p['weight'], combined_bkg_1p3p_bdt['weight'], combined_bkg_1p1p['weight'], combined_bkg_1p1p_bdt['weight'], combined_bkg_3p3p['weight'], combined_bkg_3p3p_bdt['weight'], combined_bkg_inc['weight'], combined_bkg_inc_bdt['weight']]

    pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(bkg_pt_list, bkg_w_list, "DiJet pT", 20, 200000, 1000000, eta=False)
    pt_1p3p_eff_w.SetMarkerStyle(22)
    pt_1p1p_eff_w.SetMarkerStyle(22)
    pt_3p3p_eff_w.SetMarkerStyle(22)
    pt_inc_eff_w.SetMarkerStyle(22)
    pt_1p3p_eff_w.Draw("same e")
    pt_1p1p_eff_w.Draw("same e")
    pt_3p3p_eff_w.Draw("same e")
    pt_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(pt_1p3p_eff_w, "1p3p w")
    legend.AddEntry(pt_1p1p_eff_w, "1p1p w")
    legend.AddEntry(pt_3p3p_eff_w, "3p3p w")
    legend.AddEntry(pt_inc_eff_w, "inclusive w")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_1p3p_eff_w, eta_1p1p_eff_w, eta_3p3p_eff_w, eta_inc_eff_w = plot_eff(bkg_eta_list, bkg_w_list, "DiJet eta", 40, -2.5, 2.5, eta=True)
    eta_1p3p_eff_w.SetMarkerStyle(22)
    eta_1p1p_eff_w.SetMarkerStyle(22)
    eta_3p3p_eff_w.SetMarkerStyle(22)
    eta_inc_eff_w.SetMarkerStyle(22)
    eta_1p3p_eff_w.Draw("same e")
    eta_1p1p_eff_w.Draw("same e")
    eta_3p3p_eff_w.Draw("same e")
    eta_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(eta_1p3p_eff_w, "1p3p")
    legend.AddEntry(eta_1p1p_eff_w, "1p1p")
    legend.AddEntry(eta_3p3p_eff_w, "3p3p")
    legend.AddEntry(eta_inc_eff_w, "inclusive")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_inc_eff_w.SetMarkerStyle(22)
    eta_inc_eff_w.SetLineColor(ROOT.kBlue)
    eta_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.7, 0.7, 0.7, 0.7)
    legend.AddEntry(eta_inc_eff_w, "inclusive")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    mu_1p3p_eff_w, mu_1p1p_eff_w, mu_3p3p_eff_w, mu_inc_eff_w = plot_eff(bkg_mu_list, bkg_w_list, "DiJet mu", 20, 18, 74, eta=False)
    mu_1p3p_eff_w.SetMarkerStyle(22)
    mu_1p1p_eff_w.SetMarkerStyle(22)
    mu_3p3p_eff_w.SetMarkerStyle(22)
    mu_inc_eff_w.SetMarkerStyle(22)
    mu_1p3p_eff_w.Draw("same e")
    mu_1p1p_eff_w.Draw("same e")
    mu_3p3p_eff_w.Draw("same e")
    mu_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(mu_1p3p_eff_w, "1p3p")
    legend.AddEntry(mu_1p1p_eff_w, "1p1p")
    legend.AddEntry(mu_3p3p_eff_w, "3p3p")
    legend.AddEntry(mu_inc_eff_w, "inclusive")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    mu_inc_eff_w.SetMarkerStyle(22)
    mu_inc_eff_w.SetLineColor(ROOT.kBlue)
    mu_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.7, 0.7, 0.7, 0.7)
    legend.AddEntry(mu_inc_eff_w, "inclusive")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    # pt_sig_1p3p_eff, pt_sig_1p1p_eff, pt_sig_3p3p_eff, pt_sig_inc_eff = plot_eff(ak.flatten(f1['ditau_pt']), signal_cuts_list, "Graviton pT", 20, 200000, 1000000, signal_full_event_weight, None, eta=False)
    # pt_sig_1p3p_eff_w, pt_sig_1p1p_eff_w, pt_sig_3p3p_eff_w, pt_sig_inc_eff_w = plot_eff(ak.flatten(f1['ditau_pt']), signal_cuts_list, "Graviton pT", 20, 200000, 1000000, signal_full_event_weight, weights, eta=False)
    # pt_sig_1p3p_eff.SetMarkerStyle(41)
    # pt_sig_1p1p_eff.SetMarkerStyle(41)
    # pt_sig_3p3p_eff.SetMarkerStyle(41)
    # pt_sig_inc_eff.SetMarkerStyle(41)
    # pt_sig_1p3p_eff.Draw(" e")
    # pt_sig_1p1p_eff.Draw("same e")
    # pt_sig_3p3p_eff.Draw("same e")
    # pt_sig_inc_eff.Draw("same e")
    # pt_sig_1p3p_eff_w.SetMarkerStyle(22)
    # pt_sig_1p1p_eff_w.SetMarkerStyle(22)
    # pt_sig_3p3p_eff_w.SetMarkerStyle(22)
    # pt_sig_inc_eff_w.SetMarkerStyle(22)
    # pt_sig_1p3p_eff_w.Draw("same e")
    # pt_sig_1p1p_eff_w.Draw("same e")
    # pt_sig_3p3p_eff_w.Draw("same e")
    # pt_sig_inc_eff_w.Draw("same e")
    # legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    # legend.AddEntry(pt_sig_1p3p_eff, "1p3p")
    # legend.AddEntry(pt_sig_1p1p_eff, "1p1p")
    # legend.AddEntry(pt_sig_3p3p_eff, "3p3p")
    # legend.AddEntry(pt_sig_inc_eff, "inclusive")
    # legend.AddEntry(pt_sig_1p3p_eff_w, "1p3p w")
    # legend.AddEntry(pt_sig_1p1p_eff_w, "1p1p w")
    # legend.AddEntry(pt_sig_3p3p_eff_w, "3p3p w")
    # legend.AddEntry(pt_sig_inc_eff_w, "inclusive w")
    # legend.Draw()
    # canvas.Print("eff_plots.pdf")
    # canvas.Clear()

    # eta_sig_1p3p_eff, eta_sig_1p1p_eff, eta_sig_3p3p_eff, eta_sig_inc_eff = plot_eff(ak.flatten(f1['DiTauJetsAux.eta']), signal_cuts_list, "Graviton eta", 40, -2.5, 2.5, signal_full_event_weight, None, eta=True)
    # eta_sig_1p3p_eff_w, eta_sig_1p1p_eff_w, eta_sig_3p3p_eff_w, eta_sig_inc_eff_w = plot_eff(ak.flatten(f1['DiTauJetsAux.eta']), signal_cuts_list, "Graviton eta", 40, -2.5, 2.5, signal_full_event_weight, weights, eta=True)
    # eta_sig_1p3p_eff.SetMarkerStyle(41)
    # eta_sig_1p1p_eff.SetMarkerStyle(41)
    # eta_sig_3p3p_eff.SetMarkerStyle(41)
    # eta_sig_inc_eff.SetMarkerStyle(41)
    # eta_sig_1p3p_eff.Draw(" e")
    # eta_sig_1p1p_eff.Draw("same e")
    # eta_sig_3p3p_eff.Draw("same e")
    # eta_sig_inc_eff.Draw("same e")
    # eta_sig_1p3p_eff_w.SetMarkerStyle(22)
    # eta_sig_1p1p_eff_w.SetMarkerStyle(22)
    # eta_sig_3p3p_eff_w.SetMarkerStyle(22)
    # eta_sig_inc_eff_w.SetMarkerStyle(22)
    # eta_sig_1p3p_eff_w.Draw("same e")
    # eta_sig_1p1p_eff_w.Draw("same e")
    # eta_sig_3p3p_eff_w.Draw("same e")
    # eta_sig_inc_eff_w.Draw("same e")
    # legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    # legend.AddEntry(eta_sig_1p3p_eff, "1p3p")
    # legend.AddEntry(eta_sig_1p1p_eff, "1p1p")
    # legend.AddEntry(eta_sig_3p3p_eff, "3p3p")
    # legend.AddEntry(eta_sig_inc_eff, "inclusive")
    # legend.AddEntry(eta_sig_1p3p_eff_w, "1p3p w")
    # legend.AddEntry(eta_sig_1p1p_eff_w, "1p1p w")
    # legend.AddEntry(eta_sig_3p3p_eff_w, "3p3p w")
    # legend.AddEntry(eta_sig_inc_eff_w, "inclusive w")
    # legend.Draw()
    # canvas.Print("eff_plots.pdf")
    # canvas.Clear()

    # mu_sig_1p3p_eff, mu_sig_1p1p_eff, mu_sig_3p3p_eff, mu_sig_inc_eff = plot_eff(new_mu, signal_cuts_list, "Graviton mu", 20, 18, 74, signal_full_event_weight, None, eta=False)
    # mu_sig_1p3p_eff_w, mu_sig_1p1p_eff_w, mu_sig_3p3p_eff_w, mu_sig_inc_eff_w = plot_eff(new_mu, signal_cuts_list, "Graviton mu", 20, 18, 74, signal_full_event_weight, weights, eta=False)
    # mu_sig_1p3p_eff.SetMarkerStyle(41)
    # mu_sig_1p1p_eff.SetMarkerStyle(41)
    # mu_sig_3p3p_eff.SetMarkerStyle(41)
    # mu_sig_inc_eff.SetMarkerStyle(41)
    # mu_sig_1p3p_eff.Draw(" e")
    # mu_sig_1p1p_eff.Draw("same e")
    # mu_sig_3p3p_eff.Draw("same e")
    # mu_sig_inc_eff.Draw("same e")
    # mu_sig_1p3p_eff_w.SetMarkerStyle(22)
    # mu_sig_1p1p_eff_w.SetMarkerStyle(22)
    # mu_sig_3p3p_eff_w.SetMarkerStyle(22)
    # mu_sig_inc_eff_w.SetMarkerStyle(22)
    # mu_sig_1p3p_eff_w.Draw("same e")
    # mu_sig_1p1p_eff_w.Draw("same e")
    # mu_sig_3p3p_eff_w.Draw("same e")
    # mu_sig_inc_eff_w.Draw("same e")
    # legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    # legend.AddEntry(mu_sig_1p3p_eff, "1p3p")
    # legend.AddEntry(mu_sig_1p1p_eff, "1p1p")
    # legend.AddEntry(mu_sig_3p3p_eff, "3p3p")
    # legend.AddEntry(mu_sig_inc_eff, "inclusive")
    # legend.AddEntry(mu_sig_1p3p_eff_w, "1p3p w")
    # legend.AddEntry(mu_sig_1p1p_eff_w, "1p1p w")
    # legend.AddEntry(mu_sig_3p3p_eff_w, "3p3p w")
    # legend.AddEntry(mu_sig_inc_eff_w, "inclusive w")
    # legend.Draw()
    # canvas.Print("eff_plots.pdf")
    # canvas.Clear()

    # sig_score_1p3p = plt_to_root_hist_w(signal_scores, 100, 0, 1, signal_weight, False)
    # bkg_score_1p3p = plt_to_root_hist_w(background_scores, 100, 0, 1, background_weight, False)
    # sig_score_1p1p = plt_to_root_hist_w(signal_scores_1p1p, 100, 0, 1, signal_weight_1p1p, False)
    # bkg_score_1p1p = plt_to_root_hist_w(background_scores_1p1p, 100, 0, 1, background_weight_1p1p, False)
    # sig_score_3p3p = plt_to_root_hist_w(signal_scores_3p3p, 100, 0, 1, signal_weight_3p3p, False)
    # bkg_score_3p3p = plt_to_root_hist_w(background_scores_3p3p, 100, 0, 1, background_weight_3p3p, False)
    # sig_score_inc = plt_to_root_hist_w(signal_scores_inc, 100, 0, 1, signal_weight_inc, False)
    # bkg_score_inc = plt_to_root_hist_w(background_scores_inc, 100, 0, 1, background_weight_inc, False)

    # sig_score_1p3p_w = plt_to_root_hist_w(signal_scores, 100, 0, 1, signal_weight*weights[cuts], False)
    # bkg_score_1p3p_w = plt_to_root_hist_w(background_scores, 100, 0, 1, background_weight*bkg_weights[bkg_cuts], False)
    # sig_score_1p1p_w = plt_to_root_hist_w(signal_scores_1p1p, 100, 0, 1, signal_weight_1p1p*weights[cuts_1p1p], False)
    # bkg_score_1p1p_w = plt_to_root_hist_w(background_scores_1p1p, 100, 0, 1, background_weight_1p1p*bkg_weights[bkg_cuts_1p1p], False)
    # sig_score_3p3p_w = plt_to_root_hist_w(signal_scores_3p3p, 100, 0, 1, signal_weight_3p3p*weights[cuts_3p3p], False)
    # bkg_score_3p3p_w = plt_to_root_hist_w(background_scores_3p3p, 100, 0, 1, background_weight_3p3p*bkg_weights[bkg_cuts_3p3p], False)
    # sig_score_inc_w = plt_to_root_hist_w(signal_scores_inc, 100, 0, 1, signal_weight_inc*weights[cuts_inc], False)
    # bkg_score_inc_w = plt_to_root_hist_w(background_scores_inc, 100, 0, 1, background_weight_inc*bkg_weights[bkg_cuts_inc], False)
    
    # root_1p3p_roc = create_roc_graph(sig_score_1p3p, bkg_score_1p3p, effmin=0.05, name="1p3p", normalize=False, reverse=False)
    # root_1p1p_roc = create_roc_graph(sig_score_1p1p, bkg_score_1p1p, effmin=0.05, name="1p1p", normalize=False, reverse=False)
    # root_3p3p_roc = create_roc_graph(sig_score_3p3p, bkg_score_3p3p, effmin=0.05, name="3p3p", normalize=False, reverse=False)
    # root_inc_roc = create_roc_graph(sig_score_inc, bkg_score_inc, effmin=0.05, name="inc", normalize=False, reverse=False)
    # root_1p3p_roc_w = create_roc_graph(sig_score_1p3p_w, bkg_score_1p3p_w, effmin=0.05, name="1p3p_w", normalize=False, reverse=False)
    # root_1p1p_roc_w = create_roc_graph(sig_score_1p1p_w, bkg_score_1p1p_w, effmin=0.05, name="1p1p_w", normalize=False, reverse=False)
    # root_3p3p_roc_w = create_roc_graph(sig_score_3p3p_w, bkg_score_3p3p_w, effmin=0.05, name="3p3p_w", normalize=False, reverse=False)
    # root_inc_roc_w = create_roc_graph(sig_score_inc_w, bkg_score_inc_w, effmin=0.05, name="inc_w", normalize=False, reverse=False)

    ##set colors and styles
    # root_1p3p_roc.SetLineColor(ROOT.kBlack)
    # root_1p1p_roc.SetLineColor(ROOT.kOrange)
    # root_3p3p_roc.SetLineColor(ROOT.kRed)
    # root_inc_roc.SetLineColor(ROOT.kGreen)
    # root_1p3p_roc_w.SetLineColor(ROOT.kBlack)
    # root_1p1p_roc_w.SetLineColor(ROOT.kOrange)
    # root_3p3p_roc_w.SetLineColor(ROOT.kRed)
    # root_inc_roc_w.SetLineColor(ROOT.kGreen)
    # root_1p3p_roc_w.SetLineStyle(9)
    # root_1p1p_roc_w.SetLineStyle(9)
    # root_3p3p_roc_w.SetLineStyle(9)
    # root_inc_roc_w.SetLineStyle(9)
    # # draw
    # root_1p3p_roc.Draw("")
    # root_1p1p_roc.Draw("same")
    # root_3p3p_roc.Draw("same")
    # root_inc_roc.Draw("same")
    # root_1p3p_roc_w.Draw("same")
    # root_1p1p_roc_w.Draw("same")
    # root_3p3p_roc_w.Draw("same")
    # root_inc_roc_w.Draw("same")
    # ROOT.gPad.SetLogy()
    # #legend
    # legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    # legend.AddEntry(root_1p3p_roc, "1p3p")
    # legend.AddEntry(root_1p1p_roc, "1p1p")
    # legend.AddEntry(root_3p3p_roc, "3p3p")
    # legend.AddEntry(root_inc_roc, "inclusive")
    # legend.AddEntry(root_1p3p_roc_w, "1p3p w")
    # legend.AddEntry(root_1p1p_roc_w, "1p1p w")
    # legend.AddEntry(root_3p3p_roc_w, "3p3p w")
    # legend.AddEntry(root_inc_roc_w, "inclusive w")
    # legend.Draw()
    # #log y axis
    # canvas.Print("eff_plots.pdf")
    # canvas.Clear()


    bk_pt_plt = plt_to_root_hist_w(bkg_pt, 100, 200000, 1000000, bkg_weights, False)
    bk_pt_plt.Draw("hist e")
    ROOT.gPad.SetLogy()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    canvas.Print("eff_plots.pdf]")





if __name__ == "__main__":
    plotter()
