import uproot
import ROOT
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, roc_auc_score
import time
import glob
import os
from array import array
import ctypes
import random

def divide_hists(h_num, h_den, name=None, eff=False):
    h = h_num.Clone()
    if eff: 
        h.Divide(h_num, h_den, 1., 1., 'B')
    else: 
        h.Divide(h_den)
    h.SetName(name or 'h_ratio')
    h.GetXaxis().SetTitle(h_num.GetXaxis().GetTitle())
    h.GetYaxis().SetTitle('Ratio')
    return h

def make_eff_hist(h_pass, h_total, name=None):
    h = divide_hists(h_pass, h_total, name=name or "h_eff", eff=True)
    h.GetYaxis().SetTitle('Efficiency')
    return h

def make_eff_graph(h_pass, h_total, name=None):
    g = ROOT.TGraphAsymmErrors()
    g.Divide(h_pass,h_total, 'cl=0.683 b(1,1) mode')
    g.SetName(name or 'g_eff')
    g.GetXaxis().SetTitle(h_pass.GetXaxis().GetTitle())
    g.GetYaxis().SetTitle('Efficiency')
    g.GetXaxis().SetRangeUser(h_total.GetXaxis().GetXmin(), h_total.GetXaxis().GetXmax())
    return g

def calc_roc_curve(signal_hist, background_hist):
    # Normalize histograms
    signal_hist = np.array(signal_hist, dtype=float)  / np.sum(signal_hist)
    background_hist = np.array(background_hist, dtype=float)  / np.sum(background_hist)
    # loop over each bin and calculate tpr and fpr
    tpr = []
    fpr = []
    for i in range(0, len(signal_hist)):
        # Calculate cumulative distribution functions (CDF)
        # if np.mean(signal_hist) > np.mean(background_hist):
        signal_cdf = np.sum(signal_hist[i:])
        background_cdf = np.sum(background_hist[i:])
        # else:
        #     signal_cdf = np.sum(signal_hist[:i])
        #     background_cdf = np.sum(background_hist[:i])
        # Calculate true positive rate (TPR) and false positive rate (FPR)
        tpr.append(signal_cdf)
        fpr.append(background_cdf)

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    fpr_cut = fpr[(tpr>0.05) & (fpr>0)] 
    tpr_cut = tpr[(tpr>0.05) & (fpr>0)]
    tpr_cut = tpr_cut
    fpr_cut = np.sum(background_hist)/fpr_cut
    return tpr_cut, fpr_cut

def getXS(dsid):
    xs_file = "/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc16.txt"
    try:
        with open(xs_file, "r") as f:
            for line in f:
                columns = line.split()
                if columns[0] == str(dsid):
                    return float(columns[2])*float(columns[3])*float(columns[4])
        print( "Couldn't find cross section for dsid", dsid, "so setting to 1.")
    except IOError:
        print("Cross section file not accessible on cvmfs.", dsid, " XS setting to 1.")
    return 1

def getNevents(root_files):
    sum = 0
    if isinstance(root_files, str):
        root_files = [root_files]
    for root_file in root_files:
        f = ROOT.TFile(root_file)
        hist = f.Get("CutFlow")
        sum += hist.GetBinContent(2)
    return sum

def event_weight(tree):
    non_empty_pt_arrays = ak.num(tree['DiTauJetsAuxDyn.ditau_pt']) > 0
    filtered_ditau_pt = tree['DiTauJetsAuxDyn.ditau_pt'][non_empty_pt_arrays]
    num_dijet = ak.num(filtered_ditau_pt, axis=1)
    # we use the first elment of the weights array (total size 27)
    first_element = tree['EventInfoAuxDyn.mcEventWeights'][non_empty_pt_arrays][:, 0]
    repeated_first_element = np.repeat(first_element, num_dijet)
    return repeated_first_element

def event_weight_sum(tree):
    non_empty_pt_arrays = ak.num(tree['DiTauJetsAuxDyn.ditau_pt']) > 0
    # filtered_ditau_pt = tree['DiTauJetsAuxDyn.ditau_pt'][non_empty_pt_arrays]
    # lengths = ak.num(filtered_ditau_pt, axis=1)
    # we use the first elment of the weights array (total size 27)
    first_element = tree['EventInfoAuxDyn.mcEventWeights'][:, 0]
    # repeated_first_element = np.repeat(first_element, lengths)
    return np.sum(first_element)

def read_file(args, branches):
    file = args
    f = uproot.open(file)
    branch_names = branches
    # print(branch_names)
    branches = f['CollectionTree'].arrays(branch_names, library='ak')
    return branches


def read_tree(file_paths):
    expanded_file_paths = []
    for path in file_paths:
        expanded_file_paths.extend(glob.glob(path))
    # print(expanded_file_paths)
    with Pool() as pool:
        args = [(file) for file in expanded_file_paths]
        arrays = pool.map(read_file, args)

    combined_arrays = ak.concatenate(arrays)
    return combined_arrays


#given uproot arrys object apply cuts and return the desired vairble with the cuts applied
# def apply_cuts(arrays, cuts, var):
#     if cuts == 'norm':
#         cuts = ak.where((ak.flatten(arrays['DiTauJetsAuxDyn.n_subjets']) >=2) & 
#                     (ak.flatten(arrays['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
#                     ((ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_lead']) == 1) | (ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_lead']) == 3)) & 
#                     ((ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_subl']) == 1) | (ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) )
#         var_cut = ak.flatten(arrays[var])[cuts]
#         return var_cut
#     elif cuts == 'bdt':
#         cuts_bdt = ak.where((ak.flatten(arrays['DiTauJetsAuxDyn.n_subjets']) >=2) & 
#                     (ak.flatten(arrays['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
#                     ((ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_lead']) == 1) | (ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_lead']) == 3)) & 
#                     ((ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_subl']) == 1) | (ak.flatten(arrays['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) & 
#                     (ak.flatten(arrays['DiTauJetsAuxDyn.BDTScore']) >= 0.72))
#         var_cut = ak.flatten(arrays[var])[cuts_bdt]
#         return var_cut
    
# def read_trees(file_paths, branch_names):
#     arrays = []
#     for batch in uproot.iterate(file_paths, branch_names, library='ak', use_threads=True, num_workers=8, batch_size=200000):
#         #apply cuts to the ditau_pt branch if it passes save events from all branches in arrays dict
#         pt_cut = apply_cuts(batch, 'norm', 'DiTauJetsAuxDyn.ditau_pt')
#         if(len(pt_cut) == 0):
#             continue
#         else:
#             for branch in branch_names:
#                 arrays[branch].extend(ak.flatten(batch[branch]))
#                 # arrays[branch] = ak.flatten(batch[branch])
#     return arrays

# def process_file(file, slice, branches, chunk_size=1000000, threshold_file_size=2e9):
#     # Get Nevents constant
#     slices = [364701, 364702, 364703, 364704, 364705, 364706, 364707, 364708, 364709, 364710, 364711, 364712]
#     c1 = getNevents(file)
#     data = []
#     weights = []
#     file_size = os.path.getsize(file)
#     if file_size > threshold_file_size:
#         print("LLLLLLLLLLL: ", file, " : ", file_size)
#         # root_file = uproot.open(file)
#         # tree = root_file['CollectionTree']
#         # data = []
#         # weights = []
#         # for chunk in tree.iterate(branches, step_size=chunk_size, library='ak'):
#         #     chunk_data = np.array(ak.flatten(chunk[branches[0]]))
#         #     chunk_weights = event_weight(chunk) * getXS(slices[slice]) / c1
#         #     data.extend(chunk_data)
#         #     weights.extend(chunk_weights)
#         # # Close root file
#         # root_file.close()
#     else:
#         root_file = uproot.open(file)
#         tree = root_file['CollectionTree'].arrays(branches, library='ak')
#         data = np.array(ak.flatten(tree[branches[0]]))
#         print("WWWWWWW: ", np.sum(event_weight(tree)), " : ", c1)
#         weights = event_weight(tree) * getXS(slices[slice]) / c1
#         # Close root file
#         root_file.close()
#     return data, weights

# def plot_root_files(file_paths, branches):
#     all_data = []
#     all_weights = []

#     with Pool() as pool:
#         for i, path in enumerate(file_paths):
#             files = glob.glob(os.path.join(path))
#             results = []
#             for file in files:
#                 result = pool.apply_async(process_file, args=(file, i, branches))
#                 results.append(result)

#             for result in results:
#                 result = result.get()  # Retrieve the result from the async call
#                 if result is not None:
#                     data, weights = result
#                     all_data.extend(data)
#                     all_weights.extend(weights)

#     return all_data, all_weights


def calculate_efficiency(data, bins, denom_cuts, num_cuts, weights=None):
    denom, dnom_bin_edges = np.histogram(ak.flatten(data)[denom_cuts], bins=bins, weights=weights[denom_cuts] if weights is not None else None)
    num, num_bin_edges = np.histogram(ak.flatten(data)[num_cuts], bins=bins, weights=weights[num_cuts] if weights is not None else None)
    denom = np.array(denom).astype(float)
    num = np.array(num).astype(float)
    efficiency = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0).astype(float)
    return efficiency

def calculate_efficiency_hists(data, bins, denom_cuts, num_cuts, weights=None):
    denom, dnom_bin_edges = np.histogram(ak.flatten(data)[denom_cuts], bins=bins, weights=weights[denom_cuts] if weights is not None else None)
    num, num_bin_edges = np.histogram(ak.flatten(data)[num_cuts], bins=bins, weights=weights[num_cuts] if weights is not None else None)
    return denom, dnom_bin_edges, num, num_bin_edges

def plt_to_root_hist(hist, bin_edg):
    num_bins = len(bin_edg) - 1  # Number of bins in the histogram
    x_min = bin_edg[0]  # Minimum x-value of the histogram
    x_max = bin_edg[-1]  # Maximum x-value of the histogram
    root_hist = ROOT.TH1D("root_hist", "ROOT Histogram", num_bins, x_min, x_max)
    for bin in range(1, num_bins + 1):
        root_hist.SetBinContent(bin, array('f', hist)[bin - 1])
    return root_hist

def plt_to_root_hist_w(data, num_bins, x_min, x_max, weights=None, eta=False):
    #convert data to array
    data = array('f', data)
    #convert weights to array
    if weights is not None:
        weights = array('f', weights)
    #create root histogram
    if eta == False:
        root_hist = ROOT.TH1D("root_hist", "ROOT Histogram", num_bins, x_min, x_max)
    else:
        bin_array = array('d',[-2.5, -2.2, -1.9, -1.6, -1.3, -1.1, -0.9, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.2, 2.5])
        root_hist = ROOT.TH1D("root_hist", "ROOT Histogram", len(bin_array)-1, bin_array)
    root_hist.Sumw2()
    #fill root histogram
    for i in range(len(data)):
        if weights is None:
            root_hist.Fill(data[i])
        else:
            root_hist.Fill(data[i], weights[i])
    return root_hist

def calc_roc(signal_scores, background_scores, signal_weights=None, background_weights=None):
    # filter scores above 0.4
    # signal_scores = signal_scores[signal_scores > 0.45]
    # background_scores = background_scores[background_scores > 0.45]
    # signal_weights = signal_weights[signal_scores > 0.45]
    # background_weights = background_weights[background_scores > 0.45]
    signal_labels = np.ones(len(signal_scores))
    background_labels = np.zeros(len(background_scores))

    if signal_weights is None and background_weights is None:
        #sort scores and labels
        labels = np.concatenate([np.array(signal_labels), np.array(background_labels)])
        scores = np.concatenate([np.array(signal_scores), np.array(background_scores)])
        fpr, tpr, thresholds = roc_curve(labels, scores)
    else:
        #sort scores and labels and weights
        labels = np.concatenate([np.array(signal_labels), np.array(background_labels)])
        scores = np.concatenate([np.array(signal_scores), np.array(background_scores)])
        weights = np.concatenate([np.array(signal_weights), np.array(background_weights)])
        fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=weights)

    fpr_cut = fpr[(tpr>0.05) & (fpr>0)] 
    tpr_cut = tpr[(tpr>0.05) & (fpr>0)]
    return fpr_cut, tpr_cut

def flattened_pt_weighted(data, bins):
    weights = np.zeros(len(ak.flatten(data))) 
    pt_hist, bin_edges = np.histogram(ak.flatten(data), bins=bins)
    for i in range(len(pt_hist)):
        if pt_hist[i] == 0:
            weights = np.where((ak.flatten(data) >= bin_edges[i]) & (ak.flatten(data) < bin_edges[i+1]), 1, weights)
        else:
            weights = np.where((ak.flatten(data) >= bin_edges[i]) & (ak.flatten(data) < bin_edges[i+1]), 1/pt_hist[i], weights)

    return weights

def plot_eff(data, cuts, name, num_bins, x_min, x_max, weights=None, eta=False):
    if weights is None:
        pt_1p3p_dnom = plt_to_root_hist_w(data[cuts[0]], num_bins, x_min, x_max, None, eta)
        pt_1p3p_num =  plt_to_root_hist_w(data[cuts[1]], num_bins, x_min, x_max, None, eta)
        pt_1p1p_dnom = plt_to_root_hist_w(data[cuts[2]], num_bins, x_min, x_max, None, eta)
        pt_1p1p_num =  plt_to_root_hist_w(data[cuts[3]], num_bins, x_min, x_max, None, eta)
        pt_3p3p_dnom = plt_to_root_hist_w(data[cuts[4]], num_bins, x_min, x_max, None, eta)
        pt_3p3p_num =  plt_to_root_hist_w(data[cuts[5]], num_bins, x_min, x_max, None, eta)
        pt_inc_dnom =  plt_to_root_hist_w(data[cuts[6]], num_bins, x_min, x_max, None, eta)
        pt_inc_num =   plt_to_root_hist_w(data[cuts[7]], num_bins, x_min, x_max, None, eta)
    else:
        pt_1p3p_dnom = plt_to_root_hist_w(data[cuts[0]], num_bins, x_min, x_max, (weights)[cuts[0]], eta)
        pt_1p3p_num =  plt_to_root_hist_w(data[cuts[1]], num_bins, x_min, x_max, (weights)[cuts[1]], eta)
        pt_1p1p_dnom = plt_to_root_hist_w(data[cuts[2]], num_bins, x_min, x_max, (weights)[cuts[2]], eta)
        pt_1p1p_num =  plt_to_root_hist_w(data[cuts[3]], num_bins, x_min, x_max, (weights)[cuts[3]], eta)
        pt_3p3p_dnom = plt_to_root_hist_w(data[cuts[4]], num_bins, x_min, x_max, (weights)[cuts[4]], eta)
        pt_3p3p_num =  plt_to_root_hist_w(data[cuts[5]], num_bins, x_min, x_max, (weights)[cuts[5]], eta)
        pt_inc_dnom =  plt_to_root_hist_w(data[cuts[6]], num_bins, x_min, x_max, (weights)[cuts[6]], eta)
        pt_inc_num =   plt_to_root_hist_w(data[cuts[7]], num_bins, x_min, x_max, (weights)[cuts[7]], eta)

    pt_1p3p_eff = make_eff_hist(pt_1p3p_num, pt_1p3p_dnom, "1p3p_eff")
    pt_1p1p_eff = make_eff_hist(pt_1p1p_num, pt_1p1p_dnom, "1p1p_eff")
    pt_3p3p_eff = make_eff_hist(pt_3p3p_num, pt_3p3p_dnom, "3p3p_eff")
    pt_inc_eff = make_eff_hist(pt_inc_num, pt_inc_dnom, "inc_eff")

    pt_1p3p_eff.GetYaxis().SetRangeUser(0, 1)
    pt_1p1p_eff.GetYaxis().SetRangeUser(0, 1)
    pt_3p3p_eff.GetYaxis().SetRangeUser(0, 1)
    pt_inc_eff.GetYaxis().SetRangeUser(0, 1)

    pt_1p3p_eff.GetXaxis().SetRangeUser(x_min, x_max)
    pt_1p1p_eff.GetXaxis().SetRangeUser(x_min, x_max)
    pt_3p3p_eff.GetXaxis().SetRangeUser(x_min, x_max)
    pt_inc_eff.GetXaxis().SetRangeUser(x_min, x_max)

    pt_1p3p_eff.GetXaxis().SetTitle(name)
    pt_1p1p_eff.GetXaxis().SetTitle(name)
    pt_3p3p_eff.GetXaxis().SetTitle(name)
    pt_inc_eff.GetXaxis().SetTitle(name)
    
    pt_1p3p_eff.SetLineColor(ROOT.kBlack)
    pt_1p1p_eff.SetLineColor(ROOT.kOrange)
    pt_3p3p_eff.SetLineColor(ROOT.kRed)
    pt_inc_eff.SetLineColor(ROOT.kGreen)

    # legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    # if weights is None:
    #     legend.AddEntry(pt_1p3p_eff, "1p3p")
    #     legend.AddEntry(pt_1p1p_eff, "1p1p")
    #     legend.AddEntry(pt_3p3p_eff, "3p3p")
    #     legend.AddEntry(pt_inc_eff, "inclusive")
    # else:
    #     legend.AddEntry(pt_1p3p_eff, "1p3p w")
    #     legend.AddEntry(pt_1p1p_eff, "1p1p w")
    #     legend.AddEntry(pt_3p3p_eff, "3p3p w")
    #     legend.AddEntry(pt_inc_eff, "inclusive w")

    return pt_1p3p_eff, pt_1p1p_eff, pt_3p3p_eff, pt_inc_eff


def plot_branches():
    path = "/global/u2/a/agarabag/pscratch/ditdau_samples/"

    # di_jet_branches = ['DiTauJetsAuxDyn_ditau_pt', 'DiTauJetsAuxDyn_n_subjets', 'DiTauJetsAuxDyn_IsTruthHadronic', 'DiTauJetsAuxDyn_n_tracks_lead', 'DiTauJetsAuxDyn_n_tracks_subl', 'DiTauJetsAuxDyn_BDTScore', 'DiTauJetsAuxDyn_R_max_lead', 'DiTauJetsAuxDyn_R_max_subl', 'DiTauJetsAuxDyn_R_tracks_subl', 'DiTauJetsAuxDyn_R_isotrack', 'DiTauJetsAuxDyn_d0_leadtrack_lead', 'DiTauJetsAuxDyn_d0_leadtrack_subl', 'DiTauJetsAuxDyn_f_core_lead', 'DiTauJetsAuxDyn_f_core_subl', 'DiTauJetsAuxDyn_f_subjet_subl', 'DiTauJetsAuxDyn_f_subjets', 'DiTauJetsAuxDyn_f_isotracks', 'DiTauJetsAuxDyn_m_core_lead', 'DiTauJetsAuxDyn_m_core_subl', 'DiTauJetsAuxDyn_m_tracks_lead', 'DiTauJetsAuxDyn_m_tracks_subl', 'DiTauJetsAuxDyn_n_track', 'EventInfoAuxDyn_averageInteractionsPerCrossing', 'EventInfoAuxDyn_actualInteractionsPerCrossing']
    # branches = ['DiTauJetsAux.eta',
    #             'DiTauJetsAuxDyn.ditau_pt',
    #             'DiTauJetsAuxDyn.n_subjets',
    #             'DiTauJetsAuxDyn.IsTruthHadronic',
    #             'DiTauJetsAuxDyn.n_tracks_lead',
    #             'DiTauJetsAuxDyn.n_tracks_subl',
    #             'DiTauJetsAuxDyn.BDTScore',
    #             'DiTauJetsAuxDyn.R_max_lead', 'DiTauJetsAuxDyn.R_max_subl',
    #             'DiTauJetsAuxDyn.R_tracks_subl', 'DiTauJetsAuxDyn.R_isotrack', 
    #             'DiTauJetsAuxDyn.d0_leadtrack_lead', 'DiTauJetsAuxDyn.d0_leadtrack_subl',
    #             'DiTauJetsAuxDyn.f_core_lead', 'DiTauJetsAuxDyn.f_core_subl', 'DiTauJetsAuxDyn.f_subjet_subl',
    #             'DiTauJetsAuxDyn.f_subjets', 'DiTauJetsAuxDyn.f_isotracks', 
    #             'DiTauJetsAuxDyn.m_core_lead', 'DiTauJetsAuxDyn.m_core_subl', 
    #             'DiTauJetsAuxDyn.m_tracks_lead', 'DiTauJetsAuxDyn.m_tracks_subl', 'DiTauJetsAuxDyn.n_track']

    f_1 = uproot.open('/global/u2/a/agarabag/pscratch/ditdau_samples/graviton.root')
    f1 = f_1['CollectionTree'].arrays(['DiTauJetsAux.eta',
                                       'DiTauJetsAuxDyn.ditau_pt',
                                       'DiTauJetsAuxDyn.n_subjets',
                                       'DiTauJetsAuxDyn.IsTruthHadronic',
                                       'DiTauJetsAuxDyn.n_tracks_lead',
                                       'DiTauJetsAuxDyn.n_tracks_subl',
                                       'DiTauJetsAuxDyn.BDTScore',
                                       'DiTauJetsAuxDyn.R_max_lead', 'DiTauJetsAuxDyn.R_max_subl',
                                       'DiTauJetsAuxDyn.R_tracks_subl', 'DiTauJetsAuxDyn.R_isotrack', 
                                       'DiTauJetsAuxDyn.d0_leadtrack_lead', 'DiTauJetsAuxDyn.d0_leadtrack_subl',
                                       'DiTauJetsAuxDyn.f_core_lead', 'DiTauJetsAuxDyn.f_core_subl', 'DiTauJetsAuxDyn.f_subjet_subl',
                                       'DiTauJetsAuxDyn.f_subjets', 'DiTauJetsAuxDyn.f_isotracks', 
                                       'DiTauJetsAuxDyn.m_core_lead', 'DiTauJetsAuxDyn.m_core_subl', 
                                       'DiTauJetsAuxDyn.m_tracks_lead', 'DiTauJetsAuxDyn.m_tracks_subl', 'DiTauJetsAuxDyn.n_track', 'TruthTausAuxDyn.pt_vis',
                                       'EventInfoAuxDyn.averageInteractionsPerCrossing', 'EventInfoAuxDyn.actualInteractionsPerCrossing', 'EventInfoAuxDyn.mcEventWeights'], library='ak')

    cuts = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
                    (((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 1) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) | 
                    ((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 1) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 3))) )

    cuts_bdt = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
                    (((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 1) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) | 
                    ((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 1) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 3))) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.BDTScore']) >= 0.72))

    cuts_1p1p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 1) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 1))

    cuts_bdt_1p1p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 1) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 1) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.BDTScore']) >= 0.72))

    cuts_3p3p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 3) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 3))

    cuts_bdt_3p3p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 3) & (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 3) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.BDTScore']) >= 0.72))

    cuts_inc = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    ((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 1) | (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 3)) & 
                    ((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 1) | (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1))

    cuts_bdt_inc = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.n_subjets']) >=2) &
                    (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']) <= 1000000) &
                    ((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 1) | (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_lead']) == 3)) & 
                    ((ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 1) | (ak.flatten(f1['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.IsTruthHadronic']) == 1) & 
                    (ak.flatten(f1['DiTauJetsAuxDyn.BDTScore']) >= 0.72))
    
    # file_paths= ['/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_v0_output.root/user.*.output.root',
    #              '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_v0_output.root/user.*.output.root']

    mini_branches= ['DiTauJetsAuxDyn.ditau_pt', 'EventInfoAuxDyn.mcEventWeights', 'DiTauJetsAuxDyn.BDTScore', 
                    'DiTauJetsAuxDyn.n_subjets', 'DiTauJetsAuxDyn.n_tracks_lead', 'DiTauJetsAuxDyn.n_tracks_subl', 'DiTauJetsAux.eta']
    print("tttttt: ", getXS(425102))
    # file_paths = ['/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed*.root']
    # file_paths = ['/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed1.root', 
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed2.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed3.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed4.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed5.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed6.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed7.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed8.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed9.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed10.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed11.root',
    #               '/global/u2/a/agarabag/pscratch/ditdau_samples/di_jet_skimmed12.root']
    # file_paths = ['/global/u2/a/agarabag/pscratch/di_jet_skimmed1.root', 
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed2.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed3.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed4.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed5.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed6.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed7.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed8.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed9.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed10.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed11.root',
    #               '/global/u2/a/agarabag/pscratch/di_jet_skimmed12.root']

    file_paths = [
    'user.agarabag.DiTauMC20.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_v0_output.root/user.agarabag.34455039._000002.output.root',
    'user.agarabag.DiTauMC20.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/user.agarabag.34455043._000002.output.root',
    'user.agarabag.DiTauMC20.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/user.agarabag.34455045._000001.output.root',
    'user.agarabag.DiTauMC20.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/user.agarabag.34455049._000002.output.root',
    'user.agarabag.DiTauMC20.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/user.agarabag.34455051._000001.output.root',
    'user.agarabag.DiTauMC20.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/user.agarabag.34455056._000002.output.root',
    'user.agarabag.DiTauMC20.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/user.agarabag.34455059._000001.output.root',
    'user.agarabag.DiTauMC20.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/user.agarabag.34455061._000001.output.root',
    'user.agarabag.DiTauMC20.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_v0_output.root/user.agarabag.34455064._000002.output.root',
    'user.agarabag.DiTauMC20.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_v0_output.root/user.agarabag.34455068._000001.output.root',
    'user.agarabag.DiTauMC20.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_v0_output.root/user.agarabag.34455072._000001.output.root',
    'user.agarabag.DiTauMC20.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_v0_output.root/user.agarabag.34455076._000001.output.root']
    # 'user.agarabag.DiJetMC20_JZ0.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_v0_output.root/user.agarabag.35047097._000001.output.root']
    file_paths = [path + file_path for file_path in file_paths]

    # c1 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_v0_output.root/user.*.output.root")))
    # c2 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/user.*.output.root")))
    # c3 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/user.*.output.root")))
    # c4 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/user.*.output.root")))
    # c5 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/user.*.output.root")))
    # c6 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/user.*.output.root")))
    # c7 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/user.*.output.root")))
    # c8 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/user.*.output.root")))
    # c9 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_v0_output.root/user.*.output.root")))
    # c10 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_v0_output.root/user.*.output.root")))
    # c11 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_v0_output.root/user.*.output.root")))
    # c12 = getNevents(glob.glob(os.path.join(path, "user.agarabag.DiTauMC20.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_v0_output.root/user.*.output.root")))

    # c0 = getNevents(glob.glob(file_paths[12]))
    c1 = getNevents(glob.glob(file_paths[0]))
    c2 = getNevents(glob.glob(file_paths[1]))
    c3 = getNevents(glob.glob(file_paths[2]))
    c4 = getNevents(glob.glob(file_paths[3]))
    c5 = getNevents(glob.glob(file_paths[4]))
    c6 = getNevents(glob.glob(file_paths[5]))
    c7 = getNevents(glob.glob(file_paths[6]))
    c8 = getNevents(glob.glob(file_paths[7]))
    c9 = getNevents(glob.glob(file_paths[8]))
    c10 = getNevents(glob.glob(file_paths[9]))
    c11 = getNevents(glob.glob(file_paths[10]))
    c12 = getNevents(glob.glob(file_paths[11]))
    
    # jz0 = read_file(file_paths[12], mini_branches)
    jz1 = read_file(file_paths[0], mini_branches)
    jz2 = read_file(file_paths[1], mini_branches)
    jz3 = read_file(file_paths[2], mini_branches)
    jz4 = read_file(file_paths[3], mini_branches)
    jz5 = read_file(file_paths[4], mini_branches)
    jz6 = read_file(file_paths[5], mini_branches)
    jz7 = read_file(file_paths[6], mini_branches)
    jz8 = read_file(file_paths[7], mini_branches)
    jz9 = read_file(file_paths[8], mini_branches)
    jz10 = read_file(file_paths[9], mini_branches)
    jz11 = read_file(file_paths[10], mini_branches)
    jz12 = read_file(file_paths[11], mini_branches)


    p = PdfPages("histogram.pdf") 


    # all_data, all_weights = plot_root_files(file_paths, mini_branches)
    # fig_dijet_comb2 = plt.figure()
    # bin_counts, bin_edges = np.histogram(all_data, bins=bins, weights=all_weights)
    # bin_uncertainties = np.sqrt(bin_counts)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.hist(all_data, bins=bins, weights=all_weights, histtype="step")
    # plt.errorbar(bin_centers, bin_counts, yerr=bin_uncertainties, fmt='none', ecolor='black', capsize=2)
    # plt.xlabel("ditau_pt")
    # plt.ylabel("Counts")
    # plt.yscale('log')
    # p.savefig(fig_dijet_comb2)
    # plt.close(fig_dijet_comb2)  

    pt_name = 'DiTauJetsAuxDyn.ditau_pt'
    nkg_eta = 'DiTauJetsAux.eta'
    n_sub_jet = 'DiTauJetsAuxDyn.n_subjets'
    n_tracks_lead = 'DiTauJetsAuxDyn.n_tracks_lead'
    n_tracks_subl = 'DiTauJetsAuxDyn.n_tracks_subl'
    bdt_score = 'DiTauJetsAuxDyn.BDTScore'
    #print(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12)
    #print(event_weight_sum(jz1), event_weight_sum(jz2), event_weight_sum(jz3), event_weight_sum(jz4), event_weight_sum(jz5), event_weight_sum(jz6), event_weight_sum(jz7), event_weight_sum(jz8), event_weight_sum(jz9), event_weight_sum(jz10), event_weight_sum(jz11), event_weight_sum(jz12))
    # print(getXS(364701), getXS(364702), getXS(364703), getXS(364704), getXS(364705), getXS(364706), getXS(364707), getXS(364708), getXS(364709), getXS(364710), getXS(364711), getXS(364712))

    data = np.concatenate([
    # ak.flatten(jz0[pt_name]),
    ak.flatten(jz1[pt_name]),
    ak.flatten(jz2[pt_name]),
    ak.flatten(jz3[pt_name]),
    ak.flatten(jz4[pt_name]),
    ak.flatten(jz5[pt_name]),
    ak.flatten(jz6[pt_name]),
    ak.flatten(jz7[pt_name]),
    ak.flatten(jz8[pt_name]),
    ak.flatten(jz9[pt_name]),
    ak.flatten(jz10[pt_name]),
    ak.flatten(jz11[pt_name]),
    ak.flatten(jz12[pt_name])
    ])

    data_un_flat = np.concatenate([
    jz1[pt_name],
    jz2[pt_name],
    jz3[pt_name],
    jz4[pt_name],
    jz5[pt_name],
    jz6[pt_name],
    jz7[pt_name],
    jz8[pt_name],
    jz9[pt_name],
    jz10[pt_name],
    jz11[pt_name],
    jz12[pt_name]
    ])

    data_eta = np.concatenate([
    ak.flatten(jz1[nkg_eta]),
    ak.flatten(jz2[nkg_eta]),
    ak.flatten(jz3[nkg_eta]),
    ak.flatten(jz4[nkg_eta]),
    ak.flatten(jz5[nkg_eta]),
    ak.flatten(jz6[nkg_eta]),
    ak.flatten(jz7[nkg_eta]),
    ak.flatten(jz8[nkg_eta]),
    ak.flatten(jz9[nkg_eta]),
    ak.flatten(jz10[nkg_eta]),
    ak.flatten(jz11[nkg_eta]),
    ak.flatten(jz12[nkg_eta])
    ])
    
    bkg_n_subj = np.concatenate([
    ak.flatten(jz1[n_sub_jet]),
    ak.flatten(jz2[n_sub_jet]),
    ak.flatten(jz3[n_sub_jet]),
    ak.flatten(jz4[n_sub_jet]),
    ak.flatten(jz5[n_sub_jet]),
    ak.flatten(jz6[n_sub_jet]),
    ak.flatten(jz7[n_sub_jet]),
    ak.flatten(jz8[n_sub_jet]),
    ak.flatten(jz9[n_sub_jet]),
    ak.flatten(jz10[n_sub_jet]),
    ak.flatten(jz11[n_sub_jet]),
    ak.flatten(jz12[n_sub_jet])
    ])

    bkg_t_lead = np.concatenate([
    ak.flatten(jz1[n_tracks_lead]),
    ak.flatten(jz2[n_tracks_lead]),
    ak.flatten(jz3[n_tracks_lead]),
    ak.flatten(jz4[n_tracks_lead]),
    ak.flatten(jz5[n_tracks_lead]),
    ak.flatten(jz6[n_tracks_lead]),
    ak.flatten(jz7[n_tracks_lead]),
    ak.flatten(jz8[n_tracks_lead]),
    ak.flatten(jz9[n_tracks_lead]),
    ak.flatten(jz10[n_tracks_lead]),
    ak.flatten(jz11[n_tracks_lead]),
    ak.flatten(jz12[n_tracks_lead])
    ])

    bkg_t_subl = np.concatenate([
    ak.flatten(jz1[n_tracks_subl]),
    ak.flatten(jz2[n_tracks_subl]),
    ak.flatten(jz3[n_tracks_subl]),
    ak.flatten(jz4[n_tracks_subl]),
    ak.flatten(jz5[n_tracks_subl]),
    ak.flatten(jz6[n_tracks_subl]),
    ak.flatten(jz7[n_tracks_subl]),
    ak.flatten(jz8[n_tracks_subl]),
    ak.flatten(jz9[n_tracks_subl]),
    ak.flatten(jz10[n_tracks_subl]),
    ak.flatten(jz11[n_tracks_subl]),
    ak.flatten(jz12[n_tracks_subl])
    ])

    bkg_bdt = np.concatenate([
    ak.flatten(jz1[bdt_score]),
    ak.flatten(jz2[bdt_score]),
    ak.flatten(jz3[bdt_score]),
    ak.flatten(jz4[bdt_score]),
    ak.flatten(jz5[bdt_score]),
    ak.flatten(jz6[bdt_score]),
    ak.flatten(jz7[bdt_score]),
    ak.flatten(jz8[bdt_score]),
    ak.flatten(jz9[bdt_score]),
    ak.flatten(jz10[bdt_score]),
    ak.flatten(jz11[bdt_score]),
    ak.flatten(jz12[bdt_score])
    ])


    data_stack = [
    # ak.flatten(jz0[pt_name]),
    ak.flatten(jz1[pt_name]),
    ak.flatten(jz2[pt_name]),
    ak.flatten(jz3[pt_name]),
    ak.flatten(jz4[pt_name]),
    ak.flatten(jz5[pt_name]),
    ak.flatten(jz6[pt_name]),
    ak.flatten(jz7[pt_name]),
    ak.flatten(jz8[pt_name]),
    ak.flatten(jz9[pt_name]),
    ak.flatten(jz10[pt_name]),
    ak.flatten(jz11[pt_name]),
    ak.flatten(jz12[pt_name])
    ]
    # data_scaled = np.concatenate([
    # ak.flatten(jz1[pt_name])*getXS(364701)/c1,
    # ak.flatten(jz2[pt_name])*getXS(364702)/c2,
    # ak.flatten(jz3[pt_name])*getXS(364703)/c3,
    # ak.flatten(jz4[pt_name])*getXS(364704)/c4,
    # ak.flatten(jz5[pt_name])*getXS(364705)/c5,
    # ak.flatten(jz6[pt_name])*getXS(364706)/c6,
    # ak.flatten(jz7[pt_name])*getXS(364707)/c7,
    # ak.flatten(jz8[pt_name])*getXS(364708)/c8,
    # ak.flatten(jz9[pt_name])*getXS(364709)/c9,
    # ak.flatten(jz10[pt_name])*getXS(364710)/c10,
    # ak.flatten(jz11[pt_name])*getXS(364711)/c11,
    # ak.flatten(jz12[pt_name])*getXS(364712)/c12
    # ])
    # bkg_evt_weights = np.concatenate([
    #     event_weight(jz1)*getXS(364701)/c1,
    #     event_weight(jz2)*getXS(364702)/c2,
    #     event_weight(jz3)*getXS(364703)/c3,
    #     event_weight(jz4)*getXS(364704)/c4,
    #     event_weight(jz5)*getXS(364705)/c5,
    #     event_weight(jz6)*getXS(364706)/c6,
    #     event_weight(jz7)*getXS(364707)/c7,
    #     event_weight(jz8)*getXS(364708)/c8,
    #     event_weight(jz9)*getXS(364709)/c9,
    #     event_weight(jz10)*getXS(364710)/c10,
    #     event_weight(jz11)*getXS(364711)/c11,
    #     event_weight(jz12)*getXS(364712)/c12
    # ])
    
    bkg_evt_weights = np.concatenate([
        # event_weight(jz0)*getXS(364700)/event_weight_sum(jz0),
        event_weight(jz1)*getXS(364701)/event_weight_sum(jz1),
        event_weight(jz2)*getXS(364702)/event_weight_sum(jz2),
        event_weight(jz3)*getXS(364703)/event_weight_sum(jz3),
        event_weight(jz4)*getXS(364704)/event_weight_sum(jz4),
        event_weight(jz5)*getXS(364705)/event_weight_sum(jz5),
        event_weight(jz6)*getXS(364706)/event_weight_sum(jz6),
        event_weight(jz7)*getXS(364707)/event_weight_sum(jz7),
        event_weight(jz8)*getXS(364708)/event_weight_sum(jz8),
        event_weight(jz9)*getXS(364709)/event_weight_sum(jz9),
        event_weight(jz10)*getXS(364710)/event_weight_sum(jz10),
        event_weight(jz11)*getXS(364711)/event_weight_sum(jz11),
        event_weight(jz12)*getXS(364712)/event_weight_sum(jz12)
    ])

    # bkg_evt_weights = np.concatenate([
    #     event_weight(jz1),
    #     event_weight(jz2),
    #     event_weight(jz3),
    #     event_weight(jz4),
    #     event_weight(jz5),
    #     event_weight(jz6),
    #     event_weight(jz7),
    #     event_weight(jz8),
    #     event_weight(jz9),
    #     event_weight(jz10),
    #     event_weight(jz11),
    #     event_weight(jz12)
    # ])

    pt_weights_stack = [
        # event_weight(jz0)*getXS(364700)/event_weight_sum(jz0),
        event_weight(jz1)*getXS(364701)/event_weight_sum(jz1),
        event_weight(jz2)*getXS(364702)/event_weight_sum(jz2),
        event_weight(jz3)*getXS(364703)/event_weight_sum(jz3),
        event_weight(jz4)*getXS(364704)/event_weight_sum(jz4),
        event_weight(jz5)*getXS(364705)/event_weight_sum(jz5),
        event_weight(jz6)*getXS(364706)/event_weight_sum(jz6),
        event_weight(jz7)*getXS(364707)/event_weight_sum(jz7),
        event_weight(jz8)*getXS(364708)/event_weight_sum(jz8),
        event_weight(jz9)*getXS(364709)/event_weight_sum(jz9),
        event_weight(jz10)*getXS(364710)/event_weight_sum(jz10),
        event_weight(jz11)*getXS(364711)/event_weight_sum(jz11),
        event_weight(jz12)*getXS(364712)/event_weight_sum(jz12)
    ]
    # pt_weights2 = np.concatenate([
    #     np.full(len(ak.flatten(jz1[pt_name])), getXS(364701)/c1),
    #     np.full(len(ak.flatten(jz2[pt_name])), getXS(364702)/c2),
    #     np.full(len(ak.flatten(jz3[pt_name])), getXS(364703)/c3),
    #     np.full(len(ak.flatten(jz4[pt_name])), getXS(364704)/c4),
    #     np.full(len(ak.flatten(jz5[pt_name])), getXS(364705)/c5),
    #     np.full(len(ak.flatten(jz6[pt_name])), getXS(364706)/c6),
    #     np.full(len(ak.flatten(jz7[pt_name])), getXS(364707)/c7),
    #     np.full(len(ak.flatten(jz8[pt_name])), getXS(364708)/c8),
    #     np.full(len(ak.flatten(jz9[pt_name])), getXS(364709)/c9),
    #     np.full(len(ak.flatten(jz10[pt_name])), getXS(364710)/c10),
    #     np.full(len(ak.flatten(jz11[pt_name])), getXS(364711)/c11),
    #     np.full(len(ak.flatten(jz12[pt_name])), getXS(364712)/c12)
    # ])
    # pt_weights_stack2 = [
    #     np.full(len(ak.flatten(jz1[pt_name])), getXS(364701)/c1),
    #     np.full(len(ak.flatten(jz2[pt_name])), getXS(364702)/c2),
    #     np.full(len(ak.flatten(jz3[pt_name])), getXS(364703)/c3),
    #     np.full(len(ak.flatten(jz4[pt_name])), getXS(364704)/c4),
    #     np.full(len(ak.flatten(jz5[pt_name])), getXS(364705)/c5),
    #     np.full(len(ak.flatten(jz6[pt_name])), getXS(364706)/c6),
    #     np.full(len(ak.flatten(jz7[pt_name])), getXS(364707)/c7),
    #     np.full(len(ak.flatten(jz8[pt_name])), getXS(364708)/c8),
    #     np.full(len(ak.flatten(jz9[pt_name])), getXS(364709)/c9),
    #     np.full(len(ak.flatten(jz10[pt_name])), getXS(364710)/c10),
    #     np.full(len(ak.flatten(jz11[pt_name])), getXS(364711)/c11),
    #     np.full(len(ak.flatten(jz12[pt_name])), getXS(364712)/c12)
    # ]

    # bkg_pt_bins = np.linspace(200000, 1000000, 41)
    bins = np.linspace(200000, 1000000, 41)


    bkg_cuts = ak.where((bkg_n_subj >=2) & 
                (data >= 200000) & (data <= 1000000) &
                (((bkg_t_lead == 1) & (bkg_t_subl == 3)) | 
                ((bkg_t_subl == 1) & (bkg_t_lead == 3))) )
    bkg_cuts_bdt = ak.where((bkg_n_subj >=2) &
                (data >= 200000) & (data <= 1000000) &
                (((bkg_t_lead == 1) & (bkg_t_subl == 3)) | 
                ((bkg_t_subl == 1) & (bkg_t_lead == 3))) &
                (bkg_bdt > 0.55))
    bkg_cuts_1p1p = ak.where((bkg_n_subj >=2) &
                (data >= 200000) & (data <= 1000000) &
                (bkg_t_lead == 1) & 
                (bkg_t_subl == 1))
    bkg_cuts_1p1p_bdt = ak.where((bkg_n_subj >=2) &
                (data >= 200000) & (data <= 1000000) &
                (bkg_t_lead == 1) & 
                (bkg_t_subl == 1) &
                (bkg_bdt > 0.55))
    bkg_cuts_3p3p = ak.where((bkg_n_subj >=2) &
                (data >= 200000) & (data <= 1000000) &
                (bkg_t_lead == 3) & 
                (bkg_t_subl == 3))
    bkg_cuts_3p3p_bdt = ak.where((bkg_n_subj >=2) &
                (data >= 200000) & (data <= 1000000) &
                (bkg_t_lead == 3) & 
                (bkg_t_subl == 3) &
                (bkg_bdt > 0.55))
    bkg_cuts_inc = ak.where((bkg_n_subj >= 2) &
                (data >= 200000) & (data <= 1000000) &
                ((bkg_t_lead == 1) | (bkg_t_lead == 3)) & 
                ((bkg_t_subl == 1) | (bkg_t_subl == 3)) )
    bkg_cuts_inc_bdt = ak.where((bkg_n_subj >=2) & (bkg_bdt > 0.55) &
                (data >= 200000) & (data <= 1000000) &
                ((bkg_t_lead == 1) | (bkg_t_lead == 3)) & 
                ((bkg_t_subl == 1) | (bkg_t_subl == 3)) )

    bkg_weights = flattened_pt_weighted(data_un_flat, bins)

    bkg_1p3p_pt_eff = calculate_efficiency(data_un_flat, bins, bkg_cuts, bkg_cuts_bdt)
    bkg_1p1p_pt_eff = calculate_efficiency(data_un_flat, bins, bkg_cuts_1p1p, bkg_cuts_1p1p_bdt)
    bkg_3p3p_pt_eff = calculate_efficiency(data_un_flat, bins, bkg_cuts_3p3p, bkg_cuts_3p3p_bdt)
    bkg_inc_pt_eff = calculate_efficiency(data_un_flat, bins, bkg_cuts_inc, bkg_cuts_inc_bdt)
    bkg_1p3p_pt_eff_w = calculate_efficiency(data_un_flat, bins, bkg_cuts, bkg_cuts_bdt, bkg_weights)
    bkg_1p1p_pt_eff_w = calculate_efficiency(data_un_flat, bins, bkg_cuts_1p1p, bkg_cuts_1p1p_bdt, bkg_weights)
    bkg_3p3p_pt_eff_w = calculate_efficiency(data_un_flat, bins, bkg_cuts_3p3p, bkg_cuts_3p3p_bdt, bkg_weights)
    bkg_inc_pt_eff_w = calculate_efficiency(data_un_flat, bins, bkg_cuts_inc, bkg_cuts_inc_bdt, bkg_weights)

    bkg_eff = plt.figure()
    plt.plot(bins[:-1], bkg_1p3p_pt_eff, label="1p3p unweighted", color='black')
    plt.plot(bins[:-1], bkg_1p1p_pt_eff, label="1p1p unweighted", color='orange')
    plt.plot(bins[:-1], bkg_3p3p_pt_eff, label="3p3p unweighted", color='red')
    plt.plot(bins[:-1], bkg_inc_pt_eff, label="inclusive unweighted", color='green')
    plt.plot(bins[:-1], bkg_1p3p_pt_eff_w, label="1p3p weights", linestyle='dashed', color='black')
    plt.plot(bins[:-1], bkg_1p1p_pt_eff_w, label="1p1p weights", linestyle='dashed', color='orange')
    plt.plot(bins[:-1], bkg_3p3p_pt_eff_w, label="3p3p weights", linestyle='dashed', color='red')
    plt.plot(bins[:-1], bkg_inc_pt_eff_w, label="inclusive weights", linestyle='dashed', color='green')
    plt.xlabel("dijet_pt")
    plt.ylabel("efficiency")
    plt.legend()
    p.savefig(bkg_eff)
    plt.close(bkg_eff)


    fig_dijet_stack = plt.figure()
    plt.hist(data_stack, bins=bins, stacked=True, weights=pt_weights_stack, color=cm.tab20(range(12)), histtype="step", label=["jZ1", "jZ2", "jZ3", "jZ4", "jZ5", "jZ6", "jZ7", "jZ8", "jZ9", "jZ10", "jZ11", "jZ12"])
    plt.xlabel("ditau_pt")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_dijet_stack)
    plt.close(fig_dijet_stack)

    # fig_dijet_stack2 = plt.figure()
    # plt.hist(data_stack, bins=pt_bins, stacked=True, weights=pt_weights_stack2, color=cm.tab20(range(12)), histtype="step", label=["jZ1", "jZ2", "jZ3", "jZ4", "jZ5", "jZ6", "jZ7", "jZ8", "jZ9", "jZ10", "jZ11", "jZ12"])
    # plt.xlabel("ditau_pt")
    # plt.ylabel("Counts")
    # plt.yscale('log')
    # plt.legend()
    # p.savefig(fig_dijet_stack2)
    # plt.close(fig_dijet_stack2)

    # Create the histogram with combined data and weights
    fig_dijet_comb = plt.figure()
    bin_counts1, bin_edges1 = np.histogram(data, bins=bins, weights=bkg_evt_weights)
    bin_uncertainties1 = np.sqrt(bin_counts1)
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    plt.hist(data, bins=bins, histtype="step", label="background", weights=bkg_evt_weights)
    #plot signal on top
    # plt.hist(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']), bins=bins, histtype="step", label="signal")
    plt.errorbar(bin_centers1, bin_counts1, yerr=bin_uncertainties1, fmt='none', ecolor='black', capsize=2)
    plt.xlabel("ditau_pt")
    plt.ylabel("Counts")
    plt.yscale('log') 
    plt.legend()
    p.savefig(fig_dijet_comb)
    plt.close(fig_dijet_comb)

    bk_pt_plt = plt_to_root_hist_w(data[bkg_cuts], 100, 100000, 3000000, bkg_evt_weights[bkg_cuts])
    cb = ROOT.TCanvas("cb", "cb", 800, 600)
    bk_pt_plt.Draw("hist e")
    cb.SetLogy()
    cb.SaveAs("bk_pt_plt.pdf")
    cb.Close()

    # bk_pt_plt2 = plt_to_root_hist_w(data, 100, 200000, 3000000, bkg_weights)
    # cb2 = ROOT.TCanvas("cb2", "cb2", 800, 600)
    # bk_pt_plt2.Draw("hist e")
    # cb2.SetLogy()
    # cb2.SaveAs("bk_pt_plt2.pdf")
    # cb2.Close()

    # fig_dijet_comb2 = plt.figure()
    # bin_counts, bin_edges = np.histogram(data, bins=bins, weights=pt_weights2)
    # bin_uncertainties = np.sqrt(bin_counts)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.hist(data, bins=bins, weights=pt_weights2, histtype="step")
    # plt.errorbar(bin_centers, bin_counts, yerr=bin_uncertainties, fmt='none', ecolor='black', capsize=2)
    # plt.xlabel("ditau_pt")
    # plt.ylabel("Counts")
    # plt.yscale('log')
    # p.savefig(fig_dijet_comb2)
    # plt.close(fig_dijet_comb2)
    

    weights = flattened_pt_weighted(f1['DiTauJetsAuxDyn.ditau_pt'], bins)

    # print(len(f1['EventInfoAuxDyn.averageInteractionsPerCrossing']))
    # print(len(f1['DiTauJetsAuxDyn.ditau_pt']))
    # print(len(ak.flatten(f1['DiTauJetsAuxDyn.n_subjets'])))


    # #make figure and plot the ditau_pt distribution then save it
    # fig1 = plt.figure()
    # plt.hist(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']), bins=bins, histtype="step")
    # plt.xlabel("DiTauJetsAuxDyn.ditau_pt")
    # plt.ylabel("Counts")
    # p.savefig(fig1)


    # print(len(f1['DiTauJetsAuxDyn.ditau_pt']))
    # #plot the size of each sub array in the ditau_pt array
    # fig2 = plt.figure()
    # plt.hist([len(x) for x in f1['DiTauJetsAuxDyn.ditau_pt']], bins=50, histtype="step")
    # plt.xlabel("# DiTau's")
    # plt.ylabel("Counts")
    # p.savefig(fig2)

    # fig3 = plt.figure()
    # plt.hist(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']), bins=bins, histtype="step", weights=weights)
    # plt.xlabel("DiTauJetsAuxDyn.ditau_pt flattened")
    # plt.ylabel("Counts")
    # p.savefig(fig3)


    # plot the list of branches before and after applying the weights on same hitogram
    # for branch in branches:
    #     if branch == 'DiTauJetsAuxDyn.ditau_pt':
    #         fig = plt.figure()
    #         plt.hist(ak.flatten(f1[branch]), bins=bins, density=True, histtype="step", label="no cuts")
    #         plt.hist(ak.flatten(f1[branch]), bins=bins, density=True, histtype="step", label="no cuts (weighted)", weights=weights)
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=bins, density=True, histtype="step", label="num cuts (not weighted)",)
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=bins, density=True, histtype="step", weights=weights[cuts], label="num cuts (weighted)")
    #         plt.hist(ak.flatten(f1[branch])[cuts_bdt], bins=bins, density=True, histtype="step", label="denom cuts (not weighted)")
    #         plt.hist(ak.flatten(f1[branch])[cuts_bdt], bins=bins, density=True, histtype="step", weights=weights[cuts_bdt], label="denom cuts (weighted)")
    #         plt.xlabel(branch)
    #         plt.ylabel("Counts")
    #         plt.legend()
    #         p.savefig(fig)
    #         plt.close(fig)
    #     elif branch == 'DiTauJetsAuxDyn.f_isotracks':
    #         fig = plt.figure()
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=np.linspace(0, 0.03, 41), density=True, histtype="step", label="before")
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=np.linspace(0, 0.03, 41), density=True, histtype="step", weights=weights[cuts], label="after")
    #         plt.xlabel(branch)
    #         plt.ylabel("Counts")
    #         plt.legend()
    #         p.savefig(fig)  
    #         plt.close(fig)
    #     elif branch == 'DiTauJetsAuxDyn.m_core_lead' or branch == 'DiTauJetsAuxDyn.m_core_subl' or branch == 'DiTauJetsAuxDyn.m_tracks_lead' or branch == 'DiTauJetsAuxDyn.m_tracks_subl':
    #         fig = plt.figure()
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=np.linspace(0, 5000, 41), density=True, histtype="step", label="before")
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=np.linspace(0, 5000, 41), density=True, histtype="step", weights=weights[cuts], label="after")
    #         plt.yscale('log')
    #         plt.xlabel(branch)
    #         plt.ylabel("Counts")
    #         plt.legend()
    #         p.savefig(fig)
    #         plt.close(fig)  
    #     else:
    #         fig = plt.figure()
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=50, density=True, histtype="step", label="before")
    #         plt.hist(ak.flatten(f1[branch])[cuts], bins=50, density=True, histtype="step", weights=weights[cuts], label="after")
    #         plt.xlabel(branch)
    #         plt.ylabel("Counts")
    #         plt.legend()
    #         p.savefig(fig)
    #         plt.close(fig)

    
    #plot a effiency plot with y axis as the ratio cuts_bdt to cuts and x axis as ditau_pt
    # denom, dnom_bin_edges = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts], bins=bins)
    # num, num_bin_edges = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_bdt], bins=bins)
    # denom = np.array(denom).astype(float)
    # num = np.array(num).astype(float)
    # efficiency = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0).astype(float)

    # #with wights 
    # denom_w, dnom_bin_edges_w = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts], bins=bins, weights=weights[cuts])
    # num_w, num_bin_edges_w = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_bdt], bins=bins, weights=weights[cuts_bdt])
    # denom_w = np.array(denom_w).astype(float)
    # num_w = np.array(num_w).astype(float)
    # efficiency_w = np.divide(num_w, denom_w, out=np.zeros_like(num_w), where=denom_w!=0).astype(float)

    pt_eff = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts, cuts_bdt)
    pt_eff_weighted = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts, cuts_bdt, weights)
    pt_eff_1p1p = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_1p1p, cuts_bdt_1p1p)
    pt_eff_1p1p_weighted = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_1p1p, cuts_bdt_1p1p, weights)
    pt_eff_3p3p = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_3p3p, cuts_bdt_3p3p)
    pt_eff_3p3p_weighted = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_3p3p, cuts_bdt_3p3p, weights)
    pt_eff_inc = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_inc, cuts_bdt_inc)
    pt_eff_inc_weighted = calculate_efficiency(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_inc, cuts_bdt_inc, weights)


    fig4 = plt.figure()
    plt.plot(bins[:-1], pt_eff, label="1p3p unweighted", color='black')
    plt.plot(bins[:-1], pt_eff_1p1p, label="1p1p unweighted", color='orange')
    plt.plot(bins[:-1], pt_eff_3p3p, label="3p3p unweighted", color='red')
    plt.plot(bins[:-1], pt_eff_inc, label="inclusive unweighted", color='green')
    plt.plot(bins[:-1], pt_eff_weighted, label="1p3p weights", linestyle='dashed', color='black')
    plt.plot(bins[:-1], pt_eff_1p1p_weighted, label="1p1p weights", linestyle='dashed', color='orange')
    plt.plot(bins[:-1], pt_eff_3p3p_weighted, label="3p3p weights", linestyle='dashed', color='red')
    plt.plot(bins[:-1], pt_eff_inc_weighted, label="inclusive weights", linestyle='dashed', color='green')
    plt.xlabel("ditau_pt")
    plt.ylabel("efficiency")
    plt.legend()
    p.savefig(fig4)
    plt.close(fig4)




    # plot eta histogram
    # fig_eta = plt.figure()
    # plt.hist(ak.flatten(f1['DiTauJetsAux.eta']), bins=50, histtype="step")
    # plt.xlabel("ditau_eta")
    # plt.ylabel("Counts")
    # p.savefig(fig_eta)
    # plt.close(fig_eta)

    #plot average interactions per crossing
    # fig_mu = plt.figure()
    # plt.hist(f1['EventInfoAuxDyn.averageInteractionsPerCrossing'], bins=40, histtype="step")
    # plt.xlabel("average mu")
    # plt.ylabel("Counts")
    # p.savefig(fig_mu)
    # plt.close(fig_mu)

    eta_bins = np.linspace(-2.5, 2.5, 21)
    # denom_eta, dnom_bin_edges_eta = np.histogram(ak.flatten(f1['DiTauJetsAux.eta'])[cuts], bins=eta_bins)
    # num_eta, num_bin_edges_eta = np.histogram(ak.flatten(f1['DiTauJetsAux.eta'])[cuts_bdt], bins=eta_bins)
    # denom_eta = np.array(denom_eta).astype(float)
    # num_eta = np.array(num_eta).astype(float)
    # efficiency_eta = np.divide(num_eta, denom_eta, out=np.zeros_like(num_eta), where=denom_eta!=0).astype(float)
    # denom_eta_weighted, dnom_bin_edges_eta_weighted = np.histogram(ak.flatten(f1['DiTauJetsAux.eta'])[cuts], bins=eta_bins, weights=weights[cuts])
    # num_eta_weighted, num_bin_edges_eta_weighted = np.histogram(ak.flatten(f1['DiTauJetsAux.eta'])[cuts_bdt], bins=eta_bins, weights=weights[cuts_bdt])
    # denom_eta_weighted = np.array(denom_eta_weighted).astype(float)
    # num_eta_weighted = np.array(num_eta_weighted).astype(float)
    # efficiency_eta_weighted = np.divide(num_eta_weighted, denom_eta_weighted, out=np.zeros_like(num_eta_weighted), where=denom_eta_weighted!=0).astype(float)

    eta_eff = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts, cuts_bdt)
    eta_eff_weighted = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts, cuts_bdt, weights)
    eta_eff_1p1p = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts_1p1p, cuts_bdt_1p1p)
    eta_eff_1p1p_weighted = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts_1p1p, cuts_bdt_1p1p, weights)
    eta_eff_3p3p = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts_3p3p, cuts_bdt_3p3p)
    eta_eff_3p3p_weighted = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts_3p3p, cuts_bdt_3p3p, weights)
    eta_eff_inc = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts_inc, cuts_bdt_inc)
    eta_eff_inc_weighted = calculate_efficiency(f1['DiTauJetsAux.eta'], eta_bins, cuts_inc, cuts_bdt_inc, weights)

    fig5 = plt.figure()
    plt.plot(eta_bins[:-1], eta_eff, label="1p3p unweighted", color='black')
    plt.plot(eta_bins[:-1], eta_eff_1p1p, label="1p1p unweighted", color='orange')
    plt.plot(eta_bins[:-1], eta_eff_3p3p, label="3p3p unweighted", color='red')
    plt.plot(eta_bins[:-1], eta_eff_inc, label="inclusive unweighted", color='green')
    plt.plot(eta_bins[:-1], eta_eff_weighted, label="1p3p weights", linestyle='dashed', color='black')
    plt.plot(eta_bins[:-1], eta_eff_1p1p_weighted, label="1p1p weights", linestyle='dashed', color='orange')
    plt.plot(eta_bins[:-1], eta_eff_3p3p_weighted, label="3p3p weights", linestyle='dashed', color='red')
    plt.plot(eta_bins[:-1], eta_eff_inc_weighted, label="inclusive weights", linestyle='dashed', color='green')
    # plt.scatter(eta_bins[:-1], efficiency_eta, label="not weights")
    # plt.scatter(eta_bins[:-1], efficiency_eta_weighted, label="weights", linestyle='dashed')
    plt.xlabel("ditau_eta")
    plt.ylabel("efficiency")
    plt.legend()
    p.savefig(fig5)
    plt.close(fig5)
    

    # we now this: 
    # print(len(f1['EventInfoAuxDyn.averageInteractionsPerCrossing'])) = 200000
    # print(len(f1['DiTauJetsAuxDyn.ditau_pt'])) = 200000
    # print(len(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']))) = 340018
    # the first two numbers reflect the number of events in the file
    # the last number is the number of subjets in the file
    # for each subjet in each event want assign the same mu value to it 
    # so we need to find a way to assign the same mu value to each subjet in each event

    # we can do this by making a new array with the same length as the subjet array
    new_mu = np.zeros(len(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])))
    # and then assign the same mu value in each event to each subjet in that event
    for i, subjets in enumerate(f1['DiTauJetsAuxDyn.ditau_pt']):
        # Check how many subjets are in each event
        if len(subjets) <= 1:
            # Assign the same mu value to single subjets in the event
            new_mu[i] = f1['EventInfoAuxDyn.averageInteractionsPerCrossing'][i]
        else:
            # Assign the same mu value to each subjet in the event
            new_mu[i:i+len(subjets)] = f1['EventInfoAuxDyn.averageInteractionsPerCrossing'][i]
    
    mu_bin = np.linspace(18, 74, 32)

    #plot new_mu and its weighted version
    # fig_mu_new = plt.figure()
    # plt.hist(new_mu[cuts], bins=mu_bin, histtype="step", density=True, label="not weighted")
    # plt.hist(new_mu[cuts_bdt], bins=mu_bin, histtype="step", density=True, label="weighted")
    # plt.xlabel("average mu")
    # plt.ylabel("Counts")
    # plt.legend()
    # p.savefig(fig_mu_new)
    # plt.close(fig_mu_new)

    denom_mu, dnom_bin_edges_mu = np.histogram(new_mu[cuts], bins=mu_bin)
    num_mu, num_bin_edges_mu = np.histogram(new_mu[cuts_bdt], bins=mu_bin)
    denom_mu = np.array(denom_mu).astype(float)   
    num_mu = np.array(num_mu).astype(float)
    efficiency_mu = np.divide(num_mu, denom_mu, out=np.zeros_like(num_mu), where=denom_mu!=0).astype(float)

    denom_mu_1p1p, dnom_bin_edges_mu_1p1p = np.histogram(new_mu[cuts_1p1p], bins=mu_bin)
    num_mu_1p1p, num_bin_edges_mu_1p1p = np.histogram(new_mu[cuts_bdt_1p1p], bins=mu_bin)
    denom_mu_1p1p = np.array(denom_mu_1p1p).astype(float)
    num_mu_1p1p = np.array(num_mu_1p1p).astype(float)
    efficiency_mu_1p1p = np.divide(num_mu_1p1p, denom_mu_1p1p, out=np.zeros_like(num_mu_1p1p), where=denom_mu_1p1p!=0).astype(float)

    denom_mu_3p3p, dnom_bin_edges_mu_3p3p = np.histogram(new_mu[cuts_3p3p], bins=mu_bin)
    num_mu_3p3p, num_bin_edges_mu_3p3p = np.histogram(new_mu[cuts_bdt_3p3p], bins=mu_bin)
    denom_mu_3p3p = np.array(denom_mu_3p3p).astype(float)
    num_mu_3p3p = np.array(num_mu_3p3p).astype(float)
    efficiency_mu_3p3p = np.divide(num_mu_3p3p, denom_mu_3p3p, out=np.zeros_like(num_mu_3p3p), where=denom_mu_3p3p!=0).astype(float)

    denom_mu_inc, dnom_bin_edges_mu_inc = np.histogram(new_mu[cuts_inc], bins=mu_bin)
    num_mu_inc, num_bin_edges_mu_inc = np.histogram(new_mu[cuts_bdt_inc], bins=mu_bin)
    denom_mu_inc = np.array(denom_mu_inc).astype(float)
    num_mu_inc = np.array(num_mu_inc).astype(float)
    efficiency_mu_inc = np.divide(num_mu_inc, denom_mu_inc, out=np.zeros_like(num_mu_inc), where=denom_mu_inc!=0).astype(float)

    denom_mu_w, dnom_bin_edges_mu_w = np.histogram(new_mu[cuts], bins=mu_bin, weights=weights[cuts])
    num_mu_w, num_bin_edges_mu_w = np.histogram(new_mu[cuts_bdt], bins=mu_bin, weights=weights[cuts_bdt])
    denom_mu_w = np.array(denom_mu_w).astype(float)
    num_mu_w = np.array(num_mu_w).astype(float)
    efficiency_mu_w = np.divide(num_mu_w, denom_mu_w, out=np.zeros_like(num_mu_w), where=denom_mu_w!=0).astype(float)

    denom_mu_1p1p_w, dnom_bin_edges_mu_1p1p_w = np.histogram(new_mu[cuts_1p1p], bins=mu_bin, weights=weights[cuts_1p1p])
    num_mu_1p1p_w, num_bin_edges_mu_1p1p_w = np.histogram(new_mu[cuts_bdt_1p1p], bins=mu_bin, weights=weights[cuts_bdt_1p1p])
    denom_mu_1p1p_w = np.array(denom_mu_1p1p_w).astype(float)
    num_mu_1p1p_w = np.array(num_mu_1p1p_w).astype(float)
    efficiency_mu_1p1p_w = np.divide(num_mu_1p1p_w, denom_mu_1p1p_w, out=np.zeros_like(num_mu_1p1p_w), where=denom_mu_1p1p_w!=0).astype(float)

    denom_mu_3p3p_w, dnom_bin_edges_mu_3p3p_w = np.histogram(new_mu[cuts_3p3p], bins=mu_bin, weights=weights[cuts_3p3p])
    num_mu_3p3p_w, num_bin_edges_mu_3p3p_w = np.histogram(new_mu[cuts_bdt_3p3p], bins=mu_bin, weights=weights[cuts_bdt_3p3p])
    denom_mu_3p3p_w = np.array(denom_mu_3p3p_w).astype(float)
    num_mu_3p3p_w = np.array(num_mu_3p3p_w).astype(float)
    efficiency_mu_3p3p_w = np.divide(num_mu_3p3p_w, denom_mu_3p3p_w, out=np.zeros_like(num_mu_3p3p_w), where=denom_mu_3p3p_w!=0).astype(float)

    denom_mu_inc_w, dnom_bin_edges_mu_inc_w = np.histogram(new_mu[cuts_inc], bins=mu_bin, weights=weights[cuts_inc])
    num_mu_inc_w, num_bin_edges_mu_inc_w = np.histogram(new_mu[cuts_bdt_inc], bins=mu_bin, weights=weights[cuts_bdt_inc])
    denom_mu_inc_w = np.array(denom_mu_inc_w).astype(float)
    num_mu_inc_w = np.array(num_mu_inc_w).astype(float)
    efficiency_mu_inc_w = np.divide(num_mu_inc_w, denom_mu_inc_w, out=np.zeros_like(num_mu_inc_w), where=denom_mu_inc_w!=0).astype(float)


    fig6 = plt.figure()
    plt.plot(mu_bin[:-1], efficiency_mu, label="1p3p unweighted", color='black')
    plt.plot(mu_bin[:-1], efficiency_mu_1p1p, label="1p1p unweighted", color='orange')
    plt.plot(mu_bin[:-1], efficiency_mu_3p3p, label="3p3p unweighted", color='red')
    plt.plot(mu_bin[:-1], efficiency_mu_inc, label="inclusive unweighted", color='green')
    plt.plot(mu_bin[:-1], efficiency_mu_w, label="1p3p weights", linestyle='dashed', color='black')
    plt.plot(mu_bin[:-1], efficiency_mu_1p1p_w, label="1p1p weights", linestyle='dashed', color='orange')
    plt.plot(mu_bin[:-1], efficiency_mu_3p3p_w, label="3p3p weights", linestyle='dashed', color='red')
    plt.plot(mu_bin[:-1], efficiency_mu_inc_w, label="incl weights", linestyle='dashed', color='green')
    plt.legend()
    plt.xlabel("average mu")
    plt.ylabel("efficiency")
    p.savefig(fig6)
    plt.close(fig6)



  

    # sig_1p3p_w, edg_1p3p = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts], bins=100, weights=weights[cuts])
    # sig_1p1p_w, edg_1p1p = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts_1p1p], bins=100, weights=weights[cuts_1p1p])
    # sig_3p3p_w, edg_3p3p = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts_3p3p], bins=100, weights=weights[cuts_3p3p])
    # sig_inc_w, edg_inc = np.histogram(ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts_inc], bins=100, weights=weights[cuts_inc])
    # print(sig_1p3p_w)
    # print(np.ceil(np.array(sig_1p3p_w)).astype(int))
    # w_sig_1p3p = np.repeat((edg_1p3p[:-1] + edg_1p3p[1:]) / 2, np.ceil(np.array(sig_1p3p_w)).astype(int))
    # w_sig_1p1p = np.repeat((edg_1p1p[:-1] + edg_1p1p[1:]) / 2, np.ceil(np.array(sig_1p1p_w)).astype(int))
    # w_sig_3p3p = np.repeat((edg_3p3p[:-1] + edg_3p3p[1:]) / 2, np.ceil(np.array(sig_3p3p_w)).astype(int))
    # w_sig_inc = np.repeat((edg_inc[:-1] + edg_inc[1:]) / 2, np.ceil(np.array(sig_inc_w)).astype(int))

    # background_scores_w, bkg_edg_1p3p = np.histogram(bkg_bdt[bkg_cuts], bins=100, weights=bkg_weights[bkg_cuts])
    # background_scores_1p1p_w, bkg_edg_1p1p = np.histogram(bkg_bdt[bkg_cuts_1p1p], bins=100, weights=bkg_weights[bkg_cuts_1p1p])
    # background_scores_3p3p_w, bkg_edg_3p3p = np.histogram(bkg_bdt[bkg_cuts_3p3p], bins=100, weights=bkg_weights[bkg_cuts_3p3p])
    # background_scores_inc_w, bkg_edg_inc = np.histogram(bkg_bdt[bkg_cuts_inc], bins=100, weights=bkg_weights[bkg_cuts_inc])

    # w_bkg_1p3p = np.repeat((bkg_edg_1p3p[:-1] + bkg_edg_1p3p[1:]) / 2, np.ceil(np.array(background_scores_w)).astype(int))
    # w_bkg_1p1p = np.repeat((bkg_edg_1p1p[:-1] + bkg_edg_1p1p[1:]) / 2, np.ceil(np.array(background_scores_1p1p_w)).astype(int))
    # w_bkg_3p3p = np.repeat((bkg_edg_3p3p[:-1] + bkg_edg_3p3p[1:]) / 2, np.ceil(np.array(background_scores_3p3p_w)).astype(int))
    # w_bkg_inc = np.repeat((bkg_edg_inc[:-1] + bkg_edg_inc[1:]) / 2, np.ceil(np.array(background_scores_inc_w)).astype(int))
    
 



    # background_scores_w = []
    # background_scores_1p1p_w = []
    # background_scores_3p3p_w = []
    # background_scores_inc_w = []

    # bkg_1p3p_pt_eff = []
    # bkg_1p1p_pt_eff = []
    # bkg_3p3p_pt_eff = []
    # bkg_inc_pt_eff = []

    # bkg_1p3p_pt_eff_w = []
    # bkg_1p1p_pt_eff_w = []
    # bkg_3p3p_pt_eff_w = []
    # bkg_inc_pt_eff_w = []

    #append bdt score from jz1, jz2, jz3, jz4, jz5, jz6, jz7, jz8, jz9, jz10, jz11, jz12 into background_scores
    # for jz in [jz1, jz2, jz3, jz4, jz5, jz6, jz7, jz8, jz9, jz10, jz11, jz12]:
    #     bkg_cuts = ak.where((ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2) & 
    #                 ((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 1) | (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 3)) & 
    #                 ((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 1) | (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 3)))
    #     # bkg_cuts_bdt = ak.where((ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2) &
    #     #             ((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 1) | (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 3)) & 
    #     #             ((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 1) | (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) &
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.BDTScore']) < 0.72))
    #     bkg_cuts_1p1p = ak.where((ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2) &
    #                 (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 1) & 
    #                 (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 1))
    #     # bkg_cuts_1p1p_bdt = ak.where((ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2) &
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 1) & 
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 1) &
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.BDTScore']) < 0.72))
    #     bkg_cuts_3p3p = ak.where((ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2) &
    #                 (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 3) & 
    #                 (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 3))
    #     # bkg_cuts_3p3p_bdt = ak.where((ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2) &
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 3) & 
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 3) &
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.BDTScore']) < 0.72))
    #     bkg_cuts_inc = ak.where(ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2 &
    #                         (((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 1) | (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 3)) & 
    #                         ((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 1) | (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 3))) |
    #                         ((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 1) & (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 1)) |
    #                         ((ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_lead']) == 3) & (ak.flatten(jz['DiTauJetsAuxDyn.n_tracks_subl']) == 3)) )
    #     # bkg_cuts_inc_bdt = ak.where((ak.flatten(jz['DiTauJetsAuxDyn.n_subjets']) >=2) &
    #     #             (ak.flatten(jz['DiTauJetsAuxDyn.BDTScore']) < 0.72))
    #     bkg_weights = flattened_pt_weighted(jz['DiTauJetsAuxDyn.ditau_pt'], bins)
    #     bkg_1p3p_w, _ = np.histogram(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts], bins=100, weights=bkg_weights[bkg_cuts])
    #     bkg_1p1p_w, _ = np.histogram(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts_1p1p], bins=100, weights=bkg_weights[bkg_cuts_1p1p])
    #     bkg_3p3p_w, _ = np.histogram(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts_3p3p], bins=100, weights=bkg_weights[bkg_cuts_3p3p])
    #     bkg_inc_w, _ = np.histogram(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts_inc], bins=100, weights=bkg_weights[bkg_cuts_inc])

    #     background_scores.append(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts])
    #     background_scores_1p1p.append(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts_1p1p])
    #     background_scores_3p3p.append(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts_3p3p])
    #     background_scores_inc.append(ak.flatten(jz['DiTauJetsAuxDyn.BDTScore'])[bkg_cuts_inc])

    #     background_scores_w.append(bkg_1p3p_w)
    #     background_scores_1p1p_w.append(bkg_1p1p_w)
    #     background_scores_3p3p_w.append(bkg_3p3p_w)
    #     background_scores_inc_w.append(bkg_inc_w)

    #     # bkg_1p3p_pt_eff.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts, bkg_cuts_bdt))
    #     # bkg_1p1p_pt_eff.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts_1p1p, bkg_cuts_1p1p_bdt))
    #     # bkg_3p3p_pt_eff.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts_3p3p, bkg_cuts_3p3p_bdt))
    #     # bkg_inc_pt_eff.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts_inc, bkg_cuts_inc_bdt))
    #     # bkg_1p3p_pt_eff_w.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts, bkg_cuts_bdt, bkg_weights))
    #     # bkg_1p1p_pt_eff_w.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts_1p1p, bkg_cuts_1p1p_bdt, bkg_weights))
    #     # bkg_3p3p_pt_eff_w.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts_3p3p, bkg_cuts_3p3p_bdt, bkg_weights))
    #     # bkg_inc_pt_eff_w.append(calculate_efficiency(jz['DiTauJetsAuxDyn.ditau_pt'], bkg_pt_bins, bkg_cuts_inc, bkg_cuts_inc_bdt, bkg_weights))


    signal_scores = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts]
    signal_scores_1p1p = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts_1p1p]
    signal_scores_3p3p = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts_3p3p]
    signal_scores_inc = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[cuts_inc]

    signal_weight = (event_weight(f1)*getXS(425102)/event_weight_sum(f1))[cuts]
    signal_weight_1p1p = (event_weight(f1)*getXS(425102)/event_weight_sum(f1))[cuts_1p1p]
    signal_weight_3p3p = (event_weight(f1)*getXS(425102)/event_weight_sum(f1))[cuts_3p3p]
    signal_weight_inc = (event_weight(f1)*getXS(425102)/event_weight_sum(f1))[cuts_inc]

    background_scores = bkg_bdt[bkg_cuts]
    background_scores_1p1p = bkg_bdt[bkg_cuts_1p1p]
    background_scores_3p3p = bkg_bdt[bkg_cuts_3p3p]
    background_scores_inc = bkg_bdt[bkg_cuts_inc]
    
    background_weight = bkg_evt_weights[bkg_cuts]
    background_weight_1p1p = bkg_evt_weights[bkg_cuts_1p1p]
    background_weight_3p3p = bkg_evt_weights[bkg_cuts_3p3p]
    background_weight_inc = bkg_evt_weights[bkg_cuts_inc]

    print("sssss: ", signal_weight)
    print("bbbb: ", background_weight)

    print(len(signal_scores))
    print(len(signal_scores_1p1p))
    print(len(signal_scores_3p3p))
    print(len(signal_scores_inc))
    print("**********")
    print(len(background_scores))
    print(len(background_scores_1p1p))
    print(len(background_scores_3p3p))
    print(len(background_scores_inc))

    #plot signal_weight
    # fig_sig_weight = plt.figure()
    # plt.hist(signal_weight, bins=50, histtype="step", label="1p3p sig", color='black')
    # p.savefig(fig_sig_weight)
    # plt.close(fig_sig_weight)    

    ###### roc curve with scikit 
    fpr_1p3p, tpr_1p3p = calc_roc(signal_scores, background_scores)
    fpr_1p1p, tpr_1p1p = calc_roc(signal_scores_1p1p, background_scores_1p1p)
    fpr_3p3p, tpr_3p3p = calc_roc(signal_scores_3p3p, background_scores_3p3p)
    fpr_inc, tpr_inc = calc_roc(signal_scores_inc, background_scores_inc)

    fpr_1p3p_w, tpr_1p3p_w = calc_roc(signal_scores, background_scores, signal_weight*weights[cuts], background_weight*bkg_weights[bkg_cuts])
    fpr_1p1p_w, tpr_1p1p_w = calc_roc(signal_scores_1p1p, background_scores_1p1p, signal_weight_1p1p*weights[cuts_1p1p], background_weight_1p1p*bkg_weights[bkg_cuts_1p1p])
    fpr_3p3p_w, tpr_3p3p_w = calc_roc(signal_scores_3p3p, background_scores_3p3p, signal_weight_3p3p*weights[cuts_3p3p], background_weight_3p3p*bkg_weights[bkg_cuts_3p3p])
    fpr_inc_w, tpr_inc_w = calc_roc(signal_scores_inc, background_scores_inc, signal_weight_inc*weights[cuts_inc], background_weight_inc*bkg_weights[bkg_cuts_inc])
    print("FFFFFFF222222: ", len(fpr_1p3p_w))

    fig7 = plt.figure()
    plt.plot(tpr_1p3p, 1/fpr_1p3p, label="1p3p", color='black')
    plt.plot(tpr_1p1p, 1/fpr_1p1p, label="1p1p", color='orange')
    plt.plot(tpr_3p3p, 1/fpr_3p3p, label="3p3p", color='red')
    plt.plot(tpr_inc, 1/fpr_inc, label="inclusive", color='green')
    plt.plot(tpr_1p3p_w, 1/fpr_1p3p_w, label="1p3p weighted", linestyle='dashed', color='black')
    plt.plot(tpr_1p1p_w, 1/fpr_1p1p_w, label="1p1p weighted", linestyle='dashed', color='orange')
    plt.plot(tpr_3p3p_w, 1/fpr_3p3p_w, label="3p3p weighted", linestyle='dashed', color='red')
    plt.plot(tpr_inc_w, 1/fpr_inc_w, label="incl weighted", linestyle='dashed', color='green')
    plt.legend(loc='upper right')
    plt.xlabel("TPR")
    plt.ylabel("1/FPR")
    plt.yscale('log')
    p.savefig(fig7)
    plt.close(fig7)

    #########roc curve in pt bins
    # sig_pt02_1 = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] <= 1000000))
    sig_pt02_1 = cuts
    # sig_pt1_2 = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] >= 1000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] < 2000000))
    # sig_pt2_3 = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] >= 2000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] < 3000000))
    # bkg_pt02_1 = ak.where((data[bkg_cuts] >= 200000) & (data[bkg_cuts] <= 1000000))
    bkg_pt02_1 = bkg_cuts
    # bkg_pt1_2 = ak.where((data[bkg_cuts] >= 1000000) & (data[bkg_cuts] < 2000000))
    # bkg_pt2_3 = ak.where((data[bkg_cuts] >= 2000000) & (data[bkg_cuts] < 3000000))

    # sig_pt02_1_1p1p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_1p1p] >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_1p1p] < 1000000))
    # sig_pt1_2_1p1p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_1p1p] >= 1000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_1p1p] < 2000000))
    # sig_pt2_3_1p1p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_1p1p] >= 2000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_1p1p] < 3000000))
    # bkg_pt02_1_1p1p = ak.where((data[bkg_cuts_1p1p] >= 200000) & (data[bkg_cuts_1p1p] < 1000000))
    # bkg_pt1_2_1p1p = ak.where((data[bkg_cuts_1p1p] >= 1000000) & (data[bkg_cuts_1p1p] < 2000000))
    # bkg_pt2_3_1p1p = ak.where((data[bkg_cuts_1p1p] >= 2000000) & (data[bkg_cuts_1p1p] < 3000000))

    # sig_pt02_1_3p3p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_3p3p] >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_3p3p] < 1000000))
    # sig_pt1_2_3p3p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_3p3p] >= 1000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_3p3p] < 2000000))
    # sig_pt2_3_3p3p = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_3p3p] >= 2000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_3p3p] < 3000000))
    # bkg_pt02_1_3p3p = ak.where((data[bkg_cuts_3p3p] >= 200000) & (data[bkg_cuts_3p3p] < 1000000))
    # bkg_pt1_2_3p3p = ak.where((data[bkg_cuts_3p3p] >= 1000000) & (data[bkg_cuts_3p3p] < 2000000))
    # bkg_pt2_3_3p3p = ak.where((data[bkg_cuts_3p3p] >= 2000000) & (data[bkg_cuts_3p3p] < 3000000))

    # sig_pt02_1_inc = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_inc] >= 200000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_inc] < 1000000))
    # sig_pt1_2_inc = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_inc] >= 1000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_inc] < 2000000))
    # sig_pt2_3_inc = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_inc] >= 2000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts_inc] < 3000000))
    # bkg_pt02_1_inc = ak.where((data[bkg_cuts_inc] >= 200000) & (data[bkg_cuts_inc] < 1000000))
    # bkg_pt1_2_inc = ak.where((data[bkg_cuts_inc] >= 1000000) & (data[bkg_cuts_inc] < 2000000))
    # bkg_pt2_3_inc = ak.where((data[bkg_cuts_inc] >= 2000000) & (data[bkg_cuts_inc] < 3000000))

    signal_scores_pt02_1 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt02_1]
    print("TTTTTTTTTTTT: ", len(signal_scores_pt02_1))
    # signal_scores_pt1_2 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt1_2]
    # signal_scores_pt2_3 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt2_3]
    # signal_scores_1p1p_pt02_1 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt02_1_1p1p]
    # signal_scores_1p1p_pt1_2 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt1_2_1p1p]
    # signal_scores_1p1p_pt2_3 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt2_3_1p1p]
    # signal_scores_3p3p_pt02_1 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt02_1_3p3p]
    # signal_scores_3p3p_pt1_2 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt1_2_3p3p]
    # signal_scores_3p3p_pt2_3 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt2_3_3p3p]
    # signal_scores_inc_pt02_1 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt02_1_inc]
    # signal_scores_inc_pt1_2 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt1_2_inc]
    # signal_scores_inc_pt2_3 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt2_3_inc]

    signal_weight_pt02_1 = (event_weight(f1)*getXS(425102)/event_weight_sum(f1))[sig_pt02_1]
    # signal_weight_pt1_2 = (event_weight(f1))[sig_pt1_2]
    # signal_weight_pt2_3 = (event_weight(f1))[sig_pt2_3]
    # signal_weight_1p1p_pt02_1 = (event_weight(f1))[sig_pt02_1_1p1p]
    # signal_weight_1p1p_pt1_2 = (event_weight(f1))[sig_pt1_2_1p1p]
    # signal_weight_1p1p_pt2_3 = (event_weight(f1))[sig_pt2_3_1p1p]
    # signal_weight_3p3p_pt02_1 = (event_weight(f1))[sig_pt02_1_3p3p]
    # signal_weight_3p3p_pt1_2 = (event_weight(f1))[sig_pt1_2_3p3p]
    # signal_weight_3p3p_pt2_3 = (event_weight(f1))[sig_pt2_3_3p3p]
    # signal_weight_inc_pt02_1 = (event_weight(f1))[sig_pt02_1_inc]
    # signal_weight_inc_pt1_2 = (event_weight(f1))[sig_pt1_2_inc]
    # signal_weight_inc_pt2_3 = (event_weight(f1))[sig_pt2_3_inc]

    background_scores_pt02_1 = bkg_bdt[bkg_pt02_1]
    print("BBBBBBBBBBBB: ", len(background_scores_pt02_1))
    # background_scores_pt1_2 = bkg_bdt[bkg_pt1_2]
    # background_scores_pt2_3 = bkg_bdt[bkg_pt2_3]
    # background_scores_1p1p_pt02_1 = bkg_bdt[bkg_pt02_1_1p1p]
    # background_scores_1p1p_pt1_2 = bkg_bdt[bkg_pt1_2_1p1p]
    # background_scores_1p1p_pt2_3 = bkg_bdt[bkg_pt2_3_1p1p]
    # background_scores_3p3p_pt02_1 = bkg_bdt[bkg_pt02_1_3p3p]
    # background_scores_3p3p_pt1_2 = bkg_bdt[bkg_pt1_2_3p3p]
    # background_scores_3p3p_pt2_3 = bkg_bdt[bkg_pt2_3_3p3p]
    # background_scores_inc_pt02_1 = bkg_bdt[bkg_pt02_1_inc]
    # background_scores_inc_pt1_2 = bkg_bdt[bkg_pt1_2_inc]
    # background_scores_inc_pt2_3 = bkg_bdt[bkg_pt2_3_inc]

    background_weight_pt02_1 = bkg_evt_weights[bkg_pt02_1]
    # background_weight_pt1_2 = bkg_evt_weights[bkg_pt1_2]
    # background_weight_pt2_3 = bkg_evt_weights[bkg_pt2_3]
    # background_weight_1p1p_pt02_1 = bkg_evt_weights[bkg_pt02_1_1p1p]
    # background_weight_1p1p_pt1_2 = bkg_evt_weights[bkg_pt1_2_1p1p]
    # background_weight_1p1p_pt2_3 = bkg_evt_weights[bkg_pt2_3_1p1p]
    # background_weight_3p3p_pt02_1 = bkg_evt_weights[bkg_pt02_1_3p3p]
    # background_weight_3p3p_pt1_2 = bkg_evt_weights[bkg_pt1_2_3p3p]
    # background_weight_3p3p_pt2_3 = bkg_evt_weights[bkg_pt2_3_3p3p]
    # background_weight_inc_pt02_1 = bkg_evt_weights[bkg_pt02_1_inc]
    # background_weight_inc_pt1_2 = bkg_evt_weights[bkg_pt1_2_inc]
    # background_weight_inc_pt2_3 = bkg_evt_weights[bkg_pt2_3_inc]

    fpr_1p3p_w_pt02_1, tpr_1p3p_w_pt02_1 = calc_roc(signal_scores_pt02_1, background_scores_pt02_1, signal_weight_pt02_1*weights[sig_pt02_1], background_weight_pt02_1*bkg_weights[bkg_pt02_1])
    # fpr_1p1p_w_pt02_1, tpr_1p1p_w_pt02_1 = calc_roc(signal_scores_1p1p_pt02_1, background_scores_1p1p_pt02_1, (signal_weight_1p1p_pt02_1*weights[sig_pt02_1_1p1p]), (background_weight_1p1p_pt02_1*bkg_weights[bkg_pt02_1_1p1p]))
    # fpr_3p3p_w_pt02_1, tpr_3p3p_w_pt02_1 = calc_roc(signal_scores_3p3p_pt02_1, background_scores_3p3p_pt02_1, (signal_weight_3p3p_pt02_1*weights[sig_pt02_1_3p3p]), (background_weight_3p3p_pt02_1*bkg_weights[bkg_pt02_1_3p3p]))
    # fpr_inc_w_pt02_1, tpr_inc_w_pt02_1 = calc_roc(signal_scores_inc_pt02_1, background_scores_inc_pt02_1, (signal_weight_inc_pt02_1*weights[sig_pt02_1_inc]), (background_weight_inc_pt02_1*bkg_weights[bkg_pt02_1_inc]))

    # fpr_1p3p_w_pt1_2, tpr_1p3p_w_pt1_2 = calc_roc(signal_scores_pt1_2, background_scores_pt1_2, (signal_weight_pt1_2*weights[sig_pt1_2]), (background_weight_pt1_2*bkg_weights[bkg_pt1_2]))
    # fpr_1p1p_w_pt1_2, tpr_1p1p_w_pt1_2 = calc_roc(signal_scores_1p1p_pt1_2, background_scores_1p1p_pt1_2, (signal_weight_1p1p_pt1_2*weights[sig_pt1_2_1p1p]), (background_weight_1p1p_pt1_2*bkg_weights[bkg_pt1_2_1p1p]))
    # fpr_3p3p_w_pt1_2, tpr_3p3p_w_pt1_2 = calc_roc(signal_scores_3p3p_pt1_2, background_scores_3p3p_pt1_2, (signal_weight_3p3p_pt1_2*weights[sig_pt1_2_3p3p]), (background_weight_3p3p_pt1_2*bkg_weights[bkg_pt1_2_3p3p]))
    # fpr_inc_w_pt1_2, tpr_inc_w_pt1_2 = calc_roc(signal_scores_inc_pt1_2, background_scores_inc_pt1_2, (signal_weight_inc_pt1_2*weights[sig_pt1_2_inc]), (background_weight_inc_pt1_2*bkg_weights[bkg_pt1_2_inc]))

    # fpr_1p3p_w_pt2_3, tpr_1p3p_w_pt2_3 = calc_roc(signal_scores_pt2_3, background_scores_pt2_3, (signal_weight_pt2_3*weights[sig_pt2_3]), (background_weight_pt2_3*bkg_weights[bkg_pt2_3]))
    # fpr_1p1p_w_pt2_3, tpr_1p1p_w_pt2_3 = calc_roc(signal_scores_1p1p_pt2_3, background_scores_1p1p_pt2_3, (signal_weight_1p1p_pt2_3*weights[sig_pt2_3_1p1p]), (background_weight_1p1p_pt2_3*bkg_weights[bkg_pt2_3_1p1p]))
    # fpr_3p3p_w_pt2_3, tpr_3p3p_w_pt2_3 = calc_roc(signal_scores_3p3p_pt2_3, background_scores_3p3p_pt2_3, (signal_weight_3p3p_pt2_3*weights[sig_pt2_3_3p3p]), (background_weight_3p3p_pt2_3*bkg_weights[bkg_pt2_3_3p3p]))
    # fpr_inc_w_pt2_3, tpr_inc_w_pt2_3 = calc_roc(signal_scores_inc_pt2_3, background_scores_inc_pt2_3, (signal_weight_inc_pt2_3*weights[sig_pt2_3_inc]), (background_weight_inc_pt2_3*bkg_weights[bkg_pt2_3_inc]))

    print("FFFFFFF: ", len(fpr_1p3p_w_pt02_1))
    # print("111111: ", len(signal_scores_pt1_2))
    # print("222222: ", len(signal_scores_pt2_3))

    fig10 = plt.figure()
    plt.plot(tpr_1p3p_w_pt02_1, 1/fpr_1p3p_w_pt02_1, label="1p3p", color='black')
    # plt.plot(tpr_1p1p_w_pt02_1, 1/fpr_1p1p_w_pt02_1, label="1p1p", color='orange')
    # plt.plot(tpr_3p3p_w_pt02_1, 1/fpr_3p3p_w_pt02_1, label="3p3p", color='red')
    # plt.plot(tpr_inc_w_pt02_1, 1/fpr_inc_w_pt02_1, label="inclusive", color='green')
    plt.legend(loc='upper right')
    plt.xlabel("TPR")
    plt.ylabel("1/FPR")
    plt.yscale('log')
    p.savefig(fig10)
    plt.close(fig10)

    # fig11 = plt.figure()
    # plt.plot(tpr_1p3p_w_pt1_2, 1/fpr_1p3p_w_pt1_2, label="1p3p", color='black')
    # plt.plot(tpr_1p1p_w_pt1_2, 1/fpr_1p1p_w_pt1_2, label="1p1p", color='orange')
    # plt.plot(tpr_3p3p_w_pt1_2, 1/fpr_3p3p_w_pt1_2, label="3p3p", color='red')
    # plt.plot(tpr_inc_w_pt1_2, 1/fpr_inc_w_pt1_2, label="inclusive", color='green')
    # plt.legend(loc='upper right')
    # plt.xlabel("TPR")
    # plt.ylabel("1/FPR")
    # plt.yscale('log')
    # p.savefig(fig11)

    # fig12 = plt.figure()
    # plt.plot(tpr_1p3p_w_pt2_3, 1/fpr_1p3p_w_pt2_3, label="1p3p", color='black')
    # plt.plot(tpr_1p1p_w_pt2_3, 1/fpr_1p1p_w_pt2_3, label="1p1p", color='orange')
    # plt.plot(tpr_3p3p_w_pt2_3, 1/fpr_3p3p_w_pt2_3, label="3p3p", color='red')
    # plt.plot(tpr_inc_w_pt2_3, 1/fpr_inc_w_pt2_3, label="inclusive", color='green')
    # plt.legend(loc='upper right')
    # plt.xlabel("TPR")
    # plt.ylabel("1/FPR")
    # plt.yscale('log')
    # p.savefig(fig12)
    # plt.close(fig12)

    ####### plot the score distribution
    fig_score = plt.figure()
    plt.hist(signal_scores, bins=60, histtype="step", label="1p3p sig", color='black')
    plt.hist(background_scores, bins=60, histtype="step", label="1p3p bkg", linestyle='dashed', color='black')
    plt.hist(signal_scores_1p1p, bins=60, histtype="step", label="1p1p sig", color='orange')
    plt.hist(background_scores_1p1p, bins=60, histtype="step", label="1p1p bkg", linestyle='dashed', color='orange')
    plt.hist(signal_scores_3p3p, bins=60, histtype="step", label="3p3p sig", color='red')
    plt.hist(background_scores_3p3p, bins=60, histtype="step", label="3p3p bkg", linestyle='dashed', color='red')
    plt.hist(signal_scores_inc, bins=60, histtype="step", label="inc sig", color='green')
    plt.hist(background_scores_inc, bins=60, histtype="step", label="inc bkg", linestyle='dashed', color='green')
    plt.xlabel("BDT score")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_score)
    plt.close(fig_score)

    fig_score_w = plt.figure()
    plt.hist(signal_scores, bins=60, histtype="step", label="1p3p sig", color='black', weights=signal_weight*weights[cuts])
    plt.hist(background_scores, bins=60, histtype="step", label="1p3p bkg", linestyle='dashed', color='black', weights=background_weight*bkg_weights[bkg_cuts])
    plt.hist(signal_scores_1p1p, bins=60, histtype="step", label="1p1p sig", color='orange', weights=signal_weight_1p1p*weights[cuts_1p1p])
    plt.hist(background_scores_1p1p, bins=60, histtype="step", label="1p1p bkg", linestyle='dashed', color='orange', weights=background_weight_1p1p*bkg_weights[bkg_cuts_1p1p])
    plt.hist(signal_scores_3p3p, bins=60, histtype="step", label="3p3p sig", color='red', weights=signal_weight_3p3p*weights[cuts_3p3p])
    plt.hist(background_scores_3p3p, bins=60, histtype="step", label="3p3p bkg", linestyle='dashed', color='red', weights=background_weight_3p3p*bkg_weights[bkg_cuts_3p3p])
    plt.hist(signal_scores_inc, bins=60, histtype="step", label="inc sig", color='green', weights=signal_weight_inc*weights[cuts_inc])
    plt.hist(background_scores_inc, bins=60, histtype="step", label="inc bkg", linestyle='dashed', color='green', weights=background_weight_inc*bkg_weights[bkg_cuts_inc])
    plt.xlabel("BDT score weighted")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_score_w)
    plt.close(fig_score_w)

    ####### roc curve by hand 
    s1p3p_hist, _ = np.histogram(signal_scores, bins=60)
    b1p3p_hist, _ = np.histogram(background_scores, bins=60)
    s1p1p_hist, _ = np.histogram(signal_scores_1p1p, bins=60)
    b1p1p_hist, _ = np.histogram(background_scores_1p1p, bins=60)
    s3p3p_hist, _ = np.histogram(signal_scores_3p3p, bins=60)
    b3p3p_hist, _ = np.histogram(background_scores_3p3p, bins=60)
    sinc_hist, _  = np.histogram(signal_scores_inc, bins=60)
    binc_hist, _  = np.histogram(background_scores_inc, bins=60)
    calc_tpr, calc_fpr = calc_roc_curve(s1p3p_hist, b1p3p_hist)
    calc_tpr_1p1p, calc_fpr_1p1p = calc_roc_curve(s1p1p_hist, b1p1p_hist)
    calc_tpr_3p3p, calc_fpr_3p3p = calc_roc_curve(s3p3p_hist, b3p3p_hist)
    calc_tpr_inc, calc_fpr_inc = calc_roc_curve(sinc_hist, binc_hist)
    #with weights
    s1p3p_hist_w, _ = np.histogram(signal_scores, bins=60, weights=signal_weight*weights[cuts])
    b1p3p_hist_w, _ = np.histogram(background_scores, bins=60, weights=background_weight*bkg_weights[bkg_cuts])
    s1p1p_hist_w, _ = np.histogram(signal_scores_1p1p, bins=60, weights=signal_weight_1p1p*weights[cuts_1p1p])
    b1p1p_hist_w, _ = np.histogram(background_scores_1p1p, bins=60, weights=background_weight_1p1p*bkg_weights[bkg_cuts_1p1p])
    s3p3p_hist_w, _ = np.histogram(signal_scores_3p3p, bins=60, weights=signal_weight_3p3p*weights[cuts_3p3p])
    b3p3p_hist_w, _ = np.histogram(background_scores_3p3p, bins=60, weights=background_weight_3p3p*bkg_weights[bkg_cuts_3p3p])
    sinc_hist_w, _  = np.histogram(signal_scores_inc, bins=60, weights=signal_weight_inc*weights[cuts_inc])
    binc_hist_w, _  = np.histogram(background_scores_inc, bins=60, weights=background_weight_inc*bkg_weights[bkg_cuts_inc])
    calc_tpr_w, calc_fpr_w = calc_roc_curve(s1p3p_hist_w, b1p3p_hist_w)
    calc_tpr_1p1p_w, calc_fpr_1p1p_w = calc_roc_curve(s1p1p_hist_w, b1p1p_hist_w)
    calc_tpr_3p3p_w, calc_fpr_3p3p_w = calc_roc_curve(s3p3p_hist_w, b3p3p_hist_w)
    calc_tpr_inc_w, calc_fpr_inc_w = calc_roc_curve(sinc_hist_w, binc_hist_w)
    #plot
    fig8 = plt.figure()
    plt.plot(calc_tpr, calc_fpr, label="1p3p", color='black')
    plt.plot(calc_tpr_1p1p, calc_fpr_1p1p, label="1p1p", color='orange')
    plt.plot(calc_tpr_3p3p, calc_fpr_3p3p, label="3p3p", color='red')
    plt.plot(calc_tpr_inc, calc_fpr_inc, label="inclusive", color='green')
    plt.plot(calc_tpr_w, calc_fpr_w, label="1p3p weighted", linestyle='dashed', color='black')
    plt.plot(calc_tpr_1p1p_w, calc_fpr_1p1p_w, label="1p1p weighted", linestyle='dashed', color='orange')
    plt.plot(calc_tpr_3p3p_w, calc_fpr_3p3p_w, label="3p3p weighted", linestyle='dashed', color='red')
    plt.plot(calc_tpr_inc_w, calc_fpr_inc_w, label="incl weighted", linestyle='dashed', color='green')
    plt.xlabel("TPR")
    plt.ylabel("1/FPR")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig8)
    plt.close(fig8)




    p.close() #end of plt plots


    # sig_1p3p_denom, sig_1p3p_denom_edge, sig_1p3p_num, sig_1p3p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts, cuts_bdt)
    # sig_1p1p_denom, sig_1p1p_denom_edge, sig_1p1p_num, sig_1p1p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_1p1p, cuts_bdt_1p1p)
    # sig_3p3p_denom, sig_3p3p_denom_edge, sig_3p3p_num, sig_3p3p_num_edge = calculate_efficiency_hists(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_3p3p, cuts_bdt_3p3p)
    # sig_inc_denom, sig_inc_denom_edge, sig_inc_num, sig_inc_num_edge = calculate_efficiency_hists(f1['DiTauJetsAuxDyn.ditau_pt'], bins, cuts_inc, cuts_bdt_inc)

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

    bkg_cuts_list = [bkg_cuts, bkg_cuts_bdt, bkg_cuts_1p1p, bkg_cuts_1p1p_bdt, bkg_cuts_3p3p, bkg_cuts_3p3p_bdt, bkg_cuts_inc, bkg_cuts_inc_bdt]
    signal_cuts_list = [cuts, cuts_bdt, cuts_1p1p, cuts_bdt_1p1p, cuts_3p3p, cuts_bdt_3p3p, cuts_inc, cuts_bdt_inc]

    pt_1p3p_eff, pt_1p1p_eff, pt_3p3p_eff, pt_inc_eff = plot_eff(data, bkg_cuts_list, "DiJet pT", 20, 200000, 1000000, eta=False)
    pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(data, bkg_cuts_list, "DiJet pT", 20, 200000, 1000000, bkg_evt_weights*bkg_weights, eta=False)
    pt_1p3p_eff.SetMarkerStyle(41)
    pt_1p1p_eff.SetMarkerStyle(41)
    pt_3p3p_eff.SetMarkerStyle(41)
    pt_inc_eff.SetMarkerStyle(41)
    pt_1p3p_eff.Draw(" e")
    pt_1p1p_eff.Draw("same e")
    pt_3p3p_eff.Draw("same e")
    pt_inc_eff.Draw("same e")
    pt_1p3p_eff_w.SetMarkerStyle(22)
    pt_1p1p_eff_w.SetMarkerStyle(22)
    pt_3p3p_eff_w.SetMarkerStyle(22)
    pt_inc_eff_w.SetMarkerStyle(22)
    pt_1p3p_eff_w.Draw("same e")
    pt_1p1p_eff_w.Draw("same e")
    pt_3p3p_eff_w.Draw("same e")
    pt_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(pt_1p3p_eff, "1p3p")
    legend.AddEntry(pt_1p1p_eff, "1p1p")
    legend.AddEntry(pt_3p3p_eff, "3p3p")
    legend.AddEntry(pt_inc_eff, "inclusive")
    legend.AddEntry(pt_1p3p_eff_w, "1p3p w")
    legend.AddEntry(pt_1p1p_eff_w, "1p1p w")
    legend.AddEntry(pt_3p3p_eff_w, "3p3p w")
    legend.AddEntry(pt_inc_eff_w, "inclusive w")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_1p3p_eff, eta_1p1p_eff, eta_3p3p_eff, eta_inc_eff = plot_eff(data_eta, bkg_cuts_list, "DiJet eta", 40, -2.5, 2.5, eta=True)
    eta_1p3p_eff_w, eta_1p1p_eff_w, eta_3p3p_eff_w, eta_inc_eff_w = plot_eff(data_eta, bkg_cuts_list, "DiJet eta", 40, -2.5, 2.5, bkg_evt_weights*bkg_weights, eta=True)
    eta_1p3p_eff.SetMarkerStyle(41)
    eta_1p1p_eff.SetMarkerStyle(41)
    eta_3p3p_eff.SetMarkerStyle(41)
    eta_inc_eff.SetMarkerStyle(41)
    eta_1p3p_eff.Draw(" e")
    eta_1p1p_eff.Draw("same e")
    eta_3p3p_eff.Draw("same e")
    eta_inc_eff.Draw("same e")
    eta_1p3p_eff_w.SetMarkerStyle(22)
    eta_1p1p_eff_w.SetMarkerStyle(22)
    eta_3p3p_eff_w.SetMarkerStyle(22)
    eta_inc_eff_w.SetMarkerStyle(22)
    eta_1p3p_eff_w.Draw("same e")
    eta_1p1p_eff_w.Draw("same e")
    eta_3p3p_eff_w.Draw("same e")
    eta_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(eta_1p3p_eff, "1p3p")
    legend.AddEntry(eta_1p1p_eff, "1p1p")
    legend.AddEntry(eta_3p3p_eff, "3p3p")
    legend.AddEntry(eta_inc_eff, "inclusive")
    legend.AddEntry(eta_1p3p_eff_w, "1p3p w")
    legend.AddEntry(eta_1p1p_eff_w, "1p1p w")
    legend.AddEntry(eta_3p3p_eff_w, "3p3p w")
    legend.AddEntry(eta_inc_eff_w, "inclusive w")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()
 
    pt_sig_1p3p_eff, pt_sig_1p1p_eff, pt_sig_3p3p_eff, pt_sig_inc_eff = plot_eff(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']), signal_cuts_list, "Graviton pT", 20, 200000, 1000000, eta=False)
    pt_sig_1p3p_eff_w, pt_sig_1p1p_eff_w, pt_sig_3p3p_eff_w, pt_sig_inc_eff_w = plot_eff(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']), signal_cuts_list, "Graviton pT", 20, 200000, 1000000, event_weight(f1)*weights, eta=False)
    pt_sig_1p3p_eff.SetMarkerStyle(41)
    pt_sig_1p1p_eff.SetMarkerStyle(41)
    pt_sig_3p3p_eff.SetMarkerStyle(41)
    pt_sig_inc_eff.SetMarkerStyle(41)
    pt_sig_1p3p_eff.Draw(" e")
    pt_sig_1p1p_eff.Draw("same e")
    pt_sig_3p3p_eff.Draw("same e")
    pt_sig_inc_eff.Draw("same e")
    pt_sig_1p3p_eff_w.SetMarkerStyle(22)
    pt_sig_1p1p_eff_w.SetMarkerStyle(22)
    pt_sig_3p3p_eff_w.SetMarkerStyle(22)
    pt_sig_inc_eff_w.SetMarkerStyle(22)
    pt_sig_1p3p_eff_w.Draw("same e")
    pt_sig_1p1p_eff_w.Draw("same e")
    pt_sig_3p3p_eff_w.Draw("same e")
    pt_sig_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(pt_sig_1p3p_eff, "1p3p")
    legend.AddEntry(pt_sig_1p1p_eff, "1p1p")
    legend.AddEntry(pt_sig_3p3p_eff, "3p3p")
    legend.AddEntry(pt_sig_inc_eff, "inclusive")
    legend.AddEntry(pt_sig_1p3p_eff_w, "1p3p w")
    legend.AddEntry(pt_sig_1p1p_eff_w, "1p1p w")
    legend.AddEntry(pt_sig_3p3p_eff_w, "3p3p w")
    legend.AddEntry(pt_sig_inc_eff_w, "inclusive w")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_sig_1p3p_eff, eta_sig_1p1p_eff, eta_sig_3p3p_eff, eta_sig_inc_eff = plot_eff(ak.flatten(f1['DiTauJetsAux.eta']), signal_cuts_list, "Graviton eta", 40, -2.5, 2.5, eta=True)
    eta_sig_1p3p_eff_w, eta_sig_1p1p_eff_w, eta_sig_3p3p_eff_w, eta_sig_inc_eff_w = plot_eff(ak.flatten(f1['DiTauJetsAux.eta']), signal_cuts_list, "Graviton eta", 40, -2.5, 2.5, event_weight(f1)*weights, eta=True)
    eta_sig_1p3p_eff.SetMarkerStyle(41)
    eta_sig_1p1p_eff.SetMarkerStyle(41)
    eta_sig_3p3p_eff.SetMarkerStyle(41)
    eta_sig_inc_eff.SetMarkerStyle(41)
    eta_sig_1p3p_eff.Draw(" e")
    eta_sig_1p1p_eff.Draw("same e")
    eta_sig_3p3p_eff.Draw("same e")
    eta_sig_inc_eff.Draw("same e")
    eta_sig_1p3p_eff_w.SetMarkerStyle(22)
    eta_sig_1p1p_eff_w.SetMarkerStyle(22)
    eta_sig_3p3p_eff_w.SetMarkerStyle(22)
    eta_sig_inc_eff_w.SetMarkerStyle(22)
    eta_sig_1p3p_eff_w.Draw("same e")
    eta_sig_1p1p_eff_w.Draw("same e")
    eta_sig_3p3p_eff_w.Draw("same e")
    eta_sig_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(eta_sig_1p3p_eff, "1p3p")
    legend.AddEntry(eta_sig_1p1p_eff, "1p1p")
    legend.AddEntry(eta_sig_3p3p_eff, "3p3p")
    legend.AddEntry(eta_sig_inc_eff, "inclusive")
    legend.AddEntry(eta_sig_1p3p_eff_w, "1p3p w")
    legend.AddEntry(eta_sig_1p1p_eff_w, "1p1p w")
    legend.AddEntry(eta_sig_3p3p_eff_w, "3p3p w")
    legend.AddEntry(eta_sig_inc_eff_w, "inclusive w")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    mu_sig_1p3p_eff, mu_sig_1p1p_eff, mu_sig_3p3p_eff, mu_sig_inc_eff = plot_eff(new_mu, signal_cuts_list, "Graviton mu", 20, 18, 74, eta=False)
    mu_sig_1p3p_eff_w, mu_sig_1p1p_eff_w, mu_sig_3p3p_eff_w, mu_sig_inc_eff_w = plot_eff(new_mu, signal_cuts_list, "Graviton mu", 20, 18, 74, event_weight(f1)*weights, eta=False)
    mu_sig_1p3p_eff.SetMarkerStyle(41)
    mu_sig_1p1p_eff.SetMarkerStyle(41)
    mu_sig_3p3p_eff.SetMarkerStyle(41)
    mu_sig_inc_eff.SetMarkerStyle(41)
    mu_sig_1p3p_eff.Draw(" e")
    mu_sig_1p1p_eff.Draw("same e")
    mu_sig_3p3p_eff.Draw("same e")
    mu_sig_inc_eff.Draw("same e")
    mu_sig_1p3p_eff_w.SetMarkerStyle(22)
    mu_sig_1p1p_eff_w.SetMarkerStyle(22)
    mu_sig_3p3p_eff_w.SetMarkerStyle(22)
    mu_sig_inc_eff_w.SetMarkerStyle(22)
    mu_sig_1p3p_eff_w.Draw("same e")
    mu_sig_1p1p_eff_w.Draw("same e")
    mu_sig_3p3p_eff_w.Draw("same e")
    mu_sig_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(mu_sig_1p3p_eff, "1p3p")
    legend.AddEntry(mu_sig_1p1p_eff, "1p1p")
    legend.AddEntry(mu_sig_3p3p_eff, "3p3p")
    legend.AddEntry(mu_sig_inc_eff, "inclusive")
    legend.AddEntry(mu_sig_1p3p_eff_w, "1p3p w")
    legend.AddEntry(mu_sig_1p1p_eff_w, "1p1p w")
    legend.AddEntry(mu_sig_3p3p_eff_w, "3p3p w")
    legend.AddEntry(mu_sig_inc_eff_w, "inclusive w")
    legend.Draw("same")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    canvas.Print("eff_plots.pdf]")





if __name__ == "__main__":
    plot_branches()
