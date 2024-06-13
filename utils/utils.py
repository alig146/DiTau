import glob, os, sys, json, time, tempfile
import uproot, ROOT
import random
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import pandas as pd
from array import array
import logging 
import ctypes
import h5py

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
    # h.GetYaxis().SetTitle('Efficiency')
    return h

def make_eff_graph(h_pass, h_total, name=None):
    g = ROOT.TGraphAsymmErrors()
    g.Divide(h_pass,h_total, 'cl=0.683 b(1,1) mode')
    g.SetName(name or 'g_eff')
    g.GetXaxis().SetTitle(h_pass.GetXaxis().GetTitle())
    g.GetYaxis().SetTitle('Efficiency')
    g.GetXaxis().SetRangeUser(h_total.GetXaxis().GetXmin(), h_total.GetXaxis().GetXmax())
    return g

def full_integral_and_error(hist):
    """Returns histogram integral and error including underflow/overflow

    :param hist: 1D, 2D or 3D histogram 
    :type hist: :class:`ROOT.TH1`
    :rtype: (float,float)
    """
    err = ctypes.c_double(0.)
    if isinstance(hist,ROOT.TH3):
        return [hist.IntegralAndError(0, hist.GetNbinsX()+1, 
                                      0, hist.GetNbinsY()+1,
                                      0,hist.GetNbinsZ()+1, 
                                      err), err]
    elif isinstance(hist,ROOT.TH2):
        return [hist.IntegralAndError(0, hist.GetNbinsX() + 1, 
                                      0, hist.GetNbinsY()+1,
                                      err), err]
    elif isinstance(hist,ROOT.TH1):
        return [hist.IntegralAndError(0, hist.GetNbinsX() + 1, err), err]
    log().warn('Cannot integrate non TH1/2/3 object!')
    return [0.0, 0.0]

def full_integral(hist):
    """Wrapper for :func:`full_integral_and_error` returning only integral"""
    return full_integral_and_error(hist)[0]

def normalize_hist(hist):
    """Normalize a histogram to unit area

    :param hist: histogram
    :type hist: :class:`ROOT.TH1`
    """
    n = full_integral(hist)
    if hist and n: hist.Scale(1. / n)

def create_roc_graph(h_sig,h_bkg,effmin=None,name="g_roc",normalize=True,reverse=False):
    normalize_hist(h_sig) 
    normalize_hist(h_bkg)

    # get total normalisation
    nsig_tot = full_integral(h_sig)
    nbkg_tot = full_integral(h_bkg)
    if (nsig_tot <=0) or (nbkg_tot <= 0):  
        return None

    # loop over hist, calculating efficiency/rejection
    xarr = array('d',[])
    yarr = array('d',[])
    nbins = h_sig.GetNbinsX()
    for ibin in range(0,nbins+2): 
        if reverse: 
            nsig = h_sig.Integral(0, ibin)
            nbkg = h_bkg.Integral(0, ibin)
        else:     
            nsig = h_sig.Integral(ibin,nbins+2)
            nbkg = h_bkg.Integral(ibin,nbins+2)

        esig = nsig 
        if normalize: esig /= nsig_tot
        rbkg = nbkg_tot / nbkg if nbkg else 1.e7
        
        if effmin is not None and esig < effmin: continue
        
        xarr.append(esig)
        yarr.append(rbkg)

    g = ROOT.TGraph(len(xarr),xarr,yarr)
    g.SetName(name)
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
        tpr_in_loop = signal_cdf
        fpr_in_loop = np.sum(background_hist)/background_cdf
        if tpr_in_loop < 0.05: continue
        tpr.append(tpr_in_loop)
        fpr.append(fpr_in_loop)

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    return tpr, fpr

def getXS(dsid):
    # xs_file = "/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc16.txt"
    xs_file = "/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc23.txt"
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

def Evnt_W_Sum_Cutflow(root_files):
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

def event_weight_dijet(tree):
    non_empty_pt_arrays = ak.num(tree['ditau_pt']) > 0
    filtered_ditau_pt = tree['ditau_pt'][non_empty_pt_arrays]
    num_dijet = ak.num(filtered_ditau_pt, axis=1)
    # we use the first elment of the weights array (total size 27)
    first_element = tree['event_weight'][non_empty_pt_arrays][:, 0]
    repeated_first_element = np.repeat(first_element, num_dijet)
    return repeated_first_element

def event_weight_sum_dijet(tree):
    non_empty_pt_arrays = ak.num(tree['ditau_pt']) > 0
    first_element = tree['event_weight'][:, 0]
    return np.sum(first_element)

def read_tree(file_paths, branches):
    expanded_file_paths = []
    for path in file_paths:
        expanded_file_paths.extend(glob.glob(path))
    # print(expanded_file_paths)
    arrays = {}
    with Pool() as pool:
        #read files in parallel using read_file function which has two arguments
        args = [(file) for file in expanded_file_paths]
        results = pool.starmap(read_file, zip(args, [branches]*len(args)))
        # print(results)
        for result in results:
            for branch in branches:
                if branch not in arrays.keys():
                    arrays[branch] = ak.flatten(result[branch])
                else:
                    arrays[branch] = np.concatenate((arrays[branch], ak.flatten(result[branch])))
    return arrays

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

# def flattened_pt_weighted(data, bins):
#     weights = np.zeros(len(ak.flatten(data))) 
#     pt_hist, bin_edges = np.histogram(ak.flatten(data), bins=bins)
#     for i in range(len(pt_hist)):
#         if pt_hist[i] == 0:
#             weights = np.where((ak.flatten(data) >= bin_edges[i]) & (ak.flatten(data) < bin_edges[i+1]), 1, weights)
#         else:
#             weights = np.where((ak.flatten(data) >= bin_edges[i]) & (ak.flatten(data) < bin_edges[i+1]), 1/pt_hist[i], weights)

#     return weights

def flattened_pt_weighted(data, bins, weight):
    weights = np.zeros(len(data))
    pt_hist, bin_edges = np.histogram(data, bins=bins, weights=weight)
    for i in range(len(pt_hist)):
        if pt_hist[i] == 0:
            weights = np.where((data >= bin_edges[i]) & (data < bin_edges[i+1]), 1, weights)
        else:
            weights = np.where((data >= bin_edges[i]) & (data < bin_edges[i+1]), 1/pt_hist[i], weights)

    return weights

# def plot_eff(data, cuts, name, num_bins, x_min, x_max, event_weights=None, pt_weights=None, eta=False):
#     if event_weights is None and pt_weights is None:
#         pt_1p3p_dnom = plt_to_root_hist_w(data[cuts[0]], num_bins, x_min, x_max, None, eta)
#         pt_1p3p_num =  plt_to_root_hist_w(data[cuts[1]], num_bins, x_min, x_max, None, eta)
#         pt_1p1p_dnom = plt_to_root_hist_w(data[cuts[2]], num_bins, x_min, x_max, None, eta)
#         pt_1p1p_num =  plt_to_root_hist_w(data[cuts[3]], num_bins, x_min, x_max, None, eta)
#         pt_3p3p_dnom = plt_to_root_hist_w(data[cuts[4]], num_bins, x_min, x_max, None, eta)
#         pt_3p3p_num =  plt_to_root_hist_w(data[cuts[5]], num_bins, x_min, x_max, None, eta)
#         pt_inc_dnom =  plt_to_root_hist_w(data[cuts[6]], num_bins, x_min, x_max, None, eta)
#         pt_inc_num =   plt_to_root_hist_w(data[cuts[7]], num_bins, x_min, x_max, None, eta)
#     elif event_weights is not None and pt_weights is None:
#         pt_1p3p_dnom = plt_to_root_hist_w(data[cuts[0]], num_bins, x_min, x_max, event_weights[cuts[0]], eta)
#         pt_1p3p_num =  plt_to_root_hist_w(data[cuts[1]], num_bins, x_min, x_max, event_weights[cuts[1]], eta)
#         pt_1p1p_dnom = plt_to_root_hist_w(data[cuts[2]], num_bins, x_min, x_max, event_weights[cuts[2]], eta)
#         pt_1p1p_num =  plt_to_root_hist_w(data[cuts[3]], num_bins, x_min, x_max, event_weights[cuts[3]], eta)
#         pt_3p3p_dnom = plt_to_root_hist_w(data[cuts[4]], num_bins, x_min, x_max, event_weights[cuts[4]], eta)
#         pt_3p3p_num =  plt_to_root_hist_w(data[cuts[5]], num_bins, x_min, x_max, event_weights[cuts[5]], eta)
#         pt_inc_dnom =  plt_to_root_hist_w(data[cuts[6]], num_bins, x_min, x_max, event_weights[cuts[6]], eta)
#         pt_inc_num =   plt_to_root_hist_w(data[cuts[7]], num_bins, x_min, x_max, event_weights[cuts[7]], eta)
#     else:
#         pt_1p3p_dnom = plt_to_root_hist_w(data[cuts[0]], num_bins, x_min, x_max, event_weights[cuts[0]]*pt_weights[cuts[0]], eta)
#         pt_1p3p_num =  plt_to_root_hist_w(data[cuts[1]], num_bins, x_min, x_max, event_weights[cuts[1]]*pt_weights[cuts[1]], eta)
#         pt_1p1p_dnom = plt_to_root_hist_w(data[cuts[2]], num_bins, x_min, x_max, event_weights[cuts[2]]*pt_weights[cuts[2]], eta)
#         pt_1p1p_num =  plt_to_root_hist_w(data[cuts[3]], num_bins, x_min, x_max, event_weights[cuts[3]]*pt_weights[cuts[3]], eta)
#         pt_3p3p_dnom = plt_to_root_hist_w(data[cuts[4]], num_bins, x_min, x_max, event_weights[cuts[4]]*pt_weights[cuts[4]], eta)
#         pt_3p3p_num =  plt_to_root_hist_w(data[cuts[5]], num_bins, x_min, x_max, event_weights[cuts[5]]*pt_weights[cuts[5]], eta)
#         pt_inc_dnom =  plt_to_root_hist_w(data[cuts[6]], num_bins, x_min, x_max, event_weights[cuts[6]]*pt_weights[cuts[6]], eta)
#         pt_inc_num =   plt_to_root_hist_w(data[cuts[7]], num_bins, x_min, x_max, event_weights[cuts[7]]*pt_weights[cuts[7]], eta)

#     pt_1p3p_eff = make_eff_hist(pt_1p3p_num, pt_1p3p_dnom, "1p3p_eff")
#     pt_1p1p_eff = make_eff_hist(pt_1p1p_num, pt_1p1p_dnom, "1p1p_eff")
#     pt_3p3p_eff = make_eff_hist(pt_3p3p_num, pt_3p3p_dnom, "3p3p_eff")
#     pt_inc_eff = make_eff_hist(pt_inc_num, pt_inc_dnom, "inc_eff")

#     pt_1p3p_eff.GetYaxis().SetRangeUser(0, 1)
#     pt_1p1p_eff.GetYaxis().SetRangeUser(0, 1)
#     pt_3p3p_eff.GetYaxis().SetRangeUser(0, 1)
#     pt_inc_eff.GetYaxis().SetRangeUser(0, 1)

#     pt_1p3p_eff.GetXaxis().SetRangeUser(x_min, x_max)
#     pt_1p1p_eff.GetXaxis().SetRangeUser(x_min, x_max)
#     pt_3p3p_eff.GetXaxis().SetRangeUser(x_min, x_max)
#     pt_inc_eff.GetXaxis().SetRangeUser(x_min, x_max)

#     pt_1p3p_eff.GetXaxis().SetTitle(name)
#     pt_1p1p_eff.GetXaxis().SetTitle(name)
#     pt_3p3p_eff.GetXaxis().SetTitle(name)
#     pt_inc_eff.GetXaxis().SetTitle(name)
    
#     pt_1p3p_eff.SetLineColor(ROOT.kBlack)
#     pt_1p1p_eff.SetLineColor(ROOT.kOrange)
#     pt_3p3p_eff.SetLineColor(ROOT.kRed)
#     pt_inc_eff.SetLineColor(ROOT.kGreen)

#     return pt_1p3p_eff, pt_1p1p_eff, pt_3p3p_eff, pt_inc_eff

def plot_eff(data, weights, name, num_bins, x_min, x_max, eta=False, bkg=False):
    
    pt_1p3p_dnom = plt_to_root_hist_w(data[0], num_bins, x_min, x_max, weights[0], eta)
    pt_1p3p_num =  plt_to_root_hist_w(data[1], num_bins, x_min, x_max, weights[1], eta)
    pt_1p1p_dnom = plt_to_root_hist_w(data[2], num_bins, x_min, x_max, weights[2], eta)
    pt_1p1p_num =  plt_to_root_hist_w(data[3], num_bins, x_min, x_max, weights[3], eta)
    pt_3p3p_dnom = plt_to_root_hist_w(data[4], num_bins, x_min, x_max, weights[4], eta)
    pt_3p3p_num =  plt_to_root_hist_w(data[5], num_bins, x_min, x_max, weights[5], eta)
    pt_inc_dnom =  plt_to_root_hist_w(data[6], num_bins, x_min, x_max, weights[6], eta)
    pt_inc_num =   plt_to_root_hist_w(data[7], num_bins, x_min, x_max, weights[7], eta)

    pt_1p3p_eff = make_eff_hist(pt_1p3p_num, pt_1p3p_dnom, "1p3p_eff")
    pt_1p1p_eff = make_eff_hist(pt_1p1p_num, pt_1p1p_dnom, "1p1p_eff")
    pt_3p3p_eff = make_eff_hist(pt_3p3p_num, pt_3p3p_dnom, "3p3p_eff")
    pt_inc_eff = make_eff_hist(pt_inc_num, pt_inc_dnom, "inc_eff")

    if bkg:
        pt_1p3p_eff.GetYaxis().SetRangeUser(0, 10)
        pt_1p1p_eff.GetYaxis().SetRangeUser(0, 10)
        pt_3p3p_eff.GetYaxis().SetRangeUser(0, 10)
        pt_inc_eff.GetYaxis().SetRangeUser(0, 10)
    else:
        pt_1p3p_eff.GetYaxis().SetRangeUser(0.25, 1.3)
        pt_1p1p_eff.GetYaxis().SetRangeUser(0.25, 1.3)
        pt_3p3p_eff.GetYaxis().SetRangeUser(0.25, 1.3)
        pt_inc_eff.GetYaxis().SetRangeUser(0.25, 1.3)

    pt_1p3p_eff.GetXaxis().SetRangeUser(x_min, x_max)
    pt_1p1p_eff.GetXaxis().SetRangeUser(x_min, x_max)
    pt_3p3p_eff.GetXaxis().SetRangeUser(x_min, x_max)
    pt_inc_eff.GetXaxis().SetRangeUser(x_min, x_max)

    pt_1p3p_eff.GetXaxis().SetTitle(name)
    pt_1p1p_eff.GetXaxis().SetTitle(name)
    pt_3p3p_eff.GetXaxis().SetTitle(name)
    pt_inc_eff.GetXaxis().SetTitle(name)

    if bkg:
        pt_1p3p_eff.GetYaxis().SetTitle('1/Efficiency')
        pt_1p1p_eff.GetYaxis().SetTitle('1/Efficiency')
        pt_3p3p_eff.GetYaxis().SetTitle('1/Efficiency')
        pt_inc_eff.GetYaxis().SetTitle('1/Efficiency')
    else:
        pt_1p3p_eff.GetYaxis().SetTitle('Efficiency')
        pt_1p1p_eff.GetYaxis().SetTitle('Efficiency')
        pt_3p3p_eff.GetYaxis().SetTitle('Efficiency')
        pt_inc_eff.GetYaxis().SetTitle('Efficiency')

    pt_1p3p_eff.SetLineColor(ROOT.kBlue+1)
    pt_1p1p_eff.SetLineColor(ROOT.kOrange+8)
    pt_3p3p_eff.SetLineColor(ROOT.kAzure+8)
    pt_inc_eff.SetLineColor(ROOT.kSpring-5)

    return pt_1p3p_eff, pt_1p1p_eff, pt_3p3p_eff, pt_inc_eff

def root_to_numpy(sample, branches):
    ROOT.ROOT.EnableImplicitMT(16)
    df = ROOT.RDataFrame("CollectionTree", sample)
    npy = ak.from_rdataframe(df, columns=branches, keep_order=True)
    return npy 

def uproot_open(file_path, branches):
    f_1 = uproot.open(file_path)
    # f1 = f_1['CollectionTree'].arrays(branches, library='ak', entry_stop=40000000)
    f1 = f_1.arrays(branches, library='ak')
    return f1

def h52panda(filelist, xs, cut, bdt=False, pp13=False, pp11=False, pp33=False, ppinc=False):
    combined = pd.DataFrame()
    chunk_size = 10000000  # Adjust this size to suit your system's memory
    pt_bins = np.linspace(200000, 1000000, 41)

    dataset_keys = ["event_id", "ditau_pt", "IsTruthHadronic",
                "f_core_lead", "f_core_subl", "f_subjet_subl", "f_subjets", "f_isotracks",
                "R_max_lead", "R_max_subl", "R_isotrack", "R_tracks_subl",
                "m_core_lead", "m_core_subl", "m_tracks_lead", "m_tracks_subl",
                "d0_leadtrack_lead", "d0_leadtrack_subl",
                "n_track", "n_tracks_lead", "n_tracks_subl", "n_subjets",
                "event_weight", "bdt_score", "bdt_score_new", "average_mu", "eta"]

    for index in range(len(filelist)):
        file_path = filelist[index]
    
        # Process the file in chunks
        with h5py.File(file_path, 'r') as h5_file:
            # Determine the total length of the datasets
            total_length = h5_file[dataset_keys[0]].shape[0]
            print(f'{filelist[index]}: {total_length}')

            # Read and process each chunk
            for chunk_start in range(0, total_length, chunk_size):
                chunk_end = chunk_start + chunk_size

                # Use slicing to read a chunk from each dataset in the HDF5 file
                data = {key: h5_file[key][chunk_start:chunk_end] for key in dataset_keys}

                # Convert the dictionary to a pandas DataFrame
                df_chunk = pd.DataFrame(data)

                # Apply Cut
                filtered_chunk = df_chunk[cut(df_chunk, bdt, pp13, pp11, pp33, ppinc)]
                filtered_chunk = filtered_chunk.copy()
                filtered_chunk.loc[:, 'event_weight'] = filtered_chunk['event_weight'] * getXS(xs[index])

                combined = pd.concat([combined, filtered_chunk], ignore_index=True)
    
    combined['pT_weight'] = flattened_pt_weighted(combined['ditau_pt'], pt_bins, combined['event_weight'])

    return combined


########## LOKI Functions ##########

class Sample():
    """Class to store details of event samples 

    :param name: simple text identifier for sample
    :type name: str
    :param sty: sample plotting sty attributes 
    :type sty: :class:`loki.core.sty.Style` 
    :param xsec: cross section
    :type xsec: float
    :param is_data: collision data sample (ie not MC) 
    :type is_data: bool 
    :param daughters: sub-samples 
    :type daughters: `loki.core.sample.Sample` list
    :param regex: regular expression for sample directory
    :type regex: str 
    :param files: list of paths to input MxAOD files
    :type files: str list
    :param treename: name of the TTree in the input MxAOD files
    :type treename: str    
    :param sel: selection always applied to sample
    :type sel: :class:`loki.core.var.VarBase`
    :param weight: weight expression
    :type weight: :class:`loki.core.var.VarBase`
    :param scaler: sample scaler
    :type scaler: :class:`loki.core.hist.HistScaler`
    :param mcevents: nominal number of mcevents
    :type mcevents: int
    """
    #____________________________________________________________
    def __init__(self, 
                 name = None,
                 sty = None,
                 xsec = None,
                 is_data = False,
                 daughters = None,
                 regex = None,
                 files = None,
                 treename = None,
                 sel = None,
                 weight = None,
                 scaler = None,
                 mcevents = None,
                 #maxevents = None,
                 ):
        self.name = name
        self.sty = sty 
        self.xsec = xsec
        # TODO may be redundant - remove if not used soon, Feb 2016
        self.is_data = is_data
        self.daughters = []
        if daughters:
            for d in daughters: self.add_daughter(d)
        self.regex = regex or "*%s*"%(name)
        self.files = files
        self.treename = treename or "CollectionTree"
        self.sel = sel
        self.weight = weight
        self.scaler = scaler
        self.mcevents = mcevents or 1000000
        #self.maxevents = maxevents

        #: handle on parent (if daughter) 
        self.parent = None
        #self.__curr_file = None
        #: nevents cache
        self.nev_per_file_dict = {}
        
    #____________________________________________________________
    def style_hist(self,h):
        """Apply *self.sty* to *h*"""
        if self.sty: self.sty.apply(h)

    #____________________________________________________________
    def is_parent(self):
        """Check if sample has daughters"""
        return bool(self.daughters) 

    #____________________________________________________________
    def is_mvdataset(self):
        """Check if sample is appropraite for mv trainig"""
        return len(self.files) == 1 and not self.is_parent()  

    #____________________________________________________________
    def add_daughter(self,sample):
        """Add a daughter to this sample
       
        :param sample: daughter (ie sub-sample)
        :type sample: `loki.core.sample.Sample`
        """
        if not self.daughters: self.daughters = []
        self.daughters.append(sample)
        sample.parent = self

    #____________________________________________________________
    def get_final_daughters(self):
        """Return list of non-parent daughters of this sample.

        Walks recursively through all daughters.
        Will return itself if not a parent.

        :rtype: `loki.core.sample.Sample` list
        """
        if not self.is_parent(): return [self]
        daughters = []
        for d in self.daughters: 
            daughters+=d.get_final_daughters()
        return daughters

    #____________________________________________________________
    def get_nevents(self,fname=None,tree=None):
        """Get number of events for sample or per file (*fname*)
        
        Result is cached.
        
        Need to distinguish before or after MxAOD filter.
        
        :param fname: filename
        :type fname: str
        """
        if fname is None: 
            return sum([self.get_nevents(f) for f in self.files])
            
        if not self.nev_per_file_dict.has_key(fname):
            log().debug("Calculating nevents for %s"%(fname))
            if tree:
                n = tree.GetEntries()
            else:  
                f = ROOT.TFile.Open(fname)
                t = f.Get(self.treename)
                n = t.GetEntries()
                f.Close()                 
            self.nev_per_file_dict[fname]=n
        else:
            log().debug("Using cached nevents for %s"%(fname))
        return self.nev_per_file_dict[fname]

    #____________________________________________________________
    def get_tree(self,fname=None):
        """Get tree per file (*fname*)"""
        ## NOTE: Abandoned TChain because of poor support for PyROOT
        '''
        ch = ROOT.TChain(self.treename)
        ch.SetMakeClass(1) # To prevent segfault on MxAODs since Root 6.08
        if fname:
            ch.Add(fname)
        else: 
            for f in self.files: ch.Add(f)
        return ch
        '''
        if not fname:
            if not self.files:
                log().warn("Sample has no input files, cannot return tree!")
                return None
            if len(self.files)>1:
                log().warn("Called get_tree on multi-file sample. Only using first file.")
            fname = self.files[0]
        #if self.__curr_file and self.__curr_file.GetName() == fname:
        #    f = self.__curr_file
        #else:
        f = ROOT.TFile.Open(fname)
        if not f:
            log().warn("Failed to open file: {}".format(fname))
            return None
        t = f.Get(self.treename)
        if not t:
            log().warn("Failed to get tree: {} from file: {}".format(self.treename,fname))
            f.Close()
            return None
        t._file = f
        #self.__curr_file = f
        return t

    #__________________________________________________________________________=buf=
    def get_arrays(self, invars=None, noweight=False):
        """Return data as standard python arrays
 
        If *invars* specified, only those branches will be included. 
        TODO: this is not true, would need to transfer get_invar_items method from TreeData
                 
        If *self.wei* defined, will be added as "weight" column.

        Uses :func:`loki.core.process.tree2arrays`
        
        :param invars: subset of variables to include
        :type invars: list str
        :rtype: :class:`numpy.ndarray`
        """
        if not invars: return None
        if self.weight is not None and not noweight:
            invar_names = [v.get_name() for v in invars]
            # avoid double weight entry, and put at back  
            if "weight" in invar_names:
                invars.append(invars.pop(invar_names.index("weight")))
            else: 
                invars += [self.weight]
        
        # unleash
        log().info("Stripping vars from {}".format(str(self.files)))
        from loki.core.process import tree2arrays
        return tree2arrays(self.get_tree(), invars, sel=self.sel) 
        
    #__________________________________________________________________________=buf=
    def get_ndarray(self, invars=None, noweight=False):
        """Return data as numpy ndarray

        If *invars* specified, only those branches will be included. 
        TODO: this is not true, would need to transfer get_invar_items method from TreeData
                 
        If *self.wei* defined, will be added as "weight" column.

        Uses root_numpy.tree2array.
        
        :param invars: subset of variables to include
        :type invars: list str
        :rtype: :class:`numpy.ndarray`
        """
        if not invars: return None
        if self.weight is not None and not noweight:
            invar_names = [v.get_name() for v in invars]
            # avoid double weight entry, and put at back  
            if "weight" in invar_names:
                invars.append(invars.pop(invar_names.index("weight")))
            else: 
                invars += [self.weight]
        t = self.get_tree()
        for v in invars: v.tree_init(t)        
        invar_exprs = [v.get_expr() for v in invars]
        if self.sel: 
            self.sel.tree_init(t)
            sel_expr = self.sel.get_expr()
        else:
            sel_expr = None  
        
        # unleash
        from root_numpy import tree2array
        log().info("Stripping vars from {}".format(str(self.files)))
        d = tree2array(t, branches=invar_exprs, selection=sel_expr)
        # rename weight branch
        if self.weight is not None and not noweight: 
            d.dtype.names =  list(d.dtype.names[:-1]) + ["weight"]
        return d

    #____________________________________________________________
    def get_scale(self):
        if not self.scaler: 
            log().warn("Attempt to get scale of sample %s that has no scaler"%(self.name))
            return None
        return self.scaler.scale(self)

    #____________________________________________________________
    def get_all_files(self):
        """Returns a list of paths to all input MxAOD files 
        including daughters. 
        
        Walks recursively through all daughters. 

        :rtype: str list
        """
        files = []
        if self.is_parent():
            for d in self.daughters: 
                files += d.get_all_files()
        elif self.files: files += self.files
        return files

    #____________________________________________________________
    def is_active(self):
        """Retruns true if sample has input files"""
        return bool(self.get_all_files()) 

    #____________________________________________________________
    def load_files(self,basedir=None):
        """Load files in *basedir* matching :attr:`regex`

        Walks recursively through all daughters.

        :param basedir: directoy to scan for files
        :type basedir: str 
        """
        if self.is_parent(): 
            for d in self.daughters: d.load_files(basedir)
        else: 
            # check if regex defined
            regex   = self.regex
            if not regex: 
                log().warn("No regex defined for %s, cannot scan for input files"%(self.name))
                return
            
            # scan for files
            basedir = basedir or "."
            self.files = glob("%s/%s/*.root*"%(basedir,regex))
            if not self.files: 
                log().warn("No files found for sample %s"%(self.name))

    #__________________________________________________________________________=buf=
    def has_varname(self, varname):
        """Return true if *var* exists in dataset
        
        :param varname: variable name
        :type varname: str
        :rtype: bool
        """
        t = self.get_tree()
        if not t: return False
        return bool(t.GetLeaf(varname))

    #____________________________________________________________
    def clone(self,tag,regexmod=None):
        """Return a clone of the dataset named "<name>_<tag>". 
        
        The cross section, selection, weights, sample scaler and 
        is_data flag from the original sample are retained. 
        The style, files and treename are not set, as they are 
        expected to be different and/or determined later.
        
        The regex for input file scanning can be updated (see below). 
        
        The final daughters will also be cloned with the same *tag* 
        and *regexmod* arguments. 
        
        The logic for updating the regex can be provided by the string *regexmod*.
        It can contain the following keywords, which will be replaced by their
        existing values:: 
        
            {regex} - the current regex of the sample
            {name}  - the current name of the sample
            {tag}   - the new tag
        
        For example, if you want the new regex to match the old regex
        preceeded by the new tag and allowing for additional charachters inbetween, 
        you would do this:: 
            
            regex = "*{tag}*{regex}*"
            

        :param basedir: directoy to scan for files
        :type basedir: str 
        """
        # new name
        name = "{0}_{1}".format(self.name, tag)
        
        # new regex
        regex = None
        if regexmod is not None: 
            regkw = {}
            if regexmod.count("{regex}"): regkw["regex"] = self.regex
            if regexmod.count("{name}"): regkw["name"] = self.name
            if regexmod.count("{tag}"): regkw["tag"] = tag
            regex = regexmod.format(**regkw)        
            regex = regex.replace("**","*")
            
        # new sample 
        snew = Sample(name=name,
                      regex=regex,
                      xsec=self.xsec,
                      is_data=self.is_data,
                      sel=self.sel,
                      weight=self.weight,
                      scaler=self.scaler,                   
                      )
        
        # clone the daughters:     
        if self.is_parent(): 
            # exact lineage doesn't matter, just use final daughters
            daughters = self.get_final_daughters()
            for d in daughters: 
                dnew = d.clone(tag,regexmod)
                snew.add_daughter(dnew)
    
        return snew

    #____________________________________________________________
    def serialize(self):
        """Return sample in dict format"""
        sel_str = self.sel.get_name() if self.sel else None
        wei_str = self.weight.get_name() if self.weight else None
        #return {"name":self.name, "files":self.files, "sel":sel_str, "weight":wei_str}

        # Warn against serialization of sample selection to simplify
        # Alg training workflow. Rather, you should apply selection by
        # skimming your flat ntuples. Also, don't write null sel_str,
        # to avoid encouraging people to use it.
        d = {"name": self.name, "files": self.files, "weight": wei_str}
        if sel_str:
            log().warn("Serialization of sample selection not recommended as it is not supported by AlgBase!")
            log().warn("sel_str: {}".format(sel_str))
            d["sel"] =sel_str
        return d

    #____________________________________________________________
    def __hash__(self):
        """Define hash identifier using hash of *self.name*"""
        return self.name.__hash__()


class Container(object):
    """Class that represents an xAOD object container
    
    The name of the container should not include the 'Aux' or 'Dyn' suffixes; 
    they will be auto-determined on a variable by variable basis from the input 
    tree by the :class:`loki.core.var.Var`. 
    
    Variables already existing in the container in the input tree can be registered 
    by :func:`add_var`.
    
    Complex and/or multi-variable expressions (including selection criteria) can 
    be created and added to the container with :func:`add_expr`. 
    
    Groups of selection criteria can be created using :func:`add_cuts`.
    
    Groups of weight expressions can be created using :func:`add_weights`.
    
    :param name: name of the container
    :type name: str
    :param single_valued: if container only ever has one value per event
    :type single_valued: bool
    """
    instances = dict()
    #__________________________________________________________________________=buf=
    def __init__(self, name, single_valued=False):
        # config 
        self.name = name
        self.single_valued = single_valued

        # members
        self.vars = []

        # add to global instances        
        if name in Container.instances: 
            log().warn("Attempt to create multiple containers with name {0}, duplicates will not be available on lookup".format(name))
            return
        Container.instances[name] = self

    #__________________________________________________________________________=buf=
    def get_var(self, name): 
        """Get variable by name
        
        :rtype: :class:`loki.core.var.Var`
        """ 
        matches = [v for v in self.vars if v.name == name]
        if not matches: return None
        return matches[0]

    #__________________________________________________________________________=buf=
    def has_var(self, name): 
        """Return True if variable in container"""
        return self.get_var(name) is not None

    #__________________________________________________________________________=buf=
    def add_var(self, name, var = None, xtitle = None, xunit = None, 
               short_title = None, truth_partner = None): 
        """Adds a new simple variable to container and returns it.
        
        See :class:`loki.core.var.Var` for details.
        
        :rtype: :class:`loki.core.var.Var`
        """ 
        # check duplicate
        if name in [v.name for v in self.vars]: 
            log().warn("Attempting to add duplicate var {var} to container {cont}".format(
                        var = name, cont = self.name))
            return None  
        
        # create var              
        v = Var(name, var=var, xtitle=xtitle, xunit=xunit, cont=self, 
                short_title=short_title, truth_partner=truth_partner)
        self.vars.append(v)
        
        # decorate
        setattr(self, name, v)

        return v

    #__________________________________________________________________________=buf=
    def add_expr(self, name, expr = None, invars = None, xtitle = None, xunit = None, 
               short_title = None, truth_partner = None): 
        """Adds complex and/or multi-variable expression to container and returns it.
        
        See :class:`loki.core.var.Expr` for details. 
        
        :rtype: :class:`loki.core.var.Expr`
        """
        # check duplicate 
        if name in [v.name for v in self.vars]: 
            log().warn("Attempting to add duplicate var {var} to container {cont}".format(
                        var = name, cont = self.name))
            return None                
            
        # create expression    
        v = Expr(name, expr=expr, invars=invars, xtitle=xtitle, xunit=xunit, 
                 cont=self, short_title=short_title, truth_partner=truth_partner)
        self.vars.append(v)
        
        # decorate
        setattr(self, name, v)

        return v

    #__________________________________________________________________________=buf=
    def add_cuts(self, name, cuts = None): 
        """Adds expression of compound selection criteria to container and returns it.
        
        See :class:`loki.core.var.Cuts` for details
        
        :rtype: :class:`loki.core.var.Cuts`
        """ 
        # check duplicate 
        if name in [v.name for v in self.vars]: 
            log().warn("Attempting to add duplicate var {var} to container {cont}".format(
                        var = name, cont = self.name))
            return None                
            
        # create invars        
        v = Cuts(name, cuts = cuts, cont=self)
        self.vars.append(v)
        
        # decorate
        setattr(self, name, v)

        return v

    #__________________________________________________________________________=buf=
    def add_weights(self, name, weights = None): 
        """Adds expression of compound invars to container and returns it.
        
        See :class:`loki.core.var.Weights` for details
        
        :rtype: :class:`loki.core.var.Weights`
        """ 
        # check duplicate 
        if name in [v.name for v in self.vars]: 
            log().warn("Attempting to add duplicate var {var} to container {cont}".format(
                        var = name, cont = self.name))
            return None                
            
        # create invars        
        v = Weights(name, weights = weights, cont=self)
        self.vars.append(v)
        
        # decorate
        setattr(self, name, v)

        return v

    #__________________________________________________________________________=buf=
    def del_var(self, name): 
        """Delete variable"""
        v = self.get_var(name)
        if not v: 
            log().warn("Can't delete variable {0}".format(name))
            return
        self.vars.remove(v)
        del v


def find_container(name):
    """Return container by name
    
    :param name: container name
    :type name: str
    :rtype: :class:`Container`
    """
    return Container.instances.get(name, None)


gInitialized = False
def initialize(level=None):
    """initialize global logger
    
    :param level: logging output level
    :type level: logging.LEVEL (eg. DEBUG, INFO, WARNING...)
    """
    logging.basicConfig(
        filemode='w',
        level=level if level!=None else logging.INFO,
        #format='[%(asctime)s %(name)-16s %(levelname)-7s]  %(message)s',
        format='[%(asctime)s %(levelname)-7s]  %(message)s',
        #datefmt='%Y-%m-%d %H:%M:%S',
        datefmt='%H:%M:%S',
        )
    logging.getLogger("global")
    if supports_color():
        logging.StreamHandler.emit = add_coloring_to_emit_ansi(logging.StreamHandler.emit)
def log():
    """Return global logger"""
    global gInitialized
    if not gInitialized: 
        initialize()
        gInitialized = True
    return logging.getLogger("global")   
def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.
    """
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or
                                                  'ANSICON' in os.environ)
    # isatty is not always implemented
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True

#______________________________________________________________________________=buf=
def add_coloring_to_emit_ansi(fn):
    # add methods we need to the class
    def new(*args):
        levelno = args[1].levelno
        if(  levelno>=logging.CRITICAL): color = enums.REDBKG
        elif(levelno>=logging.ERROR)   : color = enums.REDBOLD 
        elif(levelno>=logging.WARNING) : color = enums.RED 
        elif(levelno>=logging.INFO)    : color = enums.DEFAULT 
        elif(levelno>=logging.DEBUG)   : color = enums.YELLOW 
        else:                            color = enums.YELLOW 
        args[1].msg = color + args[1].msg +  enums.UNSET
        return fn(*args)
    return new
 

class VarBase(object):
    """Base class for TTree variables and expressions. 

    This class implements some basic common functionality and defines the 
    virtual interfaces to be implemented by the derived variable classes.  
    
    *tree_init* must be called before using the variable with an input tree 
    
    VarBase objects can be combined via the ``&`` operator, creating a 
    :class:`loki.core.var.Cuts` object, which represents a selection string.
    
    VarBase objects can be combined via the ``*`` operator, creating a 
    :class:`loki.core.var.Weights` object, which represents a weight string. 
    
    :param name: unique identifier
    :type name: str
    :param cont: container name in MxAOD
    :type cont: str
    :param invars: input variables
    :type invars: list :class:`loki.core.var.VarBase`
    :param xtitle: TLatex formatted title of variable to be used on plots
    :type xtitle: str
    :param short_title: short title mainly used for resolution variables
    :type short_title: str
    :param xunit: unit for variable
    :type xunit: str
    :param truth_partner: truth-tau equivalent to matched variable
    :type truth_partner: :class:`Var`
    :param temp: if True, don't add to global variable registry
    :type temp: bool
    """
    global_instances = dict()
    counter = 0
    #__________________________________________________________________________=buf=
    def __init__(self, 
            name    = None, 
            cont    = None, 
            invars  = None,
            xtitle  = None,
            short_title = None,
            xunit   = None,
            truth_partner = None,
            temp = False,
            ):      
        ## defaults 
        if xtitle is None: 
            if cont is None: xtitle = name 
            else: xtitle = "{0} {1}".format(cont.name,name)
        if short_title is None: short_title = xtitle
        if truth_partner is None: truth_partner = self
        
        ## config 
        self.name          = name
        self.cont          = cont
        self.invars        = invars or []
        self.xtitle        = xtitle
        self.short_title   = short_title
        self.xunit         = xunit                
        self.truth_partner = truth_partner
        self.temp          = temp

        ## members
        self.views = []        
        self.isint = False

        ## initialization checks
        self.__check_invars__()

        ## set unique id
        self.uid = int(VarBase.counter)
        VarBase.counter+=1
        #print "{0:4d}: {1}".format(self.uid, self.get_name())
        
        ## add to instances if not in container
        if not temp and cont is None:
            if name in VarBase.global_instances: 
                log().warn("Attempt to create multiple global vars with name {0}, duplicates will not be available on lookup".format(name))
                return
            VarBase.global_instances[name] = self


    # 'Virtual' interfaces (to be implemented in derived class)
    #__________________________________________________________________________=buf=
    def get_expr(self):
        """Return the expression string (eg. for TTree::Draw)
        
        IMPORTANT: implementation must be provided by derived class.
        
        :rtype: str        
        """
        log().warn("Class derived from VarBase must provide implementation of get_expr()!")
        return None

    #__________________________________________________________________________=buf=
    def tree_init(self,tree):
        """Initialise variable to work on *tree*. Return True if success.
        
        IMPORTANT: implementation must be provided by derived class.
        
        :param tree: target tree
        :type tree: :class:`ROOT.TTree`
        :rtype: bool
        """
        log().warn("Class derived from VarBase must provide implementation of treeinit()!")
        return False

    # Concrete interfaces (functionality provided by VarBase base class)
    #__________________________________________________________________________=buf=
    def sample_init(self,sample):
        """Initialise variable to work on *sample*. Return True if success.
        
        :param sample: target sample
        :type sample: :class:`loki.core.sample.Sample`
        :rtype: bool
        """
        t = sample.get_tree()
        if not t:
            log().warn("Cannot init var on sample without valid tree") 
            return False
        return self.tree_init(t)
    
    #__________________________________________________________________________=buf=
    def add_view(self, nbins=None, xmin=None, xmax=None, name = None, ytitle = None, 
                 do_logx = False, do_logy = False, xbins = None, binnames = None):
        """Add a *view* to the variable then return the variable. 
        
        Bins are created with standard (linear) spacing provided 
        by (*nbins*, *xmin*, *xmax*). Logarithmic spacing used if 
        *do_logx* is True. Custom bin spacing can be specified via *xbins*.
                 
        Variable is returned to support multiple concatenated calls to 'add_view' 

        See :class:`loki.core.var.View` for documentation. 
        
        :param nbins: number of bins
        :type nbins: int
        :param xmin: minimum of x-axis range
        :type xmin: float        
        :param xmax: maximum of x-axis range
        :type xmax: float
        :param name: name of view
        :type name: str
        :param ytitle: y-axis title (if default not adequate)
        :type ytitle: str
        :param do_logx: use log binning and scale for x-axis
        :type do_logx: bool
        :param do_logy: use log-scale for y-axis
        :type do_logy: bool
        :param xbins: custom bin edges (overrides *nbins*, *xmin*, *xmax*)
        :type xbins: list float
        :rtype: :class:`loki.core.var.VarBase`
        """
        # unless xbins provided, check nbins, xmin, xmax
        if xbins is None:
            if None in [nbins,xmin,xmax]: 
                log().error("Must provide (nbins,xmin,xmax) or xbins to add_view")
                raise ValueError
        
        binwidth = None
        if not xbins: 
            if do_logx: 
                xbins = log_bins(nbins,xmin,xmax)
            else:       
                xbins = bins(nbins,xmin,xmax)
                binwidth = (xmax - xmin) / float(nbins)
        v = View(self, xbins, name=name, ytitle=ytitle, do_logy=do_logy, 
                 do_logx=do_logx, binwidth=binwidth, binnames=binnames)
        
        # check for duplicate
        if v.name in [itrv.name for itrv in self.views]: 
            log().warn("Attempt to add multiple views with non-unique name {view} to var {var}".format( \
                       view = v.name, var = self.get_name()))
        else: 
            self.views.append(v)
            
        return self

    #__________________________________________________________________________=buf=
    def get_name(self):
        """Return the variable name""" 
        if self.cont is None: return self.name
        return "{0}_{1}".format(self.cont.name,self.name)

    #__________________________________________________________________________=buf=
    def get_newbranch(self):
        """Return the new branch name (eg. used on tree writeout)""" 
        if self.cont is None: return self.name
        return "{0}.{1}".format(self.cont.name,self.name)

    #__________________________________________________________________________=buf=
    def get_view(self, name = None):
        """Return view of the variable specified by *name*. 

        Return first (default) view if *name* not passed. 
        Return None if view not found.   

        :param name: name of view 
        :type name: str 
        :rtype: :class:`loki.core.var.View`
        """
        if not self.views: 
            log().warn("Attempted to retrieve view from {0} which has no views!".format(self.get_name()))
            return None

        ## default view
        if name is None: 
            return self.views[0]
        
        ## specific view
        matches = [v for v in self.views if v.name == name]
        if not matches: 
            log().warn("Couldn't find view {view} in var {var}!".format( \
                    view = name, var = self.get_name()))
            return None
        return matches[0] 

    #__________________________________________________________________________=buf=
    def has_view(self, name = None):
        """Return True if view of the variable specified by *name* exists."""
        if not self.views: return False
        matches = [v for v in self.views if v.name == name]
        return bool(matches)
        
    #__________________________________________________________________________=buf=
    def get_xtitle(self):
        """Return the x-axis title (including unit) of the varaible

        :rtype: str
        """
        xtitle = '%s'%self.xtitle
        if self.xunit: xtitle = '%s [%s]'%(self.xtitle, self.xunit)
        return xtitle

    #__________________________________________________________________________=buf=
    def get_short_title(self):
        """Return the "short title", used mainly for response variables

        Eg. in :class:`loki.core.hist.ResoProfile` we need to name 
        the y-axis something like "Tau pT Resolution", while the 
        yvar is "Reco Tau pT / True Tau pT"

        :rtype: str
        """
        return self.short_title

    #__________________________________________________________________________=buf=
    def get_inconts(self):
        """Return set of containers use by invars
        
        :rtype: set :class:`loki.core.var.Container`
        """
        if not self.invars: return set([self.cont])
        s = set()
        for v in self.invars: 
            s = s.union(v.get_inconts())
        return s

    #__________________________________________________________________________=buf=
    def get_mvinconts(self):
        """Return set of mutlivalued containers used by invars
        
        :rtype: set :class:`loki.core.var.Container`
        """
        return set([c for c in self.get_inconts() if c is not None and not c.single_valued]) 

    #__________________________________________________________________________=buf=
    def get_type(self):
        """Return the value type for the variable
        
        Note: can only be called after variable initialization
        
        :rtype: str ('f','i')
        """
        if self.is_integer(): return 'i'
        return 'f' 

    #__________________________________________________________________________=buf=
    def serialize(self):
        """Return var in string format"""
        if self.cont is None: return self.name
        return "{0}.{1}".format(self.cont.name,self.name) 

    #__________________________________________________________________________=buf=
    def is_multivalued(self):
        """Return True if from multivalued container"""
        return bool(self.get_mvinconts())

    #__________________________________________________________________________=buf=
    def is_integer(self):
        """Return True if is integer valued
        
        Note: can only be called after variable initialization
        """
        #if self.leafname is None: 
        #    log().warn("Attempt to call 'is_integer' on %s before tree initialization"%(self.get_name()))
        #    return False
        return self.isint

    # 'Private' interfaces
    #__________________________________________________________________________=buf=
    def __check_invars__(self):
        """Raise error if invars contains multiple different multivalued containers"""
        log().debug("checking invars for {0}...".format(self.get_name()))
        if not self.invars: return
        # don't allow multiple multivalued input containers
        if len(self.get_mvinconts())>1: 
            log().error("For {0}: Variables not allowed to be comprised of multiple multivalued containers!".format(self.get_name()))
            raise VarError
        log().debug("invars OK!")
        return
        
    #__________________________________________________________________________=buf=
    def __hash__(self):
        """Return unique object hash (from name)

        :rtype: hash
        """
        # TODO: need to fix this to ensure it is unique
        #if self._leafname is None: 
        #    return self.get_name().__hash__()
        #return self._leafname.__hash__()
        return self.get_name().__hash__()

    #__________________________________________________________________________=buf=
    def __eq__( self, other ):
        """Define comparison operator"""
        return bool(self.__hash__() == other.__hash__())


    #____________________________________________________________
    def __and__(self,other):
        """Override ``&`` operator to combine two VarBase objects into Cuts"""
        if other is None: return self
        name = "%s_%s"%(self.name,other.name)
        cont = self.cont
        cuts = [self,other]
        return Cuts(name,cuts,cont=cont,temp=True)

    #____________________________________________________________
    def __mul__(self,other):
        """Override ``*`` operator to combine two VarBase objects into Weights"""
        if other is None: return self
        name = "%s_%s"%(self.name,other.name)
        cont = self.cont
        weights = [self,other]
        return Weights(name,weights,cont=cont,temp=True)

    #____________________________________________________________
    def __str__(self):
        """Override string for nice output"""
        return self.serialize()


class View(object):
    """Class that provides a specific view (binning) for a given variable 

    :param var: parent variable 
    :type var: :class:`loki.core.var.VarBase`    
    :param xbins: list of bin edges
    :type xbins: float list    
    :param name: unique identifier
    :type name: str
    :param ytitle: y-axis title to be used with variable (default to Nevents)
    :type ytitle: str
    :param do_logy: if variable should be plotted with log-scale on y-axis
    :type do_logy: bool 
    :param do_logx: if variable should be plotted with log-scale on x-axis
    :type do_logx: bool
    :param binwidth: width of bins (if fixed)
    :type binwidth: float
    """
    #__________________________________________________________________________=buf=
    def __init__(self, 
            var,
            xbins,            
            name    = None,
            ytitle  = None,
            do_logy = False,
            do_logx = False,
            binwidth = None,
            binnames = None,
            ):
      
        ## config 
        self.var     = var
        self.xbins   = xbins        
        self.name    = name or "default"        
        self.ytitle  = ytitle
        self.do_logy = do_logy
        self.do_logx = do_logx
        self.binwidth = binwidth
        self.fixedwidth = binwidth is not None
        self.binnames = binnames
        if binnames: 
            assert len(binnames) <= len(xbins), \
                "binnames must not be longer than xbins in var: {0}" \
                    .format(self.get_name())

    #__________________________________________________________________________=buf=
    def get_bins(self):
        """Return list of bin edges
                
        :rtype: float list
        """
        return list(self.xbins)

    #__________________________________________________________________________=buf=
    def get_name(self):
        """Return view name, format: "<var>_<view>"
                
        :rtype: str
        """
        return "{0}_{1}".format(self.var.get_name(),self.name)

    #__________________________________________________________________________=buf=
    def get_truth_partner(self):
        """Access to self.var.get_expr()"""
        if self.var.truth_partner is None: 
            print("No Truth Partner specified for var")
            return None
        print("Returning truth partner view")
        # return same view of truth partner
        # TODO: may need to add some protection that 
        # truth partner view is identical, or get 
        # rid of them completely!
        return self.var.truth_partner.get_view(self.name)

    #__________________________________________________________________________=buf=
    def get_xmax(self):
        """Return maximum x-value
        
        :rtype: float
        """
        return self.xbins[-1]

    #__________________________________________________________________________=buf=
    def get_xmin(self):
        """Return minimum x-value
        
        :rtype: float
        """
        return self.xbins[0]

    #__________________________________________________________________________=buf=
    def get_ytitle(self):
        """Return the y-axis title (including bin width) of the variable.

        If :attr:`View.ytitle` is not set will return "Events / <bin width>"

        :rtype: str
        """
        if self.ytitle: return self.ytitle
        if not self.fixedwidth: 
            log().debug( 'plot has custom binning, cant include bin width in yaxis title' )
            return 'Events'
        bin_width = self.binwidth
        bin_width_str = '%.1g'%bin_width
        xunit = self.var.xunit
        if bin_width >= 10: bin_width_str = '%.3g'%bin_width
        if bin_width == 1:
            if xunit: return 'Events / %s'%xunit
            return 'Events'
        elif xunit:
            return 'Events / %s %s'%(bin_width_str,xunit)
        return 'Events / %s'%bin_width_str

    #__________________________________________________________________________=buf=
    def serialize(self):
        """Return view in string format"""
        return "{}:{}".format(self.var.serialize(), self.name) 

    #__________________________________________________________________________=buf=
    def new_hist(self, yvar=None, zvar=None, name=None):
        """Return an empty histogram customized for this variable.
        
        :rtype: :class:`ROOT.TH1`
        """
        # create hist
        args = histargs(self,yvar,zvar,name)
        h = new_hist(*args)
        
        # set binnames
        if self.binnames: 
            log().debug("Setting bin labels: {0}".format(str(self.binnames)))
            set_axis_binnames(h.GetXaxis(), self.binnames)
        if yvar and yvar.binnames: 
            log().debug("Setting bin labels: {0}".format(str(yvar.binnames)))
            set_axis_binnames(h.GetYaxis(), yvar.binnames)
        if zvar and zvar.binnames: 
            log().debug("Setting bin labels: {0}".format(str(yvar.binnames)))
            set_axis_binnames(h.GetZaxis(), zvar.binnames)
            
        return h

    #__________________________________________________________________________=buf=
    def frame(self,pad, xmin = None, ymin = None, xmax = None, ymax = None,
              xtitle = None, ytitle = None, yvar = None):
        """Return a frame for the variable

        If *xmin* and *xmax* not provided will be determined from variable.
        *ymin* and *ymax* should be provided, otherwise dummy 0 and 1 are used.

        :param pad: pad (or canvas) on which to draw the frame
        :type pad: :class:`ROOT.TPad`
        :param xmin: x-axis minimum
        :type xmin: float
        :param xmax: x-axis maximum
        :type xmax: float
        :param ymin: y-axis minimum
        :type ymin: float
        :param ymax: y-axis maximum
        :type ymax: float
        :param xtitle: x-axis title 
        :type xtitle: str        
        :param ytitle: y-axis title 
        :type ytitle: str
        :param yvar: y-axis variable view
        :type yvar: :class:`loki.core.var.View`

        """
        # determine boundaries
        if xmin == None and xmax is not None: xmin = self.get_xmin() 
        if xmax == None and xmin is not None: xmax = self.get_xmax()
        if ymin == None: ymin = 0.
        if ymax == None: ymax = 1.

        # determine axis titles
        xtitle = xtitle or self.get_xtitle()
        ytitle = ytitle or self.get_ytitle()

        # build and return frame
        #return pad.DrawFrame(xmin,ymin,xmax,ymax,';%s;%s'%(xtitle,ytitle))
        pad.cd()
        name = "fr_{0}".format(self.get_name())
        if yvar: name += "_{0}".format(yvar.get_name())
        fr = self.new_hist(yvar=yvar,name=name)
        if xmin is not None or xmax is not None: 
            fr.GetXaxis().SetRangeUser(xmin,xmax)
        fr.GetYaxis().SetRangeUser(ymin,ymax)
        fr.GetXaxis().SetTitle(xtitle)
        fr.GetYaxis().SetTitle(ytitle)
        fr.SetTitle("")
        fr.Draw()
        return fr

    ## Copied interfaces from self.var
    #__________________________________________________________________________=buf=
    def get_expr(self):
        """Access to self.var.get_expr()"""
        return self.var.get_expr() 

    #__________________________________________________________________________=buf=
    def get_xtitle(self):
        """Access to self.var.get_xtitle()"""
        return self.var.get_xtitle()

    #__________________________________________________________________________=buf=
    def get_short_title(self):
        """Access to self.var.get_short_title()"""
        return self.var.get_short_title()

    #__________________________________________________________________________=buf=
    def __hash__(self):
        """Return unique object hash (from name)

        :rtype: hash
        """
        return self.get_name().__hash__()

    #__________________________________________________________________________=buf=
    def __eq__( self, other ):
        """Define comparison operator"""
        return bool(self.__hash__() == other.__hash__())

    #____________________________________________________________
    def __str__(self):
        """Override string for nice output"""
        return self.serialize()
        
def find_variable(name):
    """Return variable by name

    :param name: variable name
    :type name: str    
    :rtype: :class:`VarBase` sub-class (None if not found)    
    """
    if name is None: return None
    if name.count(".") > 1: 
        log().warn("Variable name cannot contain multiple periods (ie '.')")
        return None
    # container variable
    elif name.count(".") == 1: 
        (cname, vname) = name.split('.')
        print("FFFFF: ", cname, vname)
        cont = find_container(cname)
        print("CONT: ", cont)
        if cont is None: return None
        return cont.get_var(vname)
    
    # global variable
    return VarBase.global_instances.get(name,None)

def get_view(view_str):
    """Return variable view by name or create using tuple

    Eg. for *view_str* is 'TauJets.pt:log' returning the 'log' view for 
    TauJets.pt or 'TauJets.eta' returning the default view for TauJets.eta.
    
    Can also specify a new view binning in format: VAR:NBINS;XMIN;XMAX
    
    The View will receive the name 'NBINS;XMIN;XMAX' 
    
    Note: vars don't have to be predefined. Undefined vars will be created on the 
    fly using :class:`loki.core.var.StaticExpr`. 
    
    
    :param view_str: view string: 'VAR:VIEW' or 'VAR:NBINS;XMIN;XMAX'
    :type view_str: str
    """
    if isinstance(view_str, View): return view_str
    
    viewargs = None
    # no view provided: attempt to get default view
    # var must be predefined (no StaticExpr)
    if not view_str.count(":"): 
        varname = view_str.strip()
        var = find_variable(varname)
        if not var: 
            log().error("Requested VAR {0} not found! If no view provided, var must be predefined".format(varname))
            return None
        view = var.get_view()
        if not view: 
            log().error("Default VIEW for VAR {1} not found!".format(varname))
        return view
    
    # view tuple provided: 
    (varname,viewname) = [v.strip() for v in view_str.split(':')]
    if viewname.count(";"):
        d = [v.strip() for v in viewname.split(';')] 
        viewargs = (int(d[0]), float(d[1]), float(d[2]))
        if not len(viewargs) == 3:
            log().error("View args must be in format 'nbins;xmin;xmax'") 
            return None
        var = get_variable(varname)
        var.add_view(*viewargs, name=viewname)
        return var.get_view(viewname)
    
    # named view provided
    var = find_variable(varname)
    if not var: 
        log().error("Requested VAR {0} not found! Cannot create named view".format(varname))
        return None
    view = var.get_view(viewname)
    if not view: 
        log().error("Requested VIEW {0} for VAR {1} not found!".format(viewname, varname))
    return view        

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class LokiEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert ndarray to list
            return obj.tolist()
        return super().default(obj)

class AlgBase(object):
    """Generic interface for multivariate algorithm training/prediction 
    
    Subclasses should be written for specific algorithms, and should 
    overload :func:`__subclass_train__` and :func:`__subclass_predict__`.
    
    Input data is provided in the form of :class:`~loki.core.sample.Sample`. 
    It is intended that these are comprised of single flat ntuples. If you 
    are starting from nested MxAOD samples, merge them into single trees via 
    :func:`~loki.train.ntup.flatten_ntup`.
    An example can be found in: example09_flatntup.py   
        
    Specific implemenations are available in :mod:`loki.train.algs`.

    Important Caveats when building AlgBase subclasses: 
    
    * named args passed to sub-class constructor must be set as attributes
      with the same name as the arg, otherwise alg serialization will fail.
    * as training is performed in a tmpdir, the :func:`__get_abspath_worker__` 
      and :func:`__get_sample_worker__` helper functions should be used 
      to access config files and samples in the subclass specific 
      :func:`__subclass_train__` method.

    Other than the *name* argument, no other arguments are intended to be 
    passed by the user to the subclass constructor. Rather they are 
    either set in the :func:`load` method or in the subclass constructor.

    :param name: algorithm name
    :type name: str
    :param wspath: workspace path (don't set yourself, passed by :func:`~loki.train.alg.load`) 
    :type wspath: str
    :param valtype: value type of algorithm output (set by subclass constructor)
    :type valtype: char
    :param info: algorithm info (don't set yourself, passed by :func:`~loki.train.alg.load`)
    :type info: dict
    """
    # 'static' variables
    fname_json = "config.json"
    dname_aux = "aux"    
    #__________________________________________________________________________=buf=
    def __init__(self, name = None, wspath = None, valtype = None, info = None):
        # general defaults
        if name is None: name = "MyAlgorithm"
        if valtype is None: valtype = 'f'
        if info is None: info = dict()
        
        # need to have abs path to allow processing in tmp dir
        if wspath: wspath = os.path.abspath(wspath) 

        # attributes
        self.name = name
        self.wspath = wspath
        self.valtype = valtype
        self.info = info
            
        # get constructor args (removing 'self')
        self.__stream_attrs__ = list(self.__init__.__code__.co_varnames)[1:]
        self.submitdir = os.getcwd()
        self.tmpdir = None
        self.logout = None
        self.logerr = None

        if wspath and not self.ispersistified():
            log().warn("Passing 'wspath' to alg constructor does not persistify, use alg.saveas(wspath)") 

    # 'Public' interface
    #__________________________________________________________________________=buf=
    def train(self):
        """Train the classifier"""
        self.__start_train__()
        if not self.__subclass_train__():
            log().error("Problem encountered during training")
            return False
        self.__end_train__()
        return True
        
    #__________________________________________________________________________=buf=
    def predict(self, s, **kw):
        """Predict classifier output 
                
        :param s: sample
        :type s: :class:`~loki.core.sample.Sample`
        :param kw: key-word args passed to sub-class predict
        :rtype: class:`array`
        """
        return self.__subclass_predict__(s, **kw)
    
    #__________________________________________________________________________=buf=
    def get_var_name(self, **kw):
        """Return unique tag based on kw args passed to the predict method 
        (can override in subclass)
        """
        return self.name

    #__________________________________________________________________________=buf=
    def ispersistified(self):
        """Return True if alg is persistified"""    
        return self.info.get("persistified", False)
    
    #__________________________________________________________________________=buf=
    def saveas(self, wspath):
        """Persistify the workspace at new path
        
        *wspath* cannot refer to existing workspace. To save an alg that has 
        already been persistified to a workspace, use :func:`save`.
        
        :param wspath: workspace path
        :type wspath: str
        """
        # validate wspath 
        if not wspath:
            log().error("Cannot create algorithm without workspace")
            return False        
        wspath = wspath.strip().rstrip("/") # strip whitespace and trailing '/'
        if not wspath.endswith(".alg"):
            log().error("Alg workspace must have '.alg' ext: {}".format(os.path.relpath(wspath)))
            return False
        if os.path.exists(wspath): 
            log().error("Alg workspace {} already exists, cannot overwrite".format(os.path.relpath(wspath)))
            return False
        self.wspath = os.path.abspath(wspath)

        # create workspace
        log().info("Creating alg workspace {}".format(os.path.relpath(wspath)))
        mkdir_p(wspath)

        # persistify to json
        return self.save()
            
    #__________________________________________________________________________=buf=
    def save(self):
        """Persistify the workspace. 
        
        Can only be called after the algorithm has already been persistified with 
        :func:`saveas`. 
        """
        if not self.wspath: 
            log().error("Cannot save non-persistified workspace (try saveas)")
            return False
        
        # set persistified
        self.info["persistified"] = True
        
        # prepare attributes
        (config, samples, info) = self.__get_attr_groups__()                
        d = dict()
        d["name"] = self.name
        d["class"] = self.__class__.__name__
        d["config"] = config
        #d["samples"] = {k:v.serialize() for (k,v) in samples.items()}
        d["samples"] = samples
        d["info"] = info        
        
        # persistify to json
        jconfig = os.path.join(self.wspath, self.fname_json)
        with open(jconfig, "w") as f:
            json.dump(d, f, indent=4, sort_keys=True, cls=LokiEncoder)     
   
        
        return True

    #__________________________________________________________________________=buf=
    def kwargs(self):
        """Return constructor arguments"""
        return {k:getattr(self,k) for k in self.__stream_attrs__}

    #__________________________________________________________________________=buf=
    def clone(self, **kw):
        """Return a clone of the algorithm with 'info' reset
        
        :param kw: override key-word args passed to alg constructor 
        """
        oldkw = self.kwargs()
        oldkw.pop("wspath", None)
        oldkw.pop("info", None)
        oldkw.update(kw)
        return self.__class__(**oldkw)

    #__________________________________________________________________________=buf=
    def print_config(self):
        """Print Alg Configruation Summary"""

        # get attributes
        (config, samples, info) = self.__get_attr_groups__()
        name = self.name
        wspath = self.wspath
        
        log().info("")
        log().info("=====>>>>>> Loki MVA Algorithm Configuration <<<<<<=====")
        
        # header
        log().info("Algorithm :")
        log().info("  Name    : {}".format(name))
        log().info("  Class   : {}".format(self.__class__.__name__))
        log().info("  File    : {}".format(wspath))
        log().info("")
        
        # configuration
        log().info("Configuration : ")
        self.__print_dict__(config)

        # samples
        log().info("Samples: ")
        self.__print_dict__({k:v.serialize() for (k,v) in samples.items()}, ignored=["sel"])
            
        # info
        log().info("Info: ")
        self.__print_dict__(info)

        # working dir
        log().info("Working dir: {}".format(os.getcwd()))
        log().info("")


    # 'Virtual' interfaces
    #__________________________________________________________________________=buf=
    def __subclass_train__(self):
        """Sub-class specific trainig implementation (overwrite in subclass)
        
        This method is not directly called by the user. The user calls
        :func:`train`, which does some basic pre- and post-training operations. 
        
        IMPORTANT: one of the pre-training operations is to change to a temporary
        working directory, to avoid unwanted interaction between output of 
        concurrently trained algorithms. Because of this, it is important to 
        use the helper functions :func:`__get_abspath_worker__` and 
        :func:`__get_sample_worker__` in the subclass implementation of 
        :func:`__subclass_train__`, which will convert config file paths
        and samples to be given w.r.t. the submit directory (where train 
        was called). 
        
        """
        log().warn("Train not implemented for {}".format(self.__class__.__name__))
        return True

    #__________________________________________________________________________=buf=
    def __subclass_predict__(self, s, **kw):
        """Sub-class specific prediction implementation (overwrite in subclass)"""
        log().warn("Predict not implemented for {}".format(self.__class__.__name__))
        return []

    #__________________________________________________________________________=buf=
    def __predict_single__(self, invals):
        """Predict classifier output based on input values *invals* (overwrite in subclass)
        
        This is the simplest way to implement a predict method. The method should 
        predict the output value(s) of the algorithm for one event, based on a set 
        of input values (*invals*). After overriding this method in your sub-class, 
        call :func:`__predict_python_loop__` in your :func:`__subclass_predict__`
        implementation.
        """
        log().warn("Predict single not implemented for {}".format(self.__class__.__name__))
        return 0.0

    # Concrete implementation (helper functions)
    #__________________________________________________________________________=buf=
    def __print_dict__(self, d, ignored=None):
        """Print content of a dictionary in a commonly formatted way"""
        if ignored is None: ignored = []
        if d:  
            for (k,v) in d.items():
                if isinstance(v,dict): 
                    log().info("  {} :".format(k))
                    for (k2,v2) in v.items():
                        if v2 is not None and k2 in ignored:
                            log().warn("    {0:20s} : {1:20s} --> NOT SUPPORTED, ignoring!".format(k2, str(v2)))
                        else:
                            log().info("    {0:20s} : {1:20s}".format(k2,str(v2)))
                elif isinstance(v,list): 
                    log().info("  {} :".format(k))
                    for v2 in v: 
                        log().info("    {0:20s}".format(str(v2)))                        
                else:  
                    log().info("  {0:22s} : {1:20s}".format(k,str(v)))
        else: 
            log().info("  none set")
        log().info("")    
            
    #__________________________________________________________________________=buf=
    def __start_train__(self):
        """Generic training initialization 
        
        Can be overwritten in subclass if necessary, but note, most sub-class 
        specific initialization occurs within self.__subclass_train__ itself. 
        """
        if self.info.get("trained", False): 
            log().info("Retraining previously trained algorithm!")
        self._tstart = time.time()
        self.tmpdir = self.__tmp_wkdir__()
        self.print_config()
        self.__pretrain_checks__()
        
    #__________________________________________________________________________=buf=
    def __end_train__(self):
        """Generic training finalization 
        
        Can be overwritten in subclass if necessary, but note, most sub-class
        specific finalization occurs within self.__subclass_train__ itself.
        """
        tend = time.time()
        self.info["trained"] = True
        self.info["train_time"] = tend - self._tstart
        os.chdir(self.submitdir)
            
    #__________________________________________________________________________=buf=
    def __predict_python_loop__(self, inputs):
        """Generic predict using :func:`__predict_single__` in python loop"""
        log().info("Calculating var with python-based loop")
        # print(inputs[0][1])
        entries = len(inputs[0][1])
        print("Entries: {}".format(entries))
        output = array(self.valtype, [])

        # build progress bar
        prog = None
        # if log().level >= logging.INFO:
        prog = ProgressBar(ntotal=entries,text=self.name) 
        
        # process
        step = int(entries / 100)
        for i in xrange(entries):
            if prog and i%step==0: prog.update(i)
            output.append(self.__predict_single__([a[1][i] for a in inputs]))
        if prog: prog.update(entries)
        
        # finalize
        if prog: prog.finalize()
        # print("length of array: {}".format(len(entries)))
        print(output[:10])
        return output

    #__________________________________________________________________________=buf=
    def __get_attr_groups__(self):
        """Return (config, samples, info) attributes"""
        # get attributes
        config = {k:v for (k,v) in self.__dict__.items() if k in self.__stream_attrs__}
        config.pop("name", None)
        config.pop("wspath", None)
        info = config.pop("info", None)
        samples = {k:v for (k,v) in config.items() if isinstance(v,Sample) }
        for v in samples: config.pop(v)
        return (config, samples, info)

    #__________________________________________________________________________=buf=
    def __get_abspath_worker__(self, path):
        """Return absolute path initially given wrt submitdir 
        
        Eg. on worker thread in tmp dir.
        """
        if os.path.isabs(path): return path
        return os.path.abspath(os.path.join(self.submitdir, path))

    #__________________________________________________________________________=buf=
    def __get_sample_worker__(self, s):
        """Return copy of sample with 'files' corrected to absolute path,  
        initially given wrt submitdir.
         
        Eg. on worker thread in tmp dir. 
        """
        if not s: return s
        snew = copy(s)
        snew.files =  [self.__get_abspath_worker__(f) for f in s.files]
        return snew

    #__________________________________________________________________________=buf=
    def __get_fmodel_path__(self):
        """Return path to the model file"""
        if not self.wspath: return None
        if not self.info.get("fmodel", None): return None
        return os.path.join(self.wspath,self.info["fmodel"])

    #__________________________________________________________________________=buf=
    def __tmp_wkdir__(self):
        """Create and change to temporary working dir and return its abspath"""
        temp_path = tempfile.mkdtemp(prefix='loki_') 
        os.chdir(temp_path)
        return temp_path

    #__________________________________________________________________________=buf=
    def __prepare_auxdir__(self):
        """Create aux dir"""
        auxdir = os.path.join(self.wspath, self.dname_aux)
        if not os.path.exists(auxdir):
            mkdir_p(auxdir)
        return auxdir

    #__________________________________________________________________________=buf=
    def __finalize_training_outputs__(self, fmodel_old, aux_files):
        """Move temp training outputs to final paths"""
        # model
        if fmodel_old:    
            if not os.path.exists(fmodel_old):
                log().warn("Failure writing model, output not found: {0}".format(fmodel_old))
            else:
                (_, fext) = os.path.splitext(fmodel_old)        
                fmodel_new = "model" + fext if fext else "model"
                fmodel_new_path = os.path.join(self.wspath, fmodel_new)
                if os.path.exists(fmodel_new_path): 
                    log().info("Removing existing model file: {}".format(fmodel_new_path))
                    os.remove(fmodel_new_path)
                log().info("Writing model: {0}".format(fmodel_new_path))
                shutil.move(fmodel_old, fmodel_new_path)
                self.info["fmodel"] = fmodel_new
    
        # aux files
        if aux_files:
            auxdir = self.__prepare_auxdir__()
            aux_files_new = []
            for faux_old in aux_files:
                if not os.path.exists(faux_old):
                    log().warn("Failure writing aux output, file not found: {0}".format(faux_old))
                else:
                    faux_new_abs = os.path.join(auxdir, os.path.basename(faux_old))
                    faux_new_rel = os.path.relpath(faux_new_abs, self.wspath)                    
                    if os.path.exists(faux_new_abs): 
                        log().info("Removing existing aux file: {}".format(faux_new_abs))
                        os.remove(faux_new_abs)
                    log().info("Writing aux output: {0}".format(faux_new_abs))
                    shutil.move(faux_old, auxdir)
                    aux_files_new.append(faux_new_rel)
            self.info["aux_files"] = aux_files_new

    # __________________________________________________________________________=buf=
    def __pretrain_checks__(self):
        """Perform some pre-train checks"""

        # get attributes
        (config, samples, info) = self.__get_attr_groups__()
        for s in samples.values():
            if s.sel: log().warn("AlgBase doesn't support sample selection: {}, ignoring.".format(s.sel.get_name()))

    # __________________________________________________________________________=buf=
    def __check_sample__(self, s, quiet=False):
        """Return True if sample suitable for use with AlgBase

        :param s: sample
        :type s: :class:`~loki.core.sample.Sample`
        """
        if len(s.files) != 1:
            if not quiet:
                log().error("Sample {} must have single file for use with AlgBase (has {})" \
                            .format(s.name,len(s.files)))
            return False
        if not os.path.exists(s.files[0]):
            if not quiet: log().error("File {} doesn't exist".format(s.files[0]))
            return False
        t = s.get_tree()
        if not t:
            if not quiet: log().error("Failed to get tree {} from file {}" \
                                      .format(s.treename, s.files[0]))
            return False
        return True


def new_hist(name, xbins, ybins = None, zbins = None, 
             xtitle = None, ytitle = None, ztitle = None):
    """Return an empty 1D/2D/3D histogram 

    Sumw2 is always set.

    Note: have removed rebin functionality. If desired may bring back in future using
    h.SetBit(ROOT.TH1.kCanRebin)

    Used to read:
    "If the binning isn't set, will use the `kCanRebin` option
    to allow on-fly resizing of the histogram as events are 
    added. Useful for figuring out appropriate ranges for new
    variables."

    :param name: name of histogram
    :type name: str
    :param xbins: x-axis bin edges
    :type xbins: list float
    :param ybins: y-axis bin edges
    :type ybins: list float
    :param zbins: z-axis bin edges
    :type zbins: list float
    :param xtitle: x-axis title 
    :type xtitle: str
    :param ytitle: y-axis title 
    :type ytitle: str
    :param ztitle: z-axis title 
    :type ztitle: str
    :rtype: :class:`ROOT.TH1`
    """
    # 3D hist
    if zbins:
        if not ybins: 
            log().error("Must provide ybins with zbins")
            raise ValueError             
        h = ROOT.TH3F(name, name,
                len(xbins) - 1, array('f', xbins),
                len(ybins) - 1, array('f', ybins),
                len(zbins) - 1, array('f', zbins),
                )        
    # 2D hist
    elif ybins:
        h = ROOT.TH2F(name, name,
                len(xbins) - 1, array('f', xbins),
                len(ybins) - 1, array('f', ybins),
                )        
    # 1D hist
    else:
        h = ROOT.TH1F(name, name,
                len(xbins) - 1, array('f', xbins)
                )        
    
    # titles
    if xtitle: h.GetXaxis().SetTitle(xtitle)
    if ytitle: h.GetYaxis().SetTitle(ytitle)
    if ztitle: h.GetZaxis().SetTitle(ztitle)
    
    # weighted histogram
    h.Sumw2()
    return h


class OutputFileStream():
    """Class to write ROOT objects to file
    
    Currently works with:  
    
    * :class:`loki.core.hist.RootDrawable`
    * :class:`loki.core.plot.Plot`
    
    All sub-drawable objects will also be written.
     
    :param fname: output file name
    :type fname: str
    """
    #____________________________________________________________
    def __init__(self,fname):
        try: 
            self.f = TFile.Open(fname,"RECREATE")
        except:
            log().error("Failed to open output stream %s"%(fname))
            raise

    #____________________________________________________________
    def __del__(self):
        self.f.Close()
            
    #____________________________________________________________
    def write(self, drawables, path=None):
        """Write drawables to file
        
        :param drawables: list of drawable objects 
        :type drawables: list :class:`loki.core.hist.RootDrawable`
        :param path: file sub-directory in which to store drawables 
        :type path: str        
        """
        # require drawables
        if drawables is None: 
            log().warn("'None' passed to OutputFileStream.write(), skipping")
            return
        
        # convert drawables to list (if single)
        if not isinstance(drawables,list): drawables = [drawables]
        
        # create sub-directory
        f = self.f
        if path is not None:
            path = path.strip("/")
            if not f.GetDirectory(path): 
                f.mkdir(path)
            f = f.GetDirectory(path)
            
        # write drawables to file
        for d in drawables: 
            d.write(f)


class Processor():
    """Class to process :class:`RootDrawable` objects

    Component histograms for the :class:`RootDrawable` objects 
    are filled using the cpp compiled LokiSelector class. 
    Higher-level drawable objects are constructed from the 
    component histograms. 

    The number of cores (*ncores*) to use when processing plots 
    can be specified. If not specified, all available cores 
    will be used. You can also provide a negative number, 
    in which case all but `|n|` cores are used. 

    By default, histogram caching is enabled. A unique hash 
    is generated for each histogram based on the x,y,z 
    variable draw expressions, the x,y,z binning, the 
    selection and weight expressions, and the event fraction. 
    An individual cache file is created for each input 
    file. The cache files are located under ``~/.lokicache``  
    

    :param event_frac: event fraction to process
    :type event_frac: float
    :param ncores: number of cores to use
    :type ncores: bool
    :param noweight: Disable sample weighting
    :type noweight: bool
    :param usecache: use histogram caching
    :type usecache: bool
        
    """
    #__________________________________________________________________________=buf=
    def __init__(self,
                 event_frac=None,
                 ncores=None,
                 noweight=False,
                 usecache=True,
                 ):        
        # config
        self.event_frac = self.__discritize_event_frac__(event_frac)
        self.ncores = ncores
        self.noweight = noweight
        self.usecache = usecache

        # members
        self.hists = []
        self.drawables = []
        self.processed_drawables = []
        self.jobs = {}

    #__________________________________________________________________________=buf=
    def register(self,drawables):
        """Register :class:`RootDrawable` or :class:`loki.core.plot.Plot` 
        
        :param drawables: single or list of ROOT drawables or plots (collection of drawables)
        :param drawables: single or list of :class:`RootDrawable` derivative or :class:`loki.core.plot.Plot`
        """
        if not isinstance(drawables, list):
            drawables = [drawables]
        self.drawables += drawables
        #self.hists+=drawable.get_component_hists()

    #__________________________________________________________________________=buf=
    def process(self,drawables=None):
        """Construct all registered objects
        
        If *drawables* provided, they will be registered prior to processing
        
        :param drawables: single or list of ROOT drawables or plots (collection of drawables)
        :param drawables: single or list of :class:`RootDrawable` derivative or :class:`loki.core.plot.Plot`
        """
        if drawables is not None: 
            self.register(drawables)
        self.__process__()
        
    #__________________________________________________________________________=buf=
    def draw_plots(self,plots=None):
        """Draw all registered plots
         
        If *plots* provided, they will be registered prior to processing 

        :param plots: single or list of plots
        :param plots: single or list of :class:`loki.core.plot.Plot`        
        """
        if plots is not None: 
            self.register(plots)
        self.__process__()
        for p in self.processed_drawables:
            if not isinstance(p, Plot): continue
            p.draw()
        

    #__________________________________________________________________________=buf=
    def write(self,f):
        """Write all processed drawables to file 
        
        :param f: output file
        :type f: :class:`ROOT.TFile`
        """
        for rd in self.processed_drawables:
            rd.write(f)
           
    #__________________________________________________________________________=buf=
    def __process__(self):
        """Process all registered drawable objects
        
        Workflow: 
        
        * Create hist configs for components of drawables (separate configs 
          are made for each input file)
        * Look for cached versions of hists
        * Collect uncached hists in selector configs (one selector per input file)
        * Process selectors using pool of threads, sequentially writing processed hists to cache. 
        * Merge hists for each subsample and scale. 
        * Construct higher level objects in the RootDrawables from component hists
              
        """
        log().info("Hist processor in da haus!")
        if not self.drawables: 
            log().info("Nothing to process")
            return

        # organise RootDrawable component hists into selector jobs 
        # selectors = self.__get_selectors__()
        
        # process selector jobs using pool of worker threadsfreturn self.xvar.new_hist(yvar=self.yvar, zvar=self.zvar, name = name)
        # self.__process_selectors__(selectors)

        ## construct drawables from component hists and clean up
        self.__finalize_outputs__()

    #__________________________________________________________________________=buf=
    def __discritize_event_frac__(self, event_frac):
        """Discritize event frac (necessary for creating hist hashes)"""
        if event_frac is None: return event_frac 
        new_event_frac = max(float("{:0.3f}".format(event_frac)), 0.001)
        if event_frac != new_event_frac: 
            log().warn("Discretizing event fraction: {} -> {}".format(event_frac,new_event_frac))
        return new_event_frac

    #__________________________________________________________________________=buf=
    def __get_selectors__(self):
        """Configure RootDrawable components into selectors
        
        Work flow is: 
        
        * Create hist configs for components of drawables (separate configs 
          are made for each input file)
        * Look for cached versions of hists
        * Collect uncached hists in selector configs (one selector per input file)
        
        The histograms are grouped into selectors based on their input file 
        (a separate selector job is made for each individual input file)
        and their multivalued container (a selector job can only handle 
        variables from up to one multi-valued container, eg. TauJets).
        
        """
        event_frac = self.__discritize_event_frac__(self.event_frac)
        log().info("Preparing jobs...")
        log().info("Using {:.1f}% of available events".format(event_frac*100. if event_frac else 100.))
        
        # create tmp working path
        tmpdir = tempfile.mkdtemp(prefix='loki_')
        log().info("Created tmp work dir: {}".format(tmpdir))
        
        # group hists into selector jobs based on mvcont and input file
        selector_dict = dict()
        file_dict = dict()
        for rd in self.drawables: 
            for h in rd.get_component_hists():
                # store all input components for hist in dict
                h.components = dict() 
                for s in h.sample.get_final_daughters():
                    if not s.files: continue
                    log().debug("sample files: {}".format(str(s.files)))
                    h.components[s] = []
                    
                    # combine selection from hist and sample
                    sel = default_cut()
                    if h.sel: sel = sel & h.sel
                    if s.sel: sel = sel & s.sel
            
                    # combine weight from hist and sample
                    weight = default_weight()
                    if not self.noweight: 
                        if h.weight: weight = weight * h.weight
                        if s.weight: weight = weight * s.weight

                    # determine multi-valued container group
                    # --------------------------------------
                    # this is important since only histograms 
                    # from the same mutli-valued container group 
                    # can be grouped together in a single selector
                    invars = [v.var for v in [h.xvar, h.yvar, h.zvar] if v]
                    invars += [v for v in [sel, weight] if v]
                    mvconts = set([c for v in invars for c in v.get_mvinconts()])
                    if len(mvconts) >= 2: 
                        log().warn("Hist {} has multiple multi-valued containers, skipping".format(h.name))
                        continue
                    elif len(mvconts) == 1: mvcont = list(mvconts)[0]
                    else:                   mvcont = None
                    if not mvcont in selector_dict: 
                        selector_dict[mvcont] = dict() 

                    # create a component hist for each input file
                    # and add to corresponding selector
                    for f in s.files:
                        # get and cache input file hash and tree
                        if f not in file_dict:
                            fhash = file_hash(f)
                            tree  = s.get_tree(f)
                            file_dict[f] = {"hash":fhash, "tree":tree}
                        else: 
                            fhash = file_dict[f]["hash"]
                            tree = file_dict[f]["tree"]
                            
                        # get and cache selector for this file and mvcont
                        if f not in selector_dict[mvcont]:
                            # temp output file for selector
                            tmpfile = os.path.join(tmpdir, next(tempfile._get_candidate_names()))
                            log().debug("creating output file: {}".format(tmpfile))
                            # number of events to process for selector
                            n = s.get_nevents(f)
                            if event_frac: n = int(event_frac*float(n))
                            # cache path for this input file  
                            fcache = os.path.join(os.getenv('HOME'), ".lokicache", "{}.root".format(fhash))
                            # now cache the selector and tree
                            scfg = SelectorCfg(fin=f,fout=tmpfile,fcache=fcache,
                                               tname=s.treename, nevents=n)
                            selector_dict[mvcont][f] = scfg 
                        else: 
                            scfg = selector_dict[mvcont][f]


                        # init vars using current tree
                        for var in [h.xvar, h.yvar, h.zvar]: 
                            if not var: continue
                            var.var.tree_init(tree)
                        for var in [sel, weight]: 
                            if not var: continue
                            var.tree_init(tree)
            
                        # generate unique hash for histogram
                        hhash = hist_hash(xvar=h.xvar, yvar=h.yvar, zvar=h.zvar, 
                                          sel=sel, wei=weight, event_frac=event_frac)
            
                        # check for cached hist
                        cached = False
                        if self.usecache and os.path.exists(scfg.fcache): 
                            fcache = ROOT.TFile.Open(scfg.fcache)
                            if fcache and fcache.Get(hhash): 
                                h.components[s] += [{"file":scfg.fcache, "hash":hhash, "cached":True}]
                                cached = True                            
                            fcache.Close()
                        
                        # prepare job if not cached           
                        if not cached: 
                            # hist configuration
                            zexpr = h.zvar.get_expr() if h.zvar else None
                            zbins = h.zvar.xbins if h.zvar else None
                            yexpr = h.yvar.get_expr() if h.yvar else None
                            ybins = h.yvar.xbins if h.yvar else None
                            xexpr = h.xvar.get_expr() if h.xvar else None
                            xbins = h.xvar.xbins if h.xvar else None
                            hcfg = HistCfg(hash=hhash, 
                                           xexpr=xexpr, xbins=xbins,
                                           yexpr=yexpr, ybins=ybins,
                                           zexpr=zexpr, zbins=zbins,
                                           wexpr = weight.get_expr(),
                                           sexpr = sel.get_expr(),
                                           )
                            log().debug("adding hist: {}, hash: {}".format(h.name, hhash))
                            scfg.add(hcfg)
                            h.components[s] += [{"file":scfg.fout, "hash":hhash}]

        # Remove selectors with no inputs (b/c cached versions were available)
        selectors = [scfg for sublist in selector_dict.values() 
                          for scfg in sublist.values() if scfg.hists]        
        return selectors

    #__________________________________________________________________________=buf=
    def __process_selectors__(self, selectors):
        """Process selectors using pool of worker threads"""

        # determine number of cores
        if not self.ncores:   ncores = min(2, cpu_count())
        elif self.ncores < 0: ncores = max(1, cpu_count() + self.ncores)
        else:                 ncores = min(self.ncores, cpu_count())
        
        # print job stats
        nfiles      = len(set([s.fin for s in selectors]))
        nev         = sum([s.nevents for s in selectors])
        nhist_proc  = sum([len(s.hists) for s in selectors])        
        nhist_cached= self.__get_nhist_cached__()
        nhist_total = self.__get_nhist_total__()
        nhist_dup   = nhist_total-nhist_cached-nhist_proc        
        log().info("")
        log().info("Job Summary")
        log().info("===========")
        log().info("Lighting up %d cores!!!"%(ncores))
        log().info("Total files  : %d"%(nfiles))
        log().info("Total events : %d"%(nev))
        log().info("Hist summary")
        log().info("------------")
        log().info("  process    : %d"%(nhist_proc))
        log().info("  duplicates : %d"%(nhist_dup))
        log().info("  cached     : %d"%(nhist_cached))
        log().info("  total      : %d"%(nhist_total))
        log().info("")
        
        # compile cpp classes (must be done before sending jobs)
        load_cpp_classes()
        
        # create pool and unleash the fury
        ti = time.time()
        prog = ProgressBar(ntotal=nev,text="Processing hists") if log().level >= logging.INFO else None         
        pool = Pool(processes=ncores)
        results = [pool.apply_async(process_selector, (s,)) for s in selectors]
                
        nproc=0
        nhist_tot = 0
        nhist_cached = 0
        while results: 
            for r in results:
                if r.ready():
                    scfg = r.get()
                    nproc+=scfg.nevents
                    if self.usecache:
                        (ntot,ncache) = self.__cache_selector__(scfg)
                        nhist_tot += ntot
                        nhist_cached += ncache
                    results.remove(r)                    
            time.sleep(1)
            if prog: prog.update(nproc)
        if prog: prog.finalize()
        tf = time.time()
        dt = tf-ti
        log().info("Hist processing time: %.1f s"%(dt))
        if self.usecache: 
            log().info("Cached {} / {} hists!".format(nhist_cached,nhist_tot))

    #__________________________________________________________________________=buf=
    def __cache_selector__(self, scfg):
        """Save tmp hists from selector into permanent cache files""" 
        fin_name = scfg.fout 
        fout_name = scfg.fcache
        # ensure cache dir exists
        fout_dir = os.path.dirname(fout_name)
        mkdir_p(fout_dir)
        # get lock on cache file
        fout_base = os.path.basename(fout_name)
        fout_lock = os.path.join(fout_dir, ".{}.lock".format(fout_base))
        lock = FileLock(fout_lock)
        log().debug("Saving histograms to cache file: {}".format(fout_name))
        nhist_cached = 0        
        try: 
            with lock.acquire(timeout = 20):
                # make tmp copy of cache incase failure
                ftmp_name = tempfile.mktemp()
                if os.path.exists(fout_name):
                    try: 
                        shutil.copy(fout_name, ftmp_name)
                    except: 
                        log().warn("Failed copying cache to tmp location, {} -> {}".format(fout_name, ftmp_name))
                        raise IOError
                                
                # open tmp cache
                ftmp = ROOT.TFile.Open(ftmp_name, "UPDATE")
                if not ftmp: 
                    log().warn("Failure opening cache: {}".format(ftmp_name))
                    raise IOError
                
                # open input
                fin = ROOT.TFile.Open(fin_name)
                if not fin: 
                    log().warn("Failure opening tmp: {}".format(fin_name))
                    raise IOError
                
                # copy hists from input to tmp
                for hhash in scfg.hists:
                    h = fin.Get(hhash)
                    if not h: 
                        log().warn("Couldn't get {} from {}".format(hhash,fin_name))
                        continue
                    ftmp.WriteTObject(h)
                    nhist_cached+=1
                
                # close, move back and cleanup
                fin.Close()
                ftmp.Close()
                if os.path.exists(fout_name):
                    os.remove(fout_name)
                shutil.copy(ftmp_name, fout_name)
                os.remove(ftmp_name)
        except TimeoutError: 
            log().warn("Couldn't get lock on cache: {}".format(fout_name))
        except: 
            log().warn("Couldn't write hists to cache: {}".format(fout_name))
        
        return (nhist_cached, len(scfg.hists))        
        


    #__________________________________________________________________________=buf=
    def __finalize_outputs__(self):
        """Merge component hists and construct higher-level RootDrawable objects"""
        event_frac = self.__discritize_event_frac__(self.event_frac)
        # group and merge all the outputs
        print("DDDDD: ", self.drawables)
        for rd in self.drawables:
            print("GGGGGGG: ", rd.get_component_hists())
            for h in rd.get_component_hists():
                print("HHHHHHHH: ", h)
                # h is a ndarray object converted to a a Hist object
                h = h.get_rootobj()
                
                rootobj = h.new_hist()
                h.set_rootobj(rootobj)
                for (s, components) in h.components.items():
                    # scale
                    if not self.noweight and s.scaler: 
                        scale = s.get_scale()
                        if event_frac: scale/=event_frac
                    else: 
                        scale = None
                                   
                    # merge sub-objects (from each file)
                    for c in components: 
                        f = ROOT.TFile.Open(c["file"])
                        o = f.Get(c["hash"]).Clone()
                        if scale: o.Scale(scale)
                        rootobj.Add(o)
                        f.Close()
            rd.build_rootobj()
            
        # accounting
        self.processed_drawables += self.drawables
        self.drawables = []

    #__________________________________________________________________________=buf=
    def __get_nhist_total__(self):
        """Return the total number of histograms needed to construct drawables"""
        n = 0
        for rd in self.drawables: 
            for h in rd.get_component_hists():
                for subcomponents in h.components.values():
                    n+=len(subcomponents)
        return n
                     
    #__________________________________________________________________________=buf=
    def __get_nhist_cached__(self):
        """Return the number of pre-cached histograms"""
        n = 0
        for rd in self.drawables: 
            for h in rd.get_component_hists():
                for subcomponents in h.components.values():
                    for sc in subcomponents: 
                        if sc.get("cached",False): n+=1
        return n

class Style():
    """Class to store and set style for ROOT hists/graphs etc.
 
    *stylekw* is an arbitrary set of named arguments intended
    to correspond to ROOT hist/graph style parameters, eg.: 
    Style(name="Signal",FillColor=ROOT.kBlack,LineStyle=4)

    They must be the same as used in the Setter/Getter functions
    for the object the style is applied to: TH1, TGraph, etc...

    For more info on passing arbitrary named parameters in python, 
    check out: http://stackoverflow.com/questions/1769403/understanding-kwargs-in-python

    :param name: simple text name for style
    :type name: str
    :param tlatex: TLatex formatted name for style
    :type tlatex: str
    :param drawopt: option to be used with Object::Draw() 
    :type drawopt: str
    :param stylekw: style options to be passed to your ROOT Object 
    :type stylekw: named argument list (arg1=val1,...,argN=valN)
    """

    #____________________________________________________________
    def __init__(self, name = None, tlatex = None, drawopt = None,
            **stylekw):
        self.name        = name
        self.tlatex      = tlatex or name
        self.drawopt     = drawopt 
        self.stylekw     = stylekw
        
        # set default options
        stylekw.setdefault("LineWidth",2)
        
        # attach options as attributes
        for key, value in stylekw.items():
            setattr(self,key,value)

    #____________________________________________________________
    def apply(self,h):
        """Apply style parameters to ROOT object

        :param h: ROOT object
        :type h: ROOT drawable object TH1, TGraph, etc...
        """
        #h.name = self.name
        #h.tlatex = self.tlatex
        #h.drawopt = self.drawopt
        for hprop,value in self.stylekw.items(): 
            if not value is None:
                try: 
                    getattr(h,'Set%s'%(hprop))(value)
                except: 
                    log().error("Invalid style option: %s"%(hprop))
                    exit(1)

    #____________________________________________________________
    def __add__(self,other):
        """``+`` operator for Style objects (rvalue preference)
        
        I.e. the object on the right-hand side of the ``+``
        operator will override any properties in common with 
        the object on the left-hand side
        """
        kwargs = dict()
        kwargs.update(self.__dict__)
        kwargs.update(other.__dict__)
        if "stylekw" in kwargs: del kwargs["stylekw"]
        return Style(**kwargs)


#____________________________________________________________
def default_style():
    """Return default :class:`Style`"""
    return Style(name="Default", LineColor=1,MarkerColor=1,MarkerStyle=20)


class RootDrawable():
    """Abstract base class for all ROOT drawable objects

    The function ``build_rootobj`` should be overridden in the 
    derived class with the specific implementation to construct 
    the ROOT drawable object. 
    
    The ROOT drawable object can be constructed from sub-drawables 
    (eg. the efficiency graph class :class:`EffProfile` is constructed 
    by drawing the pass and total hists and dividing them). 
    Sub-drawables should be declared in the constructor, eg::
     
        self.add_subrd(h_pass)
        self.add_subrd(h_total)   
    
    If *yvar* (*zvar*) is specified, will function as a 2D (3D) object. 
    The y-axis range will be set to the yvar range when drawing.
    
    :param xvar: x-axis variable view
    :type xvar: :class:`loki.core.var.View`
    :param yvar: y-axis variable view
    :type yvar: :class:`loki.core.var.View`
    :param zvar: z-axis variable view
    :type zvar: :class:`loki.core.var.View`            
    :param sty: style
    :type sty: :class:`loki.core.style.Style`
    :param name: name
    :type name: str
    :param stackable: whether it can be added to a :class:`ROOT.THStack`
    :type stackable: bool
    :param drawopt: default draw option
    :type drawopt: str
    :param noleg: don't put in legend (default: False)
    :type noleg: bool
    """
    #____________________________________________________________
    def __init__(self,xvar=None,yvar=None,zvar=None,sty=None,name=None,
                 stackable=False,drawopt=None,noleg=False):
        # config
        self.xvar = xvar
        self.yvar = yvar
        self.zvar = zvar
        self.sty = sty or default_style()
        self.name = name    
        self.stackable = stackable
        self.drawopt = drawopt
        self.noleg = noleg

        # members
        self._rootobj = None
        self._subrds = []
        self._extra_labels = []

    #____________________________________________________________
    def draw(self):
        """Draw the ROOT object on canvas with chosen options"""
        # base draw option
        drawopts = ["SAME"]
        # add draw option from style if specified
        if self.sty and self.sty.drawopt: 
            drawopts += [self.sty.drawopt]
        elif self.drawopt: 
            drawopts += [str(self.drawopt)]
        # check for _rootobj
        if not self._rootobj: 
            log().warn("%s trying to draw null rootobj! Skipping..."%(self.name))
            return None
        
        drawopt = ",".join(drawopts)
        return self._rootobj.Draw(drawopt)
 
    #____________________________________________________________
    def rootobj(self):
        """Returns the raw ROOT drawable object 
        :rtype: :class:`ROOT.TH1` or :class:`ROOT.TGraph` (or derivatives)
        """
        return self._rootobj

    #____________________________________________________________
    def set_rootobj(self,o):
        """Setter for raw ROOT drawable object
        
        :param o: root drawable object
        :type o: :class:`ROOT.TH1` or :class:`ROOT.TGraph` (or derivative)
        """
        if not o: 
            log().warn("%s trying to set null rootobj!"%(self.name))
            return
        if self.sty: self.sty.apply(o)

        self._rootobj = o

    #____________________________________________________________
    def is_valid(self):
        """Returns true if rootobj is valid (not None)"""
        return (self._rootobj is not None)

    #____________________________________________________________
    def add_subrd(self,rd):
        """Declare sub drawable object
        
        :param rd: sub drawable object
        :type rd: :class:`RootDrawable`
        """
        self._subrds.append(rd)

    #____________________________________________________________
    def write(self,f):
        """Write ROOT objects to file
        
        :param f: file
        :type f: :class:`ROOT.TFile`
        """
        if not f.Get(self._rootobj.GetName()): 
            f.WriteTObject(self._rootobj)
        for o in self._subrds: 
            o.write(f)

    #____________________________________________________________
    def get_component_hists(self):
        """Return list of component hists
        
        Recursively calls through sub :class:`RootDrawable` objects yielding 
        objects of derived type :class:`Hist`
        
        :rtype: list :class:`Hist`
        """
        hists = []
        for rd in self._subrds:
            hists += rd.get_component_hists()
        return hists
    
    #____________________________________________________________
    def build_rootobj(self):
        """Build the ROOT object to be drawn
        
        Should be overridden in derived class with specific implementation
        to construct drawable object
        """
        pass

    #____________________________________________________________
    def get_xtitle(self):
        """Returns the x-axis title
        
        Default is to use x-title from x-variable.
        Can be overridden for specific implementations (eg. ''Efficiency'')

        :rtype: str
        """
        return self.xvar.get_xtitle()

    #____________________________________________________________
    def get_ytitle(self):
        """Returns the y-axis title
        
        Default is to use y-title from x-variable (ie Events / bin width)
        Can be overridden for specific implementations (eg. ''Efficiency'')

        :rtype: str
        """
        return self.xvar.get_ytitle()

    #____________________________________________________________
    def get_dimension(self):
        """Returns the dimension of the object
        
        :rtype: int
        """
        if self.yvar is not None: return 2
        return 1


class Hist(RootDrawable):
    """Class for 1D/2D/3D histograms
    
    :param sample: input event sample
    :type sample: :class:`loki.core.sample.Sample`
    :param xvar: x-axis varaible view
    :type xvar: :class:`loki.core.var.View`
    :param yvar: y-axis varaible view (for 2D hists)
    :type yvar: :class:`loki.core.var.View`
    :param zvar: z-axis varaible view (for 3D hists)
    :type zvar: :class:`loki.core.var.View`    
    :param sel: selection
    :type sel: :class:`loki.core.var.VarBase`
    :param weight: weight expression
    :type weight: :class:`loki.core.var.VarBase`
    :param normalize: normalize the histogram integral to unity
    :type normalize: bool
    :param sty: style
    :type sty: :class:`loki.core.style.Style`
    :param kwargs: key-word arguments passed to :class:`RootDrawable`
    :type kwargs: key-word arguments 
    """
    #____________________________________________________________
    def __init__(self,xvar=None,yvar=None,zvar=None, 
                 sel=None,weight=None,normalize=False,sty=None,
                 xmax=None,ymax=None,
                 **kwargs):
        RootDrawable.__init__(self,xvar=xvar,yvar=yvar,zvar=zvar,
                              sty=sty,
                              stackable=True,
                            #   drawopt = "COL" if yvar else "",
                              drawopt = "COL",
                              **kwargs)
        # config
        self.sel = sel
        self.weight = weight
        self.normalize = normalize
    
        # if (yvar and not xvar) or (zvar and not (xvar and yvar)): 
        #     log().warn("Malformed hist: {}".format(self.name))


            
    
    #____________________________________________________________
    def new_hist(self,name=None):
        """Return empty TH1F/TH2F/TH3F for xvar/yvar/zvar"""
        if name is None: name = self.name
        return self.xvar.new_hist(yvar=self.yvar, zvar=self.zvar, name = name)

    #____________________________________________________________
    def histargs(self,name=None):
        """Returns list of arguments for :func:`loki.core.histutils.new_hist`"""
        return histargs(self.xvar,self.yvar,self.zvar,name=name)
     
    #____________________________________________________________
    def get_component_hists(self):
        """Returns itself in list"""
        return [self]

    #____________________________________________________________
    def build_rootobj(self):
        """Postprocess the histogram object"""
        if self.normalize: 
            normalize(self._rootobj)

    #____________________________________________________________
    def get_ytitle(self):
        """Returns y-axis title for hist"""
        if self.yvar is not None: 
            return "%s"%(self.yvar.get_xtitle())
        return self.xvar.get_ytitle()


def frange(x1,x2,x):
    """Returns values from *x1* to *x2* separated by *x*
    
    It's like range(x1,x2,x) but can take decimal x.
    """
    while x1 < x2:
        yield x1
        x1+=x


class WorkingPointExtractor(RootDrawable):
    """Extracts fixed efficiency working point parameterisations 
    
    The cuts to apply to the discriminant (*disc*) are extracted 
    as a 1D or 2D parameterization against the dependent var(s)
    *xvar* (and *yvar*). For each target efficiency in *target_effs*
    a graph and a histogram parameterization are saved to the 
    output file.
    
    A preselection (*sel*) can be specified. If a different 
    selection is required in the deominator used to define the
    efficiency (eg. if the efficiency should be calculated 
    w.r.t. truth taus), the corresponding selection (*sel_tot*) 
    and dependent vars (*xvar_tot*, *yvar_tot*) should be provided.  
    
    If you want to extend a 1D parameterization to a 2D parameterization
    using dummy values for the second variable, they extremes in the 
    second variable can be provided by *yvals2d*. 
    
    If the cut should be applied in reverse (ie. cut < disc), set 
    *reverse=True*.  
    
    
    Details below taken from Nico Madysa's original implementation: 
    
    
    Multivariate analysis (MVA) methods are used for classification of
    particles.
    They distinguish between charged tau leptons (signal or mc) and
    QCD jets (background or data).
    
    This distinction or, preferrably called, "classification" is done via
    a score.
    Each particle is given a score, rating how tau-like it is.
    The goal is to give high scores to tau leptons and low scores to
    QCD jets.
    
    In order to transition from continuous scores to discrete classes, we
    decide on a certain "score cut". The MVA considers particles with
    a score higher than this cut to be signal taus, and particles with
    a score lower than this cut to be background (or fake taus).
    
    Since we want to be able to adjust the signal efficiency
    (identified taus divided by all true taus), we have to calculate the
    correct score cut for each "working point".
    
    Our observation is that particles with a high transverse momentum pT are
    more likely to have a high score than those with low pT.
    For various reasons, this is undesirable.
    
    Thus, the score cut is a function not only of the signal efficiency,
    but also of pT.
    And we choose this score cut in such a manner, that, when applying it
    for a working point, the curve signal_efficiency(truth_pt) will be
    as flat as possible.
    
    
    :param target_effs: list of target efficiencies
    :type target_eff: list float
    :param disc: discriminant variable view
    :type disc: :class:`~loki.core.var.View`
    :param xvar: first dependent variable view
    :type xvar: :class:`~loki.core.var.View`
    :param yvar: second dependent variable view
    :type yvar: :class:`~loki.core.var.View`
    :param sel: selection
    :type sel: :class:`~loki.core.var.VarBase`
    :param xvar_tot: denominator first dependent variable view
    :type xvar_tot: :class:`~loki.core.var.View`
    :param yvar_tot: denominator second dependent variable view
    :type yvar_tot: :class:`~loki.core.var.View`
    :param sel_tot: denominator selection
    :type sel_tot: :class:`~loki.core.var.VarBase`
    :param tag: tag to be included in output graph/histogram names
    :type tag: str
    :param yvals2d: list of dummy values to extend 1D parameterization to 2D
    :type yvals2d: list float
    :param reverse: apply cut in reverse (disc < cut)
    :type reverse: bool
    :param smooth1D: smoothing option when using 1 aux var (not currently implemented) 
    :type smooth: str    
    :param smooth2D: smoothing option when using 2 aux vars (eg. k3a, k5a, k5b ) 
    :type smooth: str

    """
    #__________________________________________________________________________=buf=
    def __init__(self, target_effs=None, disc=None, xvar=None, yvar=None,  
                 sel=None, xvar_tot=None, yvar_tot=None, sel_tot=None, tag=None,
                 yvals2d=None, reverse=False, smooth1D=None, smooth2D=None):
        RootDrawable.__init__(self)
        # defaults
        if target_effs is None: target_effs = frange(0.00,1.00,0.01)
        # config
        self.target_effs = target_effs
        self.disc = disc
        self.xvar = xvar
        self.yvar = yvar
        self.tag = tag
        self.yvals2d = yvals2d
        self.reverse = reverse
        self.smooth1D = smooth1D
        self.smooth2D = smooth2D

        allowed_smooth1D = [None]
        assert smooth1D in allowed_smooth1D, "smooth1D must be one of: {0}".format(str(allowed_smooth1D))   
        
        allowed_smooth2D = [None, "k3a", "k5a", "k5b"]
        assert smooth2D in allowed_smooth2D, "smooth2D must be one of: {0}".format(str(allowed_smooth2D))   
        
        # hists
        hname = "h_wpextractor"
        if tag: hname += "_%s"%(tag)        
        # if yvar: # 2D dependence
        self.hist = Hist(xvar=xvar,yvar=yvar,zvar=disc,sel=sel,name=hname)
        # else:    # 1D dependence
            # self.hist = Hist(xvar=xvar,yvar=disc,sel=sel,name=hname)
        self.add_subrd(self.hist) 
        self.hist_tot = None
        if xvar_tot or yvar_tot or sel_tot:
            if yvar: 
                if None in [xvar_tot, yvar_tot, sel_tot]: 
                    log().warn("Must provide complete set of (xvar_tot, yvar_tot, sel_tot) to WorkingPointExtractor")
                else: 
                    self.hist_tot = Hist(xvar=xvar_tot,yvar=yvar_tot,sel=sel_tot,name=hname+"_tot")
                    self.add_subrd(self.hist_tot)
            else: 
                if None in [xvar_tot, sel_tot]: 
                    log().warn("Must provide complete set of (xvar_tot, sel_tot) to WorkingPointExtractor")
                else: 
                    self.hist_tot = Hist(xvar=xvar_tot,sel=sel_tot,name=hname+"_tot")
                    self.add_subrd(self.hist_tot)
        
        # memebers
        self._rootobjs = []

    #__________________________________________________________________________=buf=
    def build_rootobj(self):
        """Build the working point graphs"""
        
        ## input hists
        h = self.hist.rootobj()
        h_tot = None
        if self.hist_tot: h_tot = self.hist_tot.rootobj()

        ## create working point graphs/hists
        for te in self.target_effs:
            #g_cuts = self.__get_cut_graph__(h, te, h_total=h_tot)
            #h_cuts = self.__convert_cut_graph_to_hist__(g_cuts)
            h_cuts = self.__get_cut_hist__(h, te, h_total=h_tot)
            h_cuts.SetDirectory(0)
            self.__smooth_hist__(h_cuts)
            g_cuts = self.__convert_cut_hist_to_graph__(h_cuts)
            self._rootobjs.append(h_cuts)
            self._rootobjs.append(g_cuts)
            # create dummy 2D extension
            if not self.yvar and self.yvals2d is not None:
                hname2 = "h2"+str(g_cuts.GetName())[1:] 
                ybin_edges = []
                for i in range(len(self.yvals2d)-1):
                    y1 = self.yvals2d[i]
                    y2 = self.yvals2d[i+1] 
                    ybin_edges.append(y1 - (y2-y1)/2)
                ybin_edges.append(y1 + (y2-y1)/2)
                h2_cuts = convert_hist_to_2dhist(h_cuts,ybin_edges,name=hname2)
                h2_cuts.SetDirectory(0)
                self._rootobjs.append(h2_cuts)
                
                g2_cuts = self.__convert_cut_hist_to_graph__(h2_cuts) 
                self._rootobjs.append(g2_cuts)
            

    #__________________________________________________________________________=buf=
    def write(self,f):
        """Write ROOT objects to file
        
        Override baseclass function to write multiple objects
        
        :param f: file
        :type f: :class:`ROOT.TFile`
        """
        # write config
        disc = TNamed("disc", self.disc.var.get_newbranch())
        xvar = TNamed("xvar", self.xvar.var.get_newbranch())
        f.WriteTObject(disc)
        f.WriteTObject(xvar)
        if self.yvar: 
            yvar = TNamed("yvar", self.yvar.var.get_newbranch())
            f.WriteTObject(yvar)
        rev = TParameter(bool)("reverse",self.reverse)
        f.WriteTObject(rev)
        
        # write efficiency graphs
        for ro in self._rootobjs: 
            if not f.Get(ro.GetName()): 
                f.WriteTObject(ro)
                
        # write input hists
        for o in self._subrds: 
            o.write(f)

    # Internal functions
    #__________________________________________________________________________=buf=
    def __get_cuts__(self, h, target_eff, h_total=None):
        """Returns a list of tuples (*x*, *cut*)
        
        The tuples represent the bin centers in the dependent variable, *x*, 
        and the cut values, *cut*, required to reach the target efficiency, 
        *target_eff*.  
        
        If the efficiency should be calculated w.r.t. a different selection 
        than is applied in *h* (eg. if you want to calculate the efficiency 
        w.r.t truth) *h_total* can be specified, which is a 1D histogram 
        in the dependent variable.
          
        Uses :func:`__find_score_cut__`
        
        :param h: 2D hist of discriminant vs dependent variable
        :type h: :class:`ROOT.TH2`
        :param target_eff: target efficiency
        :type target_eff: float
        :param h_total: 1D hist of dependent variable before selection
        :type h_total: :class:`ROOT.TH1`
        :rtype: list of tuple (float, float)
        """
        min_cut = h.GetYaxis().GetBinLowEdge(1)
        cuts = []
        for i_bin in range(1,h.GetNbinsX()+1): 
            h_proj = h.ProjectionY("_py", i_bin,i_bin)
            # normalise
            if h_total: 
                ntot = h_total.GetBinContent(i_bin)
                if ntot: h_proj.Scale(1./ntot)
            else: 
                normalize(h_proj)    
            cut = self.__find_score_cut__(h_proj,target_eff)
            
            ## throw error if efficiency not reached
            if cut is False:
                xmin = h.GetXaxis().GetBinLowEdge(i_bin)
                xmax = h.GetXaxis().GetBinLowEdge(i_bin+1)
                # use previous cut
                if len(cuts): 
                    cut = cuts[-1][1]
                # use mincut                    
                else: 
                    cut = min_cut
                log().warn("Failed to reach target eff: %.3f for x-bin (%f,%f), using cut: %f"%(target_eff,xmin,xmax,cut))
                #raise ValueError("Failed to reach target eff: %.3f for x-bin (%f,%f)!"%(target_eff,xmin,xmax)) 
                
            var = h.GetXaxis().GetBinCenter(i_bin)
            cuts.append([var,cut])
        return cuts

    #__________________________________________________________________________=buf=    
    def __find_score_cut__(self, h, target_eff):
        """Return discriminant score cut for target efficiency                 
        
        If target not reached 'False' is returned. 
        
        :param h: (normalized) 1D hist of discriminant
        :type h: :class:`ROOT.TH1`
        :param target_eff: target efficiency
        :type target_eff: float
        :rtype: float
        
        """
        # Code gets error-prone if the overflow bin contains enough events
        # to satisfy target efficiency.
        last_bin = h.GetNbinsX()+1
        if h.GetBinContent(last_bin) > target_eff:
            return False
            #raise ValueError("Overflow bin is too full")
        # Iterate over all bins.
        # Stop when we've met target efficiency.
        integral = 0.0
        if self.reverse: 
            for i_bin in xrange(last_bin+1):
                integral += h.GetBinContent(i_bin)
                # restrict precision in comparison to avoid floating point problems on 100%
                if float("%.3f"%(integral))>=float("%.3f"%(target_eff)): 
                    return h.GetBinLowEdge(i_bin+1)            
        else: 
            for i_bin in reversed(xrange(last_bin+1)):
                integral += h.GetBinContent(i_bin)
                # restrict precision in comparison to avoid floating point problems on 100%
                if float("%.3f"%(integral))>=float("%.3f"%(target_eff)): 
                    return h.GetBinLowEdge(i_bin)
        
        return False
        #raise ValueError("Not all target efficiencies reached")

    #__________________________________________________________________________=buf=
    def __get_cut_hist_1D__(self, h, target_eff, h_total=None):
        """Return a 1D hist parameterizing the cut values against xvar. 
        
        See :func:`__get_cuts__` for arg details
        
        :rtype: :class:`ROOT.TH1`
        """
        if self.tag: hname = "h_{tag}_{eff:02.0f}".format(tag=self.tag,eff=target_eff*100.0)
        else:        hname = "h_{eff:02.0f}".format(eff=target_eff*100.0)
        cuts = self.__get_cuts__(h,target_eff,h_total)
        hnew = h.ProjectionX(hname)
        hnew.Reset()
        for i in range(len(cuts)): 
            hnew.SetBinContent(i+1, cuts[i][1])
            hnew.SetBinError(i+1,0)
        return hnew

    #__________________________________________________________________________=buf=
    def __get_cut_hist_2D__(self, h, target_eff, h_total=None):
        """Return a 2D hist parameterizing the cut value against xvar, yvar.
        
        See :func:`__get_cuts__` for arg details
        
        :rtype: :class:`ROOT.TH2`
        """
        if self.tag: hname = "h2_{tag}_{eff:02.0f}".format(tag=self.tag,eff=target_eff*100.0)
        else:        hname = "h2_{eff:02.0f}".format(eff=target_eff*100.0)
        hnew = h.Project3D("yx")
        hnew.Reset()
        hnew.SetName(hname)
        #print "bins x: ", hnew.GetNbinsX(), ", binsy: ", hnew.GetNbinsY()
        
        # loop over y-bins and project xz
        for iy in range(1, h.GetNbinsY()+1):
            log().debug("Projecting y-bin {0:d}: [{1:f}, {2:f}]".format(iy, 
                    h.GetYaxis().GetBinLowEdge(iy), h.GetYaxis().GetBinLowEdge(iy+1)))
            h.GetYaxis().SetRange(iy,iy)
            htemp = h.Project3D("zx")
            h.GetYaxis().SetRange() #reset range
            if h_total: 
                pname = "{0}_x{1:d}".format(h_total.GetName(), iy)
                htemp_total = h_total.ProjectionX(pname, iy, iy)
            else: htemp_total = None
            cuts = self.__get_cuts__(htemp,target_eff,htemp_total)
            for ix in range(1, h.GetNbinsX()+1): 
                hnew.SetBinContent(ix, iy, cuts[ix-1][1])
            
        return hnew

    #__________________________________________________________________________=buf=
    def __get_cut_hist__(self, h, target_eff, h_total=None):
        """Return a hist parameterizing cut values against the dependent variable(s)

        See :func:`__get_cuts__` for arg details
        
        :rtype: :class:`ROOT.TH1` or :class:`ROOT.TH2`
        """        
        if self.yvar: 
            return self.__get_cut_hist_2D__(h, target_eff, h_total=h_total)
        else: 
            return self.__get_cut_hist_1D__(h, target_eff, h_total=h_total)
    
    #__________________________________________________________________________=buf=    
    def __convert_cut_graph_to_hist__(self, g):
        """Convert graph cut parameterization to hist parameterization"""
        if isinstance(g,TGraph2D): 
            h = self.xvar.new_hist(yvar=self.yvar, name = 'h' + str(g.GetName())[1:])
            for i in range(1,h.GetNbinsX()+1):
                x = h.GetXaxis().GetBinCenter(i) 
                for j in range(1,h.GetNbinsY()+1): 
                    y = h.GetYaxis().GetBinCenter(j)
                    h.SetBinContent(i,j,g.Interpolate(x,y))
        else: 
            h = self.xvar.new_hist(name = 'h' + str(g.GetName())[1:])
            for i in range(1,h.GetNbinsX()+1):
                x = h.GetXaxis().GetBinCenter(i) 
                h.SetBinContent(i,g.Eval(x))
        return h

    #__________________________________________________________________________=buf=    
    def __convert_cut_hist_to_graph__(self, h):
        """Convert graph cut parameterization to hist parameterization"""
        if isinstance(h,TH2): g = TGraph2D(h)
        else:                      g = TGraph(h)
        g.SetName('g' + str(h.GetName())[1:])
        return g

    #__________________________________________________________________________=buf=    
    def __smooth_hist__(self, h):
        """Convert graph cut parameterization to hist parameterization"""
        if self.yvar and self.smooth2D: 
            h.Smooth(1, self.smooth2D)
        elif self.smooth1D:
            log().warn("Smoothing for 1D hists not yet implemented")  


class MVScoreTuner(AlgBase):
    """Working point tuner and score flattener 
    
    The train method extracts fixed efficiency working points for the specified 
    discriminant (*disc*). The efficiency can be flattened against up to 2 
    auxiliary variables (*xvar* and *yvar*). After training, the pass/fail 
    decisions of the working points can be predicted by passing the *eff* 
    argument to the predict function, or the working points can be used to 
    predict a 'flattened discriminant' which is transformed to have a uniform 
    distribution on [0,1].
    
    Cut extraction (training): 
    
    The cuts to apply to the discriminant are extracted as a 1D or 2D 
    parameterization against the dependent var(s). For each target efficiency 
    ranging from 1-99%, a graph and a histogram parameterization are saved to 
    the model file.
        
    If you want to extend a 1D parameterization to a 2D parameterization
    using dummy values for the second variable, the extremes in the 
    second variable can be provided by *yvals2d*. 
    
    If the cut should be applied in reverse (ie. cut < disc), set 
    *reverse=True*.  
    

    Working point prediction: 

    To predict pass/fail decisions for a flattened working point, passt the *eff* 
    argument to the predict method (must be an integer in the range [1,99]). 


    Flat score prediction: 
    
    By default, the predict function returns a flattened discriminant, transformed 
    to be uniform on [0,1].

    *wspath* and *info* are not to be set by the user (see :class:`~loki.train.alg.AlgBase`)  
    
    :param name: algorithm name
    :type name: str
    :param disc: discriminant 
    :type disc: :class:`~loki.core.var.View`
    :param xvar: first dependent variable 
    :type xvar: :class:`~loki.core.var.View`
    :param yvar: second dependent variable 
    :type yvar: :class:`~loki.core.var.View`
    :param reverse: apply the mv score cut in reverse
    :type reverse: bool
    :param smooth: smoothing option, currently only valid for 2D (eg. k3a, k5a, k5b ) 
    :type smooth: str
    :param usehist: set False to use graph-based cut parameterisation (NOT recommended, as it returns 0 outside the graph boundary)  
    :type usehist: bool
    :param yvals2d: list of dummy values to extend 1D parameterization to 2D
    :type yvals2d: tuple (float, float)
    :param sig_train: signal training sample
    :type sig_train: :class:`~loki.core.sample.Sample`
    :param kw: key-word args passed to :class:`~loki.train.alg.AlgBase`    
    """
    #__________________________________________________________________________=buf=
    def __init__(self, name=None, wspath = None, info = None, 
                 sig_train = None, disc = None, xvar = None, yvar = None, 
                 reverse = None, smooth = None, usehist = None, yvals2d = None):    
        if name is None: name = "MVScoreTuner"
        AlgBase.__init__(self, name=name, wspath=wspath, info=info, valtype='f')
        
        # set defaults
        # if sig_train is None: sig_train = Sample("sig_train", weight=vars.weight, files=[])
        # if sig_train is None: sig_train = Sample("sig_train", files = ['combined_1p.root'], treename="tree")
        if usehist is None: usehist = True
        
        # process complex types (Sample done automatically)
        # print("GGGGGGG")
        # disc = get_view(disc)
        # print("GGG22222GGGG")
        # xvar = get_view(xvar)
        # if yvar: yvar = get_view(yvar)
        if reverse is None: reverse = False

        # set attributes
        # self.sig_train = sig_train
        self.disc = disc
        self.xvar = xvar
        self.yvar = yvar
        self.reverse = reverse
        self.smooth = smooth
        self.usehist = usehist
        self.yvals2d = yvals2d        
        
    # Subclass overrides
    #__________________________________________________________________________=buf=
    def __subclass_train__(self):
        """Train the classifier"""
        # config
        fname_model = "model.root"
        reverse = self.reverse
        disc = self.disc
        xvar = self.xvar
        yvar = self.yvar
        smooth1D = smooth2D = None
        if self.smooth: 
            if yvar: smooth2D = self.smooth
            else:    smooth1D = self.smooth

        # pre-train checks
        if not self.ispersistified(): 
            log().error("MVScoreTuner must be persistified before training (call 'saveas')")
            return False        
        # if not disc:
        #     log().error("No disc view configured")
        #     return False 
        # if not xvar: 
        #     log().error("No xvar view configured")
        #     return False 

        # get samples with abspath
        # s = self.__get_sample_worker__(self.sig_train)
        # print("TTTTTTT: ", s)
        # if not self.__check_sample__(s): return False

        # define working point extractor
        wpe = WorkingPointExtractor(disc=disc, xvar=xvar, yvar=yvar, 
              reverse=reverse, smooth1D=smooth1D, smooth2D=smooth2D)

        # process
        p = Processor(ncores=1)
        p.process(wpe)
        p.draw_plots()
    
        ofstream = OutputFileStream(fname_model)
        ofstream.write(wpe)
        ofstream.f.Close()
        
        ## move outputs to workspace
        self.__finalize_training_outputs__(fname_model, None)

        return True

    #__________________________________________________________________________=buf=
    def __subclass_predict__(self, s, eff=None):
        """Sub-class specific prediction implementation
                
        :param eff: working point efficiency in percentage 
        :type eff: int
        """
        fmodel = self.__get_fmodel_path__()
        disc = self.disc.var
        xvar = self.xvar.var
        yvar = self.yvar.var if self.yvar else None
        usehist = self.usehist
        
        # checks 
        # if not fmodel: 
        #     log().error("No model file, cannot predict")
        #     return None
        # if not s: 
        #     log().error("No input sample, cannot predict")
        #     return None
        # if not s.files: 
        #     log().error("No input files, cannot predict")
        #     return None
        # if not disc:
        #     log().error("No disc view configured, cannot predict")
        #     return None 
        # if not xvar: 
        #     log().error("No xvar view configured, cannot predict")
        #     return None 
                
        # retrieve graph / hist        
        f = TFile.Open(fmodel)
        if not f: 
            log().error("Error opening config file {0}.".format(fmodel))
            return None
        graphs = []
        gname = "h" if usehist else "g"
        if yvar: gname+="2"
        gname += "_{0:02d}"        
        for i in range(0,100):
            gname_temp = gname.format(i)
            g = f.Get(gname_temp)
            if not g: 
                log().debug("Failed to retrieve graph {0} from config file {1}".format(gname_temp,self.fconfig))
                continue
            g = g.Clone()
            if usehist or isinstance(g, TGraph2D): 
                g.SetDirectory(0)
            graphs += [(float(i)/100., g)]
        f.Close()

        # get x-y limits
        if usehist: 
            xlims = (min([g.GetXaxis().GetXmin() for (i,g) in graphs]), 
                     max([g.GetXaxis().GetBinCenter(g.GetNbinsX()) for (i,g) in graphs]))
        else: 
            xlims = (min([g.GetXaxis().GetXmin() for (i,g) in graphs]), 
                     max([g.GetXaxis().GetXmax() for (i,g) in graphs]))
        if yvar:
            if usehist: 
                ylims = (min([g.GetYaxis().GetXmin() for (i,g) in graphs]), 
                         max([g.GetYaxis().GetBinCenter(g.GetNbinsY()) for (i,g) in graphs]))
            else:      
                ylims = (min([g.GetYaxis().GetXmin() for (i,g) in graphs]), 
                         max([g.GetYaxis().GetXmax() for (i,g) in graphs]))        


        # python based predict        
        self.xlims = xlims
        if yvar: self.ylims = ylims
        if eff: 
            gname_temp = gname.format(int(eff))
            gscan = [g[1] for g in graphs if g[1].GetName() == gname_temp]
            if not gscan: 
                log().error("Couldn't find graph with name {}".format(gname_temp))
                return None 
            self.g = gscan[0]
            self.__predict_single__ = self.__predict_single_eff__
            self.valtype = 'i'
        else:
            self.graphs = graphs
            self.__predict_single__ = self.__predict_single_score__
            self.valtype = 'f'
        
        # extract inputs
        if yvar: invars = [disc,xvar,yvar]
        else:    invars = [disc,xvar] 
        inputs = s.get_arrays(invars=invars, noweight=True)
        
        # predict output
        return self.__predict_python_loop__(inputs)

    #__________________________________________________________________________=buf=
    def get_var_name(self, eff=None):
        """Return unique var naem based on kw args passed to the predict method
        
        :param eff: working point efficiency in percentage 
        :type eff: int
        """
        name = str(self.name)
        if eff is None: name += "Score"
        else: name += "Eff{:02d}".format(int(eff))
        return name
    
    #__________________________________________________________________________=buf=
    def __predict_single_score__(self, invals):
        """Sub-class specific prediction implementation for flat score"""
        # pre-process values
        disc = invals[0] 
        x = min(self.xlims[1], max(self.xlims[0], invals[1]))
        if self.yvar: 
            y = min(self.ylims[1], max(self.ylims[0], invals[2]))
                
        # calculate flattened disc
        cut_lo = None
        eff_lo = None
        cut_hi = None
        eff_hi = None
        for (eff,g) in self.graphs:
            if self.yvar: 
                cut = g.Interpolate(x,y)
            else: 
                if self.usehist: cut = g.Inerpolate(x)
                else:            cut = g.Eval(x)
            if cut <= disc and ((not cut_lo) or abs(cut-disc) < abs(cut_lo-disc)): 
                eff_lo = eff
                cut_lo = cut
            elif cut > disc and ((not cut_hi) or abs(cut-disc) < abs(cut_hi-disc)):
                eff_hi = eff
                cut_hi = cut

            # early break if both lo/hi cuts found
            if cut_hi is not None and cut_lo is not None: break
            
        assert (cut_hi is not None or cut_lo is not None), "Must have at least upper or lower efficiency boundary!"
         
        ## default upper boundary    
        if cut_hi is None:  
            cut_hi = 1.1
            eff_hi = 0.0
        ## default lower boundary            
        if cut_lo is None: 
            cut_lo = -1.1 
            eff_lo = 1.0
            
        # disc higher than default upper boundary (rare)
        if disc > cut_hi:
            newdisc = cut_lo
        # disc lower than default lower boundary (rare)            
        elif disc < cut_lo:
            newdisc = cut_hi
        # disc inside boundaries            
        else:             
            newdisc = self.__transform__(disc, cut_lo, eff_lo, cut_hi, eff_hi)
            
        return newdisc

    #__________________________________________________________________________=buf=
    def __predict_single_eff__(self, invals):
        """Sub-class specific prediction implementation for working point"""                
        # pre-process values
        disc = invals[0] 
        x = min(self.xlims[1], max(self.xlims[0], invals[1]))
        if self.yvar: 
            y = min(self.ylims[1], max(self.ylims[0], invals[2]))
                
        # calculate cut decision
        if self.yvar:
            if self.usehist:
                cut = self.g.Interpolate(x,y)
            else: 
                cut = self.g.Interpolate(x,y)
        else: 
            if self.usehist: 
                cut = self.g.Interpolate(x)
            else: 
                cut = self.g.Eval(x)
        if self.reverse: 
            result = disc < cut
        else: 
            result = disc > cut

        return result
        
    #__________________________________________________________________________=buf=
    def __transform__(self, disc, cut_lo, eff_lo, cut_hi, eff_hi):
        """Interpolate discriminant between two efficiency boundaries"""
        newdisc = 1. - eff_lo - (disc - cut_lo)/(cut_hi - cut_lo) * (eff_hi - eff_lo)
        if self.reverse: newdisc = 1.0 - newdisc
        return newdisc
     

