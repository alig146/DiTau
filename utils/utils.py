import glob, os, sys
import uproot, ROOT
import random
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import pandas as pd
from array import array
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

def plot_eff(data, weights, name, num_bins, x_min, x_max, eta=False):
    
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

    # pt_1p3p_eff = make_eff_hist(pt_1p3p_dnom, pt_1p3p_num, "1p3p_eff")
    # pt_1p1p_eff = make_eff_hist(pt_1p1p_dnom, pt_1p1p_num, "1p1p_eff")
    # pt_3p3p_eff = make_eff_hist(pt_3p3p_dnom, pt_3p3p_num, "3p3p_eff")
    # pt_inc_eff = make_eff_hist(pt_inc_dnom, pt_inc_num, "inc_eff")

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