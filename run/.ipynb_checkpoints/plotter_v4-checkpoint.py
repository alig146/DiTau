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

    (((DiTauJetsAuxDyn.n_tracks_lead == 1) && (DiTauJetsAuxDyn.n_tracks_subl == 3)) || ((DiTauJetsAuxDyn.n_tracks_subl == 1) && (DiTauJetsAuxDyn.n_tracks_lead == 3)))
    
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
    
    file_paths= ['/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_v0_output.root/user.*.output.root',
                 '/global/u2/a/agarabag/pscratch/ditdau_samples/user.agarabag.DiTauMC20.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_v0_output.root/user.*.output.root']
    mini_branches= ['DiTauJetsAuxDyn.ditau_pt', 'EventInfoAuxDyn.mcEventWeights', 'DiTauJetsAuxDyn.BDTScore', 
                    'DiTauJetsAuxDyn.n_subjets', 'DiTauJetsAuxDyn.n_tracks_lead', 'DiTauJetsAuxDyn.n_tracks_subl', 
                    'DiTauJetsAux.eta', 'EventInfoAuxDyn.averageInteractionsPerCrossing']
    test_Tree = read_tree(file_paths, mini_branches)
    # file_paths = [
    # 'user.agarabag.DiTauMC20.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_v0_output.root/user.agarabag.34455039._000002.output.root',
    # 'user.agarabag.DiTauMC20.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/user.agarabag.34455043._000002.output.root',
    # 'user.agarabag.DiTauMC20.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/user.agarabag.34455045._000001.output.root',
    # 'user.agarabag.DiTauMC20.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/user.agarabag.34455049._000002.output.root',
    # 'user.agarabag.DiTauMC20.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/user.agarabag.34455051._000001.output.root',
    # 'user.agarabag.DiTauMC20.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/user.agarabag.34455056._000002.output.root',
    # 'user.agarabag.DiTauMC20.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/user.agarabag.34455059._000001.output.root',
    # 'user.agarabag.DiTauMC20.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/user.agarabag.34455061._000001.output.root',
    # 'user.agarabag.DiTauMC20.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_v0_output.root/user.agarabag.34455064._000002.output.root',
    # 'user.agarabag.DiTauMC20.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_v0_output.root/user.agarabag.34455068._000001.output.root',
    # 'user.agarabag.DiTauMC20.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_v0_output.root/user.agarabag.34455072._000001.output.root',
    # 'user.agarabag.DiTauMC20.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_v0_output.root/user.agarabag.34455076._000001.output.root']
    # # 'user.agarabag.DiJetMC20_JZ0.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_v0_output.root/user.agarabag.35047097._000001.output.root']
    # file_paths = [path + file_path for file_path in file_paths]

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
    # c1 = getNevents(glob.glob(file_paths[0]))
    # c2 = getNevents(glob.glob(file_paths[1]))
    # c3 = getNevents(glob.glob(file_paths[2]))
    # c4 = getNevents(glob.glob(file_paths[3]))
    # c5 = getNevents(glob.glob(file_paths[4]))
    # c6 = getNevents(glob.glob(file_paths[5]))
    # c7 = getNevents(glob.glob(file_paths[6]))
    # c8 = getNevents(glob.glob(file_paths[7]))
    # c9 = getNevents(glob.glob(file_paths[8]))
    # c10 = getNevents(glob.glob(file_paths[9]))
    # c11 = getNevents(glob.glob(file_paths[10]))
    # c12 = getNevents(glob.glob(file_paths[11]))
    
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
    bkg_mu = 'EventInfoAuxDyn.averageInteractionsPerCrossing'
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

    bkg_data_mu = np.concatenate([
    jz1[bkg_mu],
    jz2[bkg_mu],
    jz3[bkg_mu],
    jz4[bkg_mu],
    jz5[bkg_mu],
    jz6[bkg_mu],
    jz7[bkg_mu],
    jz8[bkg_mu],
    jz9[bkg_mu],
    jz10[bkg_mu],
    jz11[bkg_mu],
    jz12[bkg_mu]
    ])
    print("bkg_mu: ", len(bkg_mu))
    print("data: ", len(data))
    print("data_un_flat: ", len(data_un_flat))

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

    bk_pt_plt = plt_to_root_hist_w(data, 100, 200000, 3000000, bkg_evt_weights, False)
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
    
    ###### mu calcalution for Graviton ######
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
    ######################################

    ###### mu calcalution for Dijet ######
    dijet_mu = np.zeros(len(data))
    # and then assign the same mu value in each event to each subjet in that event
    for i, subjets in enumerate(data_un_flat):
        # Check how many subjets are in each event
        if len(subjets) <= 1:
            # Assign the same mu value to single subjets in the event
            dijet_mu[i] = bkg_data_mu[i]
        else:
            # Assign the same mu value to each subjet in the event
            dijet_mu[i:i+len(subjets)] = bkg_data_mu[i]
    ######################################
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

    # denom_mu, dnom_bin_edges_mu = np.histogram(new_mu[cuts], bins=mu_bin)
    # num_mu, num_bin_edges_mu = np.histogram(new_mu[cuts_bdt], bins=mu_bin)
    # denom_mu = np.array(denom_mu).astype(float)   
    # num_mu = np.array(num_mu).astype(float)
    # efficiency_mu = np.divide(num_mu, denom_mu, out=np.zeros_like(num_mu), where=denom_mu!=0).astype(float)

    # denom_mu_1p1p, dnom_bin_edges_mu_1p1p = np.histogram(new_mu[cuts_1p1p], bins=mu_bin)
    # num_mu_1p1p, num_bin_edges_mu_1p1p = np.histogram(new_mu[cuts_bdt_1p1p], bins=mu_bin)
    # denom_mu_1p1p = np.array(denom_mu_1p1p).astype(float)
    # num_mu_1p1p = np.array(num_mu_1p1p).astype(float)
    # efficiency_mu_1p1p = np.divide(num_mu_1p1p, denom_mu_1p1p, out=np.zeros_like(num_mu_1p1p), where=denom_mu_1p1p!=0).astype(float)

    # denom_mu_3p3p, dnom_bin_edges_mu_3p3p = np.histogram(new_mu[cuts_3p3p], bins=mu_bin)
    # num_mu_3p3p, num_bin_edges_mu_3p3p = np.histogram(new_mu[cuts_bdt_3p3p], bins=mu_bin)
    # denom_mu_3p3p = np.array(denom_mu_3p3p).astype(float)
    # num_mu_3p3p = np.array(num_mu_3p3p).astype(float)
    # efficiency_mu_3p3p = np.divide(num_mu_3p3p, denom_mu_3p3p, out=np.zeros_like(num_mu_3p3p), where=denom_mu_3p3p!=0).astype(float)

    # denom_mu_inc, dnom_bin_edges_mu_inc = np.histogram(new_mu[cuts_inc], bins=mu_bin)
    # num_mu_inc, num_bin_edges_mu_inc = np.histogram(new_mu[cuts_bdt_inc], bins=mu_bin)
    # denom_mu_inc = np.array(denom_mu_inc).astype(float)
    # num_mu_inc = np.array(num_mu_inc).astype(float)
    # efficiency_mu_inc = np.divide(num_mu_inc, denom_mu_inc, out=np.zeros_like(num_mu_inc), where=denom_mu_inc!=0).astype(float)

    # denom_mu_w, dnom_bin_edges_mu_w = np.histogram(new_mu[cuts], bins=mu_bin, weights=weights[cuts])
    # num_mu_w, num_bin_edges_mu_w = np.histogram(new_mu[cuts_bdt], bins=mu_bin, weights=weights[cuts_bdt])
    # denom_mu_w = np.array(denom_mu_w).astype(float)
    # num_mu_w = np.array(num_mu_w).astype(float)
    # efficiency_mu_w = np.divide(num_mu_w, denom_mu_w, out=np.zeros_like(num_mu_w), where=denom_mu_w!=0).astype(float)

    # denom_mu_1p1p_w, dnom_bin_edges_mu_1p1p_w = np.histogram(new_mu[cuts_1p1p], bins=mu_bin, weights=weights[cuts_1p1p])
    # num_mu_1p1p_w, num_bin_edges_mu_1p1p_w = np.histogram(new_mu[cuts_bdt_1p1p], bins=mu_bin, weights=weights[cuts_bdt_1p1p])
    # denom_mu_1p1p_w = np.array(denom_mu_1p1p_w).astype(float)
    # num_mu_1p1p_w = np.array(num_mu_1p1p_w).astype(float)
    # efficiency_mu_1p1p_w = np.divide(num_mu_1p1p_w, denom_mu_1p1p_w, out=np.zeros_like(num_mu_1p1p_w), where=denom_mu_1p1p_w!=0).astype(float)

    # denom_mu_3p3p_w, dnom_bin_edges_mu_3p3p_w = np.histogram(new_mu[cuts_3p3p], bins=mu_bin, weights=weights[cuts_3p3p])
    # num_mu_3p3p_w, num_bin_edges_mu_3p3p_w = np.histogram(new_mu[cuts_bdt_3p3p], bins=mu_bin, weights=weights[cuts_bdt_3p3p])
    # denom_mu_3p3p_w = np.array(denom_mu_3p3p_w).astype(float)
    # num_mu_3p3p_w = np.array(num_mu_3p3p_w).astype(float)
    # efficiency_mu_3p3p_w = np.divide(num_mu_3p3p_w, denom_mu_3p3p_w, out=np.zeros_like(num_mu_3p3p_w), where=denom_mu_3p3p_w!=0).astype(float)

    # denom_mu_inc_w, dnom_bin_edges_mu_inc_w = np.histogram(new_mu[cuts_inc], bins=mu_bin, weights=weights[cuts_inc])
    # num_mu_inc_w, num_bin_edges_mu_inc_w = np.histogram(new_mu[cuts_bdt_inc], bins=mu_bin, weights=weights[cuts_bdt_inc])
    # denom_mu_inc_w = np.array(denom_mu_inc_w).astype(float)
    # num_mu_inc_w = np.array(num_mu_inc_w).astype(float)
    # efficiency_mu_inc_w = np.divide(num_mu_inc_w, denom_mu_inc_w, out=np.zeros_like(num_mu_inc_w), where=denom_mu_inc_w!=0).astype(float)


    # fig6 = plt.figure()
    # plt.plot(mu_bin[:-1], efficiency_mu, label="1p3p unweighted", color='black')
    # plt.plot(mu_bin[:-1], efficiency_mu_1p1p, label="1p1p unweighted", color='orange')
    # plt.plot(mu_bin[:-1], efficiency_mu_3p3p, label="3p3p unweighted", color='red')
    # plt.plot(mu_bin[:-1], efficiency_mu_inc, label="inclusive unweighted", color='green')
    # plt.plot(mu_bin[:-1], efficiency_mu_w, label="1p3p weights", linestyle='dashed', color='black')
    # plt.plot(mu_bin[:-1], efficiency_mu_1p1p_w, label="1p1p weights", linestyle='dashed', color='orange')
    # plt.plot(mu_bin[:-1], efficiency_mu_3p3p_w, label="3p3p weights", linestyle='dashed', color='red')
    # plt.plot(mu_bin[:-1], efficiency_mu_inc_w, label="incl weights", linestyle='dashed', color='green')
    # plt.legend()
    # plt.xlabel("average mu")
    # plt.ylabel("efficiency")
    # p.savefig(fig6)
    # plt.close(fig6)



  

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

    signal_full_event_weight = event_weight(f1)*getXS(425102)/event_weight_sum(f1)
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
    fpr_1p3p, tpr_1p3p = calc_roc(signal_scores, background_scores, signal_weight, background_weight)
    fpr_1p1p, tpr_1p1p = calc_roc(signal_scores_1p1p, background_scores_1p1p, signal_weight_1p1p, background_weight_1p1p)
    fpr_3p3p, tpr_3p3p = calc_roc(signal_scores_3p3p, background_scores_3p3p, signal_weight_3p3p, background_weight_3p3p)
    fpr_inc, tpr_inc = calc_roc(signal_scores_inc, background_scores_inc, signal_weight_inc, background_weight_inc)

    fpr_1p3p_w, tpr_1p3p_w = calc_roc(signal_scores, background_scores, signal_weight*weights[cuts], background_weight*bkg_weights[bkg_cuts])
    fpr_1p1p_w, tpr_1p1p_w = calc_roc(signal_scores_1p1p, background_scores_1p1p, signal_weight_1p1p*weights[cuts_1p1p], background_weight_1p1p*bkg_weights[bkg_cuts_1p1p])
    fpr_3p3p_w, tpr_3p3p_w = calc_roc(signal_scores_3p3p, background_scores_3p3p, signal_weight_3p3p*weights[cuts_3p3p], background_weight_3p3p*bkg_weights[bkg_cuts_3p3p])
    fpr_inc_w, tpr_inc_w = calc_roc(signal_scores_inc, background_scores_inc, signal_weight_inc*weights[cuts_inc], background_weight_inc*bkg_weights[bkg_cuts_inc])

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
    # sig_pt02_1 = cuts
    # sig_pt1_2 = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] >= 1000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] < 2000000))
    # sig_pt2_3 = ak.where((ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] >= 2000000) & (ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt'])[cuts] < 3000000))
    # bkg_pt02_1 = ak.where((data[bkg_cuts] >= 200000) & (data[bkg_cuts] <= 1000000))
    # bkg_pt02_1 = bkg_cuts
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

    # signal_scores_pt02_1 = ak.flatten(f1['DiTauJetsAuxDyn.BDTScore'])[sig_pt02_1]
    # print("TTTTTTTTTTTT: ", len(signal_scores_pt02_1))
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

    # signal_weight_pt02_1 = (event_weight(f1)*getXS(425102)/event_weight_sum(f1))[sig_pt02_1]
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

    # background_scores_pt02_1 = bkg_bdt[bkg_pt02_1]
    # print("BBBBBBBBBBBB: ", len(background_scores_pt02_1))
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

    # background_weight_pt02_1 = bkg_evt_weights[bkg_pt02_1]
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

    # fpr_1p3p_w_pt02_1, tpr_1p3p_w_pt02_1 = calc_roc(signal_scores_pt02_1, background_scores_pt02_1, signal_weight_pt02_1*weights[sig_pt02_1], background_weight_pt02_1*bkg_weights[bkg_pt02_1])
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

    # print("111111: ", len(signal_scores_pt1_2))
    # print("222222: ", len(signal_scores_pt2_3))

    # fig10 = plt.figure()
    # plt.plot(tpr_1p3p_w_pt02_1, 1/fpr_1p3p_w_pt02_1, label="1p3p", color='black')
    # plt.plot(tpr_1p1p_w_pt02_1, 1/fpr_1p1p_w_pt02_1, label="1p1p", color='orange')
    # plt.plot(tpr_3p3p_w_pt02_1, 1/fpr_3p3p_w_pt02_1, label="3p3p", color='red')
    # plt.plot(tpr_inc_w_pt02_1, 1/fpr_inc_w_pt02_1, label="inclusive", color='green')
    # plt.legend(loc='upper right')
    # plt.xlabel("TPR")
    # plt.ylabel("1/FPR")
    # plt.yscale('log')
    # p.savefig(fig10)
    # plt.close(fig10)

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
    plt.hist(signal_scores, bins=60, histtype="step", label="1p3p sig", color='black', weights=signal_weight)
    plt.hist(background_scores, bins=60, histtype="step", label="1p3p bkg", linestyle='dashed', color='black', weights=background_weight)
    plt.hist(signal_scores_1p1p, bins=60, histtype="step", label="1p1p sig", color='orange', weights=signal_weight_1p1p)
    plt.hist(background_scores_1p1p, bins=60, histtype="step", label="1p1p bkg", linestyle='dashed', color='orange', weights=background_weight_1p1p)
    plt.hist(signal_scores_3p3p, bins=60, histtype="step", label="3p3p sig", color='red', weights=signal_weight_3p3p)
    plt.hist(background_scores_3p3p, bins=60, histtype="step", label="3p3p bkg", linestyle='dashed', color='red', weights=background_weight_3p3p)
    plt.hist(signal_scores_inc, bins=60, histtype="step", label="inc sig", color='green', weights=signal_weight_inc)
    plt.hist(background_scores_inc, bins=60, histtype="step", label="inc bkg", linestyle='dashed', color='green', weights=background_weight_inc)
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

    pt_1p3p_eff, pt_1p1p_eff, pt_3p3p_eff, pt_inc_eff = plot_eff(data, bkg_cuts_list, "DiJet pT", 20, 200000, 1000000, bkg_evt_weights, None, eta=False)
    pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(data, bkg_cuts_list, "DiJet pT", 20, 200000, 1000000, bkg_evt_weights, bkg_weights, eta=False)
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

    eta_1p3p_eff, eta_1p1p_eff, eta_3p3p_eff, eta_inc_eff = plot_eff(data_eta, bkg_cuts_list, "DiJet eta", 40, -2.5, 2.5, bkg_evt_weights, None, eta=True)
    eta_1p3p_eff_w, eta_1p1p_eff_w, eta_3p3p_eff_w, eta_inc_eff_w = plot_eff(data_eta, bkg_cuts_list, "DiJet eta", 40, -2.5, 2.5, bkg_evt_weights, bkg_weights, eta=True)
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


    eta_inc_eff.SetMarkerStyle(41)
    eta_inc_eff.SetLineColor(ROOT.kGreen)
    eta_inc_eff.Draw(" e")
    eta_inc_eff_w.SetMarkerStyle(22)
    eta_inc_eff_w.SetLineColor(ROOT.kBlue)
    eta_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.7, 0.7, 0.7, 0.7)
    legend.AddEntry(eta_inc_eff, "inclusive")
    legend.AddEntry(eta_inc_eff_w, "inclusive weighted")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    mu_1p3p_eff, mu_1p1p_eff, mu_3p3p_eff, mu_inc_eff = plot_eff(dijet_mu, bkg_cuts_list, "DiJet mu", 20, 18, 74, bkg_evt_weights, None, eta=False)
    mu_1p3p_eff_w, mu_1p1p_eff_w, mu_3p3p_eff_w, mu_inc_eff_w = plot_eff(dijet_mu, bkg_cuts_list, "DiJet mu", 20, 18, 74, bkg_evt_weights, bkg_weights, eta=False)
    mu_1p3p_eff.SetMarkerStyle(41)
    mu_1p1p_eff.SetMarkerStyle(41)
    mu_3p3p_eff.SetMarkerStyle(41)
    mu_inc_eff.SetMarkerStyle(41)
    mu_1p3p_eff.Draw(" e")
    mu_1p1p_eff.Draw("same e")
    mu_3p3p_eff.Draw("same e")
    mu_inc_eff.Draw("same e")
    mu_1p3p_eff_w.SetMarkerStyle(22)
    mu_1p1p_eff_w.SetMarkerStyle(22)
    mu_3p3p_eff_w.SetMarkerStyle(22)
    mu_inc_eff_w.SetMarkerStyle(22)
    mu_1p3p_eff_w.Draw("same e")
    mu_1p1p_eff_w.Draw("same e")
    mu_3p3p_eff_w.Draw("same e")
    mu_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(mu_1p3p_eff, "1p3p")
    legend.AddEntry(mu_1p1p_eff, "1p1p")
    legend.AddEntry(mu_3p3p_eff, "3p3p")
    legend.AddEntry(mu_inc_eff, "inclusive")
    legend.AddEntry(mu_1p3p_eff_w, "1p3p w")
    legend.AddEntry(mu_1p1p_eff_w, "1p1p w")
    legend.AddEntry(mu_3p3p_eff_w, "3p3p w")
    legend.AddEntry(mu_inc_eff_w, "inclusive w")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    mu_inc_eff.SetMarkerStyle(41)
    mu_inc_eff.SetLineColor(ROOT.kGreen)
    mu_inc_eff.Draw(" e")
    mu_inc_eff_w.SetMarkerStyle(22)
    mu_inc_eff_w.SetLineColor(ROOT.kBlue)
    mu_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.7, 0.7, 0.7, 0.7)
    legend.AddEntry(mu_inc_eff, "inclusive")
    legend.AddEntry(mu_inc_eff_w, "inclusive weighted")
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    pt_sig_1p3p_eff, pt_sig_1p1p_eff, pt_sig_3p3p_eff, pt_sig_inc_eff = plot_eff(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']), signal_cuts_list, "Graviton pT", 20, 200000, 1000000, signal_full_event_weight, None, eta=False)
    pt_sig_1p3p_eff_w, pt_sig_1p1p_eff_w, pt_sig_3p3p_eff_w, pt_sig_inc_eff_w = plot_eff(ak.flatten(f1['DiTauJetsAuxDyn.ditau_pt']), signal_cuts_list, "Graviton pT", 20, 200000, 1000000, signal_full_event_weight, weights, eta=False)
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

    eta_sig_1p3p_eff, eta_sig_1p1p_eff, eta_sig_3p3p_eff, eta_sig_inc_eff = plot_eff(ak.flatten(f1['DiTauJetsAux.eta']), signal_cuts_list, "Graviton eta", 40, -2.5, 2.5, signal_full_event_weight, None, eta=True)
    eta_sig_1p3p_eff_w, eta_sig_1p1p_eff_w, eta_sig_3p3p_eff_w, eta_sig_inc_eff_w = plot_eff(ak.flatten(f1['DiTauJetsAux.eta']), signal_cuts_list, "Graviton eta", 40, -2.5, 2.5, signal_full_event_weight, weights, eta=True)
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

    mu_sig_1p3p_eff, mu_sig_1p1p_eff, mu_sig_3p3p_eff, mu_sig_inc_eff = plot_eff(new_mu, signal_cuts_list, "Graviton mu", 20, 18, 74, signal_full_event_weight, None, eta=False)
    mu_sig_1p3p_eff_w, mu_sig_1p1p_eff_w, mu_sig_3p3p_eff_w, mu_sig_inc_eff_w = plot_eff(new_mu, signal_cuts_list, "Graviton mu", 20, 18, 74, signal_full_event_weight, weights, eta=False)
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
    legend.Draw()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    sig_score_1p3p = plt_to_root_hist_w(signal_scores, 100, 0, 1, signal_weight, False)
    bkg_score_1p3p = plt_to_root_hist_w(background_scores, 100, 0, 1, background_weight, False)
    sig_score_1p1p = plt_to_root_hist_w(signal_scores_1p1p, 100, 0, 1, signal_weight_1p1p, False)
    bkg_score_1p1p = plt_to_root_hist_w(background_scores_1p1p, 100, 0, 1, background_weight_1p1p, False)
    sig_score_3p3p = plt_to_root_hist_w(signal_scores_3p3p, 100, 0, 1, signal_weight_3p3p, False)
    bkg_score_3p3p = plt_to_root_hist_w(background_scores_3p3p, 100, 0, 1, background_weight_3p3p, False)
    sig_score_inc = plt_to_root_hist_w(signal_scores_inc, 100, 0, 1, signal_weight_inc, False)
    bkg_score_inc = plt_to_root_hist_w(background_scores_inc, 100, 0, 1, background_weight_inc, False)

    sig_score_1p3p_w = plt_to_root_hist_w(signal_scores, 100, 0, 1, signal_weight*weights[cuts], False)
    bkg_score_1p3p_w = plt_to_root_hist_w(background_scores, 100, 0, 1, background_weight*bkg_weights[bkg_cuts], False)
    sig_score_1p1p_w = plt_to_root_hist_w(signal_scores_1p1p, 100, 0, 1, signal_weight_1p1p*weights[cuts_1p1p], False)
    bkg_score_1p1p_w = plt_to_root_hist_w(background_scores_1p1p, 100, 0, 1, background_weight_1p1p*bkg_weights[bkg_cuts_1p1p], False)
    sig_score_3p3p_w = plt_to_root_hist_w(signal_scores_3p3p, 100, 0, 1, signal_weight_3p3p*weights[cuts_3p3p], False)
    bkg_score_3p3p_w = plt_to_root_hist_w(background_scores_3p3p, 100, 0, 1, background_weight_3p3p*bkg_weights[bkg_cuts_3p3p], False)
    sig_score_inc_w = plt_to_root_hist_w(signal_scores_inc, 100, 0, 1, signal_weight_inc*weights[cuts_inc], False)
    bkg_score_inc_w = plt_to_root_hist_w(background_scores_inc, 100, 0, 1, background_weight_inc*bkg_weights[bkg_cuts_inc], False)
    
    root_1p3p_roc = create_roc_graph(sig_score_1p3p, bkg_score_1p3p, effmin=0.05, name="1p3p", normalize=False, reverse=False)
    root_1p1p_roc = create_roc_graph(sig_score_1p1p, bkg_score_1p1p, effmin=0.05, name="1p1p", normalize=False, reverse=False)
    root_3p3p_roc = create_roc_graph(sig_score_3p3p, bkg_score_3p3p, effmin=0.05, name="3p3p", normalize=False, reverse=False)
    root_inc_roc = create_roc_graph(sig_score_inc, bkg_score_inc, effmin=0.05, name="inc", normalize=False, reverse=False)
    root_1p3p_roc_w = create_roc_graph(sig_score_1p3p_w, bkg_score_1p3p_w, effmin=0.05, name="1p3p_w", normalize=False, reverse=False)
    root_1p1p_roc_w = create_roc_graph(sig_score_1p1p_w, bkg_score_1p1p_w, effmin=0.05, name="1p1p_w", normalize=False, reverse=False)
    root_3p3p_roc_w = create_roc_graph(sig_score_3p3p_w, bkg_score_3p3p_w, effmin=0.05, name="3p3p_w", normalize=False, reverse=False)
    root_inc_roc_w = create_roc_graph(sig_score_inc_w, bkg_score_inc_w, effmin=0.05, name="inc_w", normalize=False, reverse=False)
    #set colors and styles
    root_1p3p_roc.SetLineColor(ROOT.kBlack)
    root_1p1p_roc.SetLineColor(ROOT.kOrange)
    root_3p3p_roc.SetLineColor(ROOT.kRed)
    root_inc_roc.SetLineColor(ROOT.kGreen)
    root_1p3p_roc_w.SetLineColor(ROOT.kBlack)
    root_1p1p_roc_w.SetLineColor(ROOT.kOrange)
    root_3p3p_roc_w.SetLineColor(ROOT.kRed)
    root_inc_roc_w.SetLineColor(ROOT.kGreen)
    root_1p3p_roc_w.SetLineStyle(9)
    root_1p1p_roc_w.SetLineStyle(9)
    root_3p3p_roc_w.SetLineStyle(9)
    root_inc_roc_w.SetLineStyle(9)
    # draw
    root_1p3p_roc.Draw("")
    root_1p1p_roc.Draw("same")
    root_3p3p_roc.Draw("same")
    root_inc_roc.Draw("same")
    root_1p3p_roc_w.Draw("same")
    root_1p1p_roc_w.Draw("same")
    root_3p3p_roc_w.Draw("same")
    root_inc_roc_w.Draw("same")
    ROOT.gPad.SetLogy()
    #legend
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.AddEntry(root_1p3p_roc, "1p3p")
    legend.AddEntry(root_1p1p_roc, "1p1p")
    legend.AddEntry(root_3p3p_roc, "3p3p")
    legend.AddEntry(root_inc_roc, "inclusive")
    legend.AddEntry(root_1p3p_roc_w, "1p3p w")
    legend.AddEntry(root_1p1p_roc_w, "1p1p w")
    legend.AddEntry(root_3p3p_roc_w, "3p3p w")
    legend.AddEntry(root_inc_roc_w, "inclusive w")
    legend.Draw()
    #log y axis
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    canvas.Print("eff_plots.pdf]")





if __name__ == "__main__":
    plot_branches()
