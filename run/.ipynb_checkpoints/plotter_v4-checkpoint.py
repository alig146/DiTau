import glob, os, sys
import uproot, ROOT
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, roc_auc_score, auc
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

    # cut8 = (df_chunk['bdt_score'] >= 0.72) 
    cut8 = (df_chunk['bdt_score_new'] >= 0.72) 


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

    # cut8 = (df_chunk['bdt_score'] > 0.55)
    cut8 = (df_chunk['bdt_score_new'] > 0.55)


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
    graviton_xs = [425108, 425100, 425101, 425102, 425103, 425104, 425105, 425106, 425107, 425200]
    # gamma_xs = [425200]

    # File Location. order maatch XS. 
    bkg_filelist = []
    for index in range(12):
        bkg_filelist.append(path+f"jz_w_newbdt/dijet_flattened_jz{index+1}.h5")

    graviton_filelist = [path+"signal_w_newbdt/graviton_flattened_M1000.h5",
                    path+"signal_w_newbdt/graviton_flattened_M1500.h5",
                    path+"signal_w_newbdt/graviton_flattened_M1750.h5",
                    path+"signal_w_newbdt/graviton_flattened_M2000.h5",
                    path+"signal_w_newbdt/graviton_flattened_M2250.h5",
                    path+"signal_w_newbdt/graviton_flattened_M2500.h5",
                    path+"signal_w_newbdt/graviton_flattened_M3000.h5",
                    path+"signal_w_newbdt/graviton_flattened_M4000.h5",
                    path+"signal_w_newbdt/graviton_flattened_M5000.h5",
                    path+"signal_w_newbdt/gamma_flattened_0.h5"]

    # gamma_filelist = [path+"signal_w_newbdt/gamma_flattened_0.h5"]   


    combined_signal_1p3p = h52panda(graviton_filelist, graviton_xs, signal_cut, pp13=True)
    combined_signal_1p3p_bdt = h52panda(graviton_filelist, graviton_xs, signal_cut, pp13=True, bdt=True)
    combined_signal_1p1p = h52panda(graviton_filelist, graviton_xs, signal_cut, pp11=True)
    combined_signal_1p1p_bdt = h52panda(graviton_filelist, graviton_xs, signal_cut, pp11=True, bdt=True)
    combined_signal_3p3p = h52panda(graviton_filelist, graviton_xs, signal_cut, pp33=True)
    combined_signal_3p3p_bdt = h52panda(graviton_filelist, graviton_xs, signal_cut, pp33=True, bdt=True)
    combined_signal_inc = h52panda(graviton_filelist, graviton_xs, signal_cut, ppinc=True)
    combined_signal_inc_bdt = h52panda(graviton_filelist, graviton_xs, signal_cut, ppinc=True, bdt=True)
    combined_signal_1p3p['weight'] = combined_signal_1p3p['event_weight'] * combined_signal_1p3p['pT_weight']
    combined_signal_1p3p_bdt['weight'] = combined_signal_1p3p_bdt['event_weight'] * combined_signal_1p3p_bdt['pT_weight']
    combined_signal_1p1p['weight'] = combined_signal_1p1p['event_weight'] * combined_signal_1p1p['pT_weight']
    combined_signal_1p1p_bdt['weight'] = combined_signal_1p1p_bdt['event_weight'] * combined_signal_1p1p_bdt['pT_weight']
    combined_signal_3p3p['weight'] = combined_signal_3p3p['event_weight'] * combined_signal_3p3p['pT_weight']
    combined_signal_3p3p_bdt['weight'] = combined_signal_3p3p_bdt['event_weight'] * combined_signal_3p3p_bdt['pT_weight']
    combined_signal_inc['weight'] = combined_signal_inc['event_weight'] * combined_signal_inc['pT_weight']
    combined_signal_inc_bdt['weight'] = combined_signal_inc_bdt['event_weight'] * combined_signal_inc_bdt['pT_weight']


    # combined_bkg_1p3p = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp13=True)
    # combined_bkg_1p3p_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp13=True, bdt=True)
    # combined_bkg_1p1p = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp11=True)
    # combined_bkg_1p1p_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp11=True, bdt=True)
    # combined_bkg_3p3p = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp33=True)
    # combined_bkg_3p3p_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, pp33=True, bdt=True)
    # combined_bkg_inc = h52panda(bkg_filelist, bkg_xs, bkg_cut, ppinc=True)
    # combined_bkg_inc_bdt = h52panda(bkg_filelist, bkg_xs, bkg_cut, ppinc=True, bdt=True)
    # combined_bkg_1p3p['weight'] = combined_bkg_1p3p['event_weight'] * combined_bkg_1p3p['pT_weight']
    # combined_bkg_1p3p_bdt['weight'] = combined_bkg_1p3p_bdt['event_weight'] * combined_bkg_1p3p_bdt['pT_weight']
    # combined_bkg_1p1p['weight'] = combined_bkg_1p1p['event_weight'] * combined_bkg_1p1p['pT_weight']
    # combined_bkg_1p1p_bdt['weight'] = combined_bkg_1p1p_bdt['event_weight'] * combined_bkg_1p1p_bdt['pT_weight']
    # combined_bkg_3p3p['weight'] = combined_bkg_3p3p['event_weight'] * combined_bkg_3p3p['pT_weight']
    # combined_bkg_3p3p_bdt['weight'] = combined_bkg_3p3p_bdt['event_weight'] * combined_bkg_3p3p_bdt['pT_weight']
    # combined_bkg_inc['weight'] = combined_bkg_inc['event_weight'] * combined_bkg_inc['pT_weight']
    # combined_bkg_inc_bdt['weight'] = combined_bkg_inc_bdt['event_weight'] * combined_bkg_inc_bdt['pT_weight']
    
    # combined_bkg_1p3p.to_csv(path+'combined_bkg_1p3p_new.csv', index=False)
    # combined_bkg_1p3p_bdt.to_csv(path+'combined_bkg_1p3p_bdt_new.csv', index=False)
    # combined_bkg_1p1p.to_csv(path+'combined_bkg_1p1p_new.csv', index=False)
    # combined_bkg_1p1p_bdt.to_csv(path+'combined_bkg_1p1p_bdt_new.csv', index=False)
    # combined_bkg_3p3p.to_csv(path+'combined_bkg_3p3p_new.csv', index=False)
    # combined_bkg_3p3p_bdt.to_csv(path+'combined_bkg_3p3p_bdt_new.csv', index=False)
    # combined_bkg_inc.to_csv(path+'combined_bkg_inc_new.csv', index=False)
    # combined_bkg_inc_bdt.to_csv(path+'combined_bkg_inc_bdt_new.csv', index=False)

    # combined_bkg_1p3p = pd.read_csv(path+'combined_bkg_1p3p.csv')
    # combined_bkg_1p3p_bdt = pd.read_csv(path+'combined_bkg_1p3p_bdt.csv')
    # combined_bkg_1p1p = pd.read_csv(path+'combined_bkg_1p1p.csv')
    # combined_bkg_1p1p_bdt = pd.read_csv(path+'combined_bkg_1p1p_bdt.csv')
    # combined_bkg_3p3p = pd.read_csv(path+'combined_bkg_3p3p.csv')
    # combined_bkg_3p3p_bdt = pd.read_csv(path+'combined_bkg_3p3p_bdt.csv')
    # combined_bkg_inc = pd.read_csv(path+'combined_bkg_inc.csv')
    # combined_bkg_inc_bdt = pd.read_csv(path+'combined_bkg_inc_bdt.csv')

    combined_bkg_1p3p = pd.read_csv(path+'combined_bkg_1p3p_new.csv')
    combined_bkg_1p3p_bdt = pd.read_csv(path+'combined_bkg_1p3p_bdt_new.csv')
    combined_bkg_1p1p = pd.read_csv(path+'combined_bkg_1p1p_new.csv')
    combined_bkg_1p1p_bdt = pd.read_csv(path+'combined_bkg_1p1p_bdt_new.csv')
    combined_bkg_3p3p = pd.read_csv(path+'combined_bkg_3p3p_new.csv')
    combined_bkg_3p3p_bdt = pd.read_csv(path+'combined_bkg_3p3p_bdt_new.csv')
    combined_bkg_inc = pd.read_csv(path+'combined_bkg_inc_new.csv')
    combined_bkg_inc_bdt = pd.read_csv(path+'combined_bkg_inc_bdt_new.csv')

    # print(len(combined_bkg_1p3p), len(combined_bkg_1p3p_bdt), len(combined_bkg_1p1p), len(combined_bkg_1p1p_bdt), len(combined_bkg_3p3p), len(combined_bkg_3p3p_bdt), len(combined_bkg_inc), len(combined_bkg_inc_bdt))
    
    # use non trained data
    combined_bkg_1p3p = combined_bkg_1p3p[(combined_bkg_1p3p['event_id']%10) >= 7] # 30% of data
    combined_bkg_1p3p_bdt = combined_bkg_1p3p_bdt[(combined_bkg_1p3p_bdt['event_id']%10) >= 7] # 30% of data
    combined_bkg_1p1p = combined_bkg_1p1p[(combined_bkg_1p1p['event_id']%10) >= 7] # 30% of data
    combined_bkg_1p1p_bdt = combined_bkg_1p1p_bdt[(combined_bkg_1p1p_bdt['event_id']%10) >= 7] # 30% of data
    combined_bkg_3p3p = combined_bkg_3p3p[(combined_bkg_3p3p['event_id']%10) >= 7] # 30% of data
    combined_bkg_3p3p_bdt = combined_bkg_3p3p_bdt[(combined_bkg_3p3p_bdt['event_id']%10) >= 7] # 30% of data
    combined_bkg_inc = combined_bkg_inc[(combined_bkg_inc['event_id']%10) >= 7] # 30% of data
    combined_bkg_inc_bdt = combined_bkg_inc_bdt[(combined_bkg_inc_bdt['event_id']%10) >= 7] # 30% of data

    combined_signal_1p3p = combined_signal_1p3p[(combined_signal_1p3p['event_id']%10) >= 7] # 30% of data
    combined_signal_1p3p_bdt = combined_signal_1p3p_bdt[(combined_signal_1p3p_bdt['event_id']%10) >= 7] # 30% of data
    combined_signal_1p1p = combined_signal_1p1p[(combined_signal_1p1p['event_id']%10) >= 7] # 30% of data
    combined_signal_1p1p_bdt = combined_signal_1p1p_bdt[(combined_signal_1p1p_bdt['event_id']%10) >= 7] # 30% of data
    combined_signal_3p3p = combined_signal_3p3p[(combined_signal_3p3p['event_id']%10) >= 7] # 30% of data
    combined_signal_3p3p_bdt = combined_signal_3p3p_bdt[(combined_signal_3p3p_bdt['event_id']%10) >= 7] # 30% of data
    combined_signal_inc = combined_signal_inc[(combined_signal_inc['event_id']%10) >= 7] # 30% of data
    combined_signal_inc_bdt = combined_signal_inc_bdt[(combined_signal_inc_bdt['event_id']%10) >= 7] # 30% of data

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
    fpr_1p3p, tpr_1p3p = calc_roc(combined_signal_1p3p['bdt_score'], combined_bkg_1p3p['bdt_score'], combined_signal_1p3p['weight'], combined_bkg_1p3p['weight'])
    fpr_1p1p, tpr_1p1p = calc_roc(combined_signal_1p1p['bdt_score'], combined_bkg_1p1p['bdt_score'], combined_signal_1p1p['weight'], combined_bkg_1p1p['weight'])
    fpr_3p3p, tpr_3p3p = calc_roc(combined_signal_3p3p['bdt_score'], combined_bkg_3p3p['bdt_score'], combined_signal_3p3p['weight'], combined_bkg_3p3p['weight'])
    fpr_inc, tpr_inc = calc_roc(combined_signal_inc['bdt_score'], combined_bkg_inc['bdt_score'], combined_signal_inc['weight'], combined_bkg_inc['weight'])

    fpr_1p3p_new, tpr_1p3p_new = calc_roc(combined_signal_1p3p['bdt_score_new'], combined_bkg_1p3p['bdt_score_new'], combined_signal_1p3p['weight'], combined_bkg_1p3p['weight'])
    fpr_1p1p_new, tpr_1p1p_new = calc_roc(combined_signal_1p1p['bdt_score_new'], combined_bkg_1p1p['bdt_score_new'], combined_signal_1p1p['weight'], combined_bkg_1p1p['weight'])
    fpr_3p3p_new, tpr_3p3p_new = calc_roc(combined_signal_3p3p['bdt_score_new'], combined_bkg_3p3p['bdt_score_new'], combined_signal_3p3p['weight'], combined_bkg_3p3p['weight'])
    fpr_inc_new, tpr_inc_new = calc_roc(combined_signal_inc['bdt_score_new'], combined_bkg_inc['bdt_score_new'], combined_signal_inc['weight'], combined_bkg_inc['weight'])

    #calculate auc 
    auc_1p3p = auc(fpr_1p3p, tpr_1p3p)
    auc_1p1p = auc(fpr_1p1p, tpr_1p1p)
    auc_3p3p = auc(fpr_3p3p, tpr_3p3p)
    auc_inc = auc(fpr_inc, tpr_inc)
    auc_1p3p_new = auc(fpr_1p3p_new, tpr_1p3p_new)
    auc_1p1p_new = auc(fpr_1p1p_new, tpr_1p1p_new)
    auc_3p3p_new = auc(fpr_3p3p_new, tpr_3p3p_new)
    auc_inc_new = auc(fpr_inc_new, tpr_inc_new)
    
    fig7 = plt.figure()
    plt.plot(tpr_1p3p, 1/fpr_1p3p, label=f"1p3p: {auc_1p3p:.4f}" , color='black')
    plt.plot(tpr_1p1p, 1/fpr_1p1p, label=f"1p1p: {auc_1p1p:.4f}", color='orange')
    plt.plot(tpr_3p3p, 1/fpr_3p3p, label=f"3p3p: {auc_3p3p:.4f}", color='red')
    plt.plot(tpr_inc, 1/fpr_inc, label=f"inclusive: {auc_inc:.4f}", color='green')
    plt.plot(tpr_1p3p_new, 1/fpr_1p3p_new, label=f"1p3p new: {auc_1p3p_new:.4f}" , linestyle='dashed', color='black')
    plt.plot(tpr_1p1p_new, 1/fpr_1p1p_new, label=f"1p1p new: {auc_1p1p_new:.4f}", linestyle='dashed', color='orange')
    plt.plot(tpr_3p3p_new, 1/fpr_3p3p_new, label=f"3p3p new: {auc_3p3p_new:.4f}", linestyle='dashed', color='red')
    plt.plot(tpr_inc_new, 1/fpr_inc_new, label=f"inclusive new: {auc_inc_new:.4f}", linestyle='dashed', color='green')

    plt.legend(loc='upper right')
    plt.xlabel("TPR")
    plt.ylabel("1/FPR")
    plt.yscale('log')
    p.savefig(fig7)
    plt.close(fig7)


    ####### plot the score distribution
    fig_score = plt.figure()
    plt.hist(combined_bkg_1p3p['bdt_score'], bins=60, histtype="step", label="1p3p bkg", linestyle='dashed', color='black', weights=combined_bkg_1p3p['weight'])
    plt.hist(combined_bkg_1p3p['bdt_score_new'], bins=60, histtype="step", label="1p3p bkg new", linestyle='dashed', color='blue', weights=combined_bkg_1p3p['weight'])
    plt.hist(combined_signal_1p3p['bdt_score'], bins=60, histtype="step", label="1p3p sig", linestyle='dashed', color='red', weights=combined_signal_1p3p['weight'])
    plt.hist(combined_signal_1p3p['bdt_score_new'], bins=60, histtype="step", label="1p3p sig new", linestyle='dashed', color='green', weights=combined_signal_1p3p['weight'])
    plt.xlabel("BDT score")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_score)
    plt.close(fig_score)

    fig_score2 = plt.figure()
    plt.hist(combined_bkg_1p1p['bdt_score'], bins=60, histtype="step", label="1p1p bkg", linestyle='dashed', color='black', weights=combined_bkg_1p1p['weight'])
    plt.hist(combined_bkg_1p1p['bdt_score_new'], bins=60, histtype="step", label="1p1p bkg new", linestyle='dashed', color='blue', weights=combined_bkg_1p1p['weight'])
    plt.hist(combined_signal_1p1p['bdt_score'], bins=60, histtype="step", label="1p1p sig", linestyle='dashed', color='red', weights=combined_signal_1p1p['weight'])
    plt.hist(combined_signal_1p1p['bdt_score_new'], bins=60, histtype="step", label="1p1p sig new", linestyle='dashed', color='green', weights=combined_signal_1p1p['weight'])
    plt.xlabel("BDT score")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_score2)
    plt.close(fig_score2)

    fig_score3 = plt.figure()
    plt.hist(combined_bkg_3p3p['bdt_score'], bins=60, histtype="step", label="3p3p bkg", linestyle='dashed', color='black', weights=combined_bkg_3p3p['weight'])
    plt.hist(combined_bkg_3p3p['bdt_score_new'], bins=60, histtype="step", label="3p3p bkg new", linestyle='dashed', color='blue', weights=combined_bkg_3p3p['weight'])
    plt.hist(combined_signal_3p3p['bdt_score'], bins=60, histtype="step", label="3p3p sig", linestyle='dashed', color='red', weights=combined_signal_3p3p['weight'])
    plt.hist(combined_signal_3p3p['bdt_score_new'], bins=60, histtype="step", label="3p3p sig new", linestyle='dashed', color='green', weights=combined_signal_3p3p['weight'])
    plt.xlabel("BDT score")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_score3)
    plt.close(fig_score3)

    fig_score4 = plt.figure()
    plt.hist(combined_bkg_inc['bdt_score'], bins=60, histtype="step", label="inc bkg", linestyle='dashed', color='black', weights=combined_bkg_inc['weight'])
    plt.hist(combined_bkg_inc['bdt_score_new'], bins=60, histtype="step", label="inc bkg new", linestyle='dashed', color='blue', weights=combined_bkg_inc['weight'])
    plt.hist(combined_signal_inc['bdt_score'], bins=60, histtype="step", label="inc sig", linestyle='dashed', color='red', weights=combined_signal_inc['weight'])
    plt.hist(combined_signal_inc['bdt_score_new'], bins=60, histtype="step", label="inc sig new", linestyle='dashed', color='green', weights=combined_signal_inc['weight'])
    plt.xlabel("BDT score")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    p.savefig(fig_score4)
    plt.close(fig_score4)


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
    ROOT.gROOT.SetStyle("ATLAS")
    # ROOT.gStyle.SetOptStat(0)
    # ROOT.gStyle.SetOptStat(False)
    # ROOT.gROOT.SetBatch(True)
    canvas = ROOT.TCanvas("canvas", "eff_plots", 500, 500)
    canvas.cd()
    canvas.Print("eff_plots.pdf[")

    sig_pt_list = [combined_signal_1p3p['ditau_pt'], combined_signal_1p3p_bdt['ditau_pt'], combined_signal_1p1p['ditau_pt'], combined_signal_1p1p_bdt['ditau_pt'], combined_signal_3p3p['ditau_pt'], combined_signal_3p3p_bdt['ditau_pt'], combined_signal_inc['ditau_pt'], combined_signal_inc_bdt['ditau_pt']]
    sig_eta_list = [combined_signal_1p3p['eta'], combined_signal_1p3p_bdt['eta'], combined_signal_1p1p['eta'], combined_signal_1p1p_bdt['eta'], combined_signal_3p3p['eta'], combined_signal_3p3p_bdt['eta'], combined_signal_inc['eta'], combined_signal_inc_bdt['eta']]
    sig_mu_list = [combined_signal_1p3p['average_mu'], combined_signal_1p3p_bdt['average_mu'], combined_signal_1p1p['average_mu'], combined_signal_1p1p_bdt['average_mu'], combined_signal_3p3p['average_mu'], combined_signal_3p3p_bdt['average_mu'], combined_signal_inc['average_mu'], combined_signal_inc_bdt['average_mu']]
    sig_w_list = [combined_signal_1p3p['event_weight'], combined_signal_1p3p_bdt['event_weight'], combined_signal_1p1p['event_weight'], combined_signal_1p1p_bdt['event_weight'], combined_signal_3p3p['event_weight'], combined_signal_3p3p_bdt['event_weight'], combined_signal_inc['event_weight'], combined_signal_inc_bdt['event_weight']]

    bkg_pt_list = [combined_bkg_1p3p['ditau_pt'], combined_bkg_1p3p_bdt['ditau_pt'], combined_bkg_1p1p['ditau_pt'], combined_bkg_1p1p_bdt['ditau_pt'], combined_bkg_3p3p['ditau_pt'], combined_bkg_3p3p_bdt['ditau_pt'], combined_bkg_inc['ditau_pt'], combined_bkg_inc_bdt['ditau_pt']]
    bkg_eta_list = [combined_bkg_1p3p['eta'], combined_bkg_1p3p_bdt['eta'], combined_bkg_1p1p['eta'], combined_bkg_1p1p_bdt['eta'], combined_bkg_3p3p['eta'], combined_bkg_3p3p_bdt['eta'], combined_bkg_inc['eta'], combined_bkg_inc_bdt['eta']]
    bkg_mu_list = [combined_bkg_1p3p['average_mu'], combined_bkg_1p3p_bdt['average_mu'], combined_bkg_1p1p['average_mu'], combined_bkg_1p1p_bdt['average_mu'], combined_bkg_3p3p['average_mu'], combined_bkg_3p3p_bdt['average_mu'], combined_bkg_inc['average_mu'], combined_bkg_inc_bdt['average_mu']]
    bkg_w_list = [combined_bkg_1p3p['event_weight'], combined_bkg_1p3p_bdt['event_weight'], combined_bkg_1p1p['event_weight'], combined_bkg_1p1p_bdt['event_weight'], combined_bkg_3p3p['event_weight'], combined_bkg_3p3p_bdt['event_weight'], combined_bkg_inc['event_weight'], combined_bkg_inc_bdt['event_weight']]


    pt_1p3p_eff_w, pt_1p1p_eff_w, pt_3p3p_eff_w, pt_inc_eff_w = plot_eff(bkg_pt_list, bkg_w_list, "DiJet pT", 20, 200000, 1000000, eta=False)
    pt_1p3p_eff_w.SetMarkerStyle(20)
    pt_1p1p_eff_w.SetMarkerStyle(20)
    pt_3p3p_eff_w.SetMarkerStyle(20)
    pt_inc_eff_w.SetMarkerStyle(20)
    pt_1p3p_eff_w.SetMarkerSize(0.6)
    pt_1p1p_eff_w.SetMarkerSize(0.6)
    pt_3p3p_eff_w.SetMarkerSize(0.6)
    pt_inc_eff_w.SetMarkerSize(0.6)
    pt_1p3p_eff_w.Draw(" e")
    pt_1p1p_eff_w.Draw("same e")
    pt_3p3p_eff_w.Draw("same e")
    pt_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9, "", "NDC")
    legend.SetBorderSize(0)
    legend.AddEntry(pt_1p3p_eff_w, "1p3p w")
    legend.AddEntry(pt_1p1p_eff_w, "1p1p w")
    legend.AddEntry(pt_3p3p_eff_w, "3p3p w")
    legend.AddEntry(pt_inc_eff_w, "inclusive w")
    legend.SetFillStyle(0)
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.85, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_1p3p_eff_w, eta_1p1p_eff_w, eta_3p3p_eff_w, eta_inc_eff_w = plot_eff(bkg_eta_list, bkg_w_list, "DiJet eta", 40, -2.5, 2.5, eta=True)
    eta_1p3p_eff_w.SetMarkerStyle(20)
    eta_1p1p_eff_w.SetMarkerStyle(20)
    eta_3p3p_eff_w.SetMarkerStyle(20)
    eta_inc_eff_w.SetMarkerStyle(20)
    eta_1p3p_eff_w.SetMarkerSize(0.6)
    eta_1p1p_eff_w.SetMarkerSize(0.6)
    eta_3p3p_eff_w.SetMarkerSize(0.6)
    eta_inc_eff_w.SetMarkerSize(0.6)
    eta_1p3p_eff_w.Draw(" e")
    eta_1p1p_eff_w.Draw("same e")
    eta_3p3p_eff_w.Draw("same e")
    eta_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.AddEntry(eta_1p3p_eff_w, "1p3p")
    legend.AddEntry(eta_1p1p_eff_w, "1p1p")
    legend.AddEntry(eta_3p3p_eff_w, "3p3p")
    legend.AddEntry(eta_inc_eff_w, "inclusive")
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.85, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_inc_eff_w.SetMarkerStyle(20)
    eta_inc_eff_w.SetMarkerSize(0.6)
    eta_inc_eff_w.SetLineColor(ROOT.kBlue)
    eta_inc_eff_w.Draw(" e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.AddEntry(eta_inc_eff_w, "inclusive")
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.85, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    mu_1p3p_eff_w, mu_1p1p_eff_w, mu_3p3p_eff_w, mu_inc_eff_w = plot_eff(bkg_mu_list, bkg_w_list, "DiJet mu", 20, 18, 74, eta=False)
    mu_1p3p_eff_w.SetMarkerStyle(20)
    mu_1p1p_eff_w.SetMarkerStyle(20)
    mu_3p3p_eff_w.SetMarkerStyle(20)
    mu_inc_eff_w.SetMarkerStyle(20)
    mu_1p3p_eff_w.SetMarkerSize(0.6)
    mu_1p1p_eff_w.SetMarkerSize(0.6)
    mu_3p3p_eff_w.SetMarkerSize(0.6)
    mu_inc_eff_w.SetMarkerSize(0.6)
    mu_1p3p_eff_w.Draw(" e")
    mu_1p1p_eff_w.Draw("same e")
    mu_3p3p_eff_w.Draw("same e")
    mu_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.AddEntry(mu_1p3p_eff_w, "1p3p")
    legend.AddEntry(mu_1p1p_eff_w, "1p1p")
    legend.AddEntry(mu_3p3p_eff_w, "3p3p")
    legend.AddEntry(mu_inc_eff_w, "inclusive")
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.85, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    mu_inc_eff_w.SetMarkerStyle(20)
    mu_inc_eff_w.SetMarkerSize(0.6)
    mu_inc_eff_w.SetLineColor(ROOT.kBlue)
    mu_inc_eff_w.Draw(" e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.AddEntry(mu_inc_eff_w, "inclusive")
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.85, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    ##### signal eff plots

    pt_sig_1p3p_eff_w, pt_sig_1p1p_eff_w, pt_sig_3p3p_eff_w, pt_sig_inc_eff_w = plot_eff(sig_pt_list, sig_w_list, "Signal pT", 20, 200000, 1000000, eta=False)
    pt_sig_1p3p_eff_w.SetMarkerStyle(20)
    pt_sig_1p1p_eff_w.SetMarkerStyle(20)
    pt_sig_3p3p_eff_w.SetMarkerStyle(20)
    pt_sig_inc_eff_w.SetMarkerStyle(20)
    pt_sig_1p3p_eff_w.SetMarkerSize(0.6)
    pt_sig_1p1p_eff_w.SetMarkerSize(0.6)
    pt_sig_3p3p_eff_w.SetMarkerSize(0.6)
    pt_sig_inc_eff_w.SetMarkerSize(0.6)
    pt_sig_1p3p_eff_w.Draw(" e")
    pt_sig_1p1p_eff_w.Draw("same e")
    pt_sig_3p3p_eff_w.Draw("same e")
    pt_sig_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.AddEntry(pt_sig_1p3p_eff_w, "1p3p w")
    legend.AddEntry(pt_sig_1p1p_eff_w, "1p1p w")
    legend.AddEntry(pt_sig_3p3p_eff_w, "3p3p w")
    legend.AddEntry(pt_sig_inc_eff_w, "inclusive w")
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.25, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    eta_sig_1p3p_eff_w, eta_sig_1p1p_eff_w, eta_sig_3p3p_eff_w, eta_sig_inc_eff_w = plot_eff(sig_eta_list, sig_w_list, "Signal eta", 40, -2.5, 2.5, eta=True)
    eta_sig_1p3p_eff_w.SetMarkerStyle(20)
    eta_sig_1p1p_eff_w.SetMarkerStyle(20)
    eta_sig_3p3p_eff_w.SetMarkerStyle(20)
    eta_sig_inc_eff_w.SetMarkerStyle(20)
    eta_sig_1p3p_eff_w.SetMarkerSize(0.6)
    eta_sig_1p1p_eff_w.SetMarkerSize(0.6)
    eta_sig_3p3p_eff_w.SetMarkerSize(0.6)
    eta_sig_inc_eff_w.SetMarkerSize(0.6)
    eta_sig_1p3p_eff_w.Draw(" e")
    eta_sig_1p1p_eff_w.Draw("same e")
    eta_sig_3p3p_eff_w.Draw("same e")
    eta_sig_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.AddEntry(eta_sig_1p3p_eff_w, "1p3p w")
    legend.AddEntry(eta_sig_1p1p_eff_w, "1p1p w")
    legend.AddEntry(eta_sig_3p3p_eff_w, "3p3p w")
    legend.AddEntry(eta_sig_inc_eff_w, "inclusive w")
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.25, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    mu_sig_1p3p_eff_w, mu_sig_1p1p_eff_w, mu_sig_3p3p_eff_w, mu_sig_inc_eff_w = plot_eff(sig_mu_list, sig_w_list, "Signal mu", 20, 18, 74, eta=False)
    mu_sig_1p3p_eff_w.SetMarkerStyle(20)
    mu_sig_1p1p_eff_w.SetMarkerStyle(20)
    mu_sig_3p3p_eff_w.SetMarkerStyle(20)
    mu_sig_inc_eff_w.SetMarkerStyle(20)
    mu_sig_1p3p_eff_w.SetMarkerSize(0.6)
    mu_sig_1p1p_eff_w.SetMarkerSize(0.6)
    mu_sig_3p3p_eff_w.SetMarkerSize(0.6)
    mu_sig_inc_eff_w.SetMarkerSize(0.6)
    mu_sig_1p3p_eff_w.Draw(" e")
    mu_sig_1p1p_eff_w.Draw("same e")
    mu_sig_3p3p_eff_w.Draw("same e")
    mu_sig_inc_eff_w.Draw("same e")
    legend = ROOT.TLegend(0.8, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.AddEntry(mu_sig_1p3p_eff_w, "1p3p w")
    legend.AddEntry(mu_sig_1p1p_eff_w, "1p1p w")
    legend.AddEntry(mu_sig_3p3p_eff_w, "3p3p w")
    legend.AddEntry(mu_sig_inc_eff_w, "inclusive w")
    legend.Draw()
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.DrawLatex(0.2+0.02,0.25, "#bf{#it{ATLAS}} Preliminary")
    canvas.Print("eff_plots.pdf")
    canvas.Clear()

    bk_pt_plt = plt_to_root_hist_w(combined_bkg_1p3p['ditau_pt'], 100, 200000, 1000000, combined_bkg_1p3p['event_weight'], False)
    bk_pt_plt.SetMarkerStyle(20)
    bk_pt_plt.SetMarkerSize(0.1)
    bk_pt_plt.Draw("hist e")
    ROOT.gPad.SetLogy()
    canvas.Print("eff_plots.pdf")
    canvas.Clear()


    canvas.Print("eff_plots.pdf]")







if __name__ == "__main__":
    plotter()
