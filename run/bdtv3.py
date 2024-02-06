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
from xgboost import XGBClassifier
sys.path.append("..")
from utils.utils import *

branches = ['ditau_pt',
 'n_subjets',
 'IsTruthHadronic',
 'n_tracks_lead',
 'n_tracks_subl',
 'R_max_lead',
 'R_max_subl',
 'R_tracks_subl',
 'R_isotrack',
 'd0_leadtrack_lead',
 'd0_leadtrack_subl',
 'f_core_lead',
 'f_core_subl',
 'f_subjet_subl',
 'f_subjets',
 'f_isotracks',
 'm_core_lead',
 'm_core_subl',
 'm_tracks_lead',
 'm_tracks_subl',
 'n_track']

path = '/global/homes/a/agarabag/pscratch/ditdau_samples/'
ditau = uproot_open(path+f'ditau_skimmed0_11_v1.root:CollectionTree', branches.append(['event_weight']))
gamma = uproot_open(path+f'gamma_skimmed_ak.root:CollectionTree', branches.append(['event_weight/event_weight_sum']))
graviton = uproot_open(path+f'graviton_skimmed_ak_v1.root:CollectionTree', branches.append(['event_weight']))

graviton_evt_weights = graviton['event_weight']
gamma_evt_weights = gamma['event_weight/event_weight_sum']*getXS(425200)
bkg_evt_weights = ditau['event_weight']

# create signal and background pt weights
pt_bins = np.linspace(200000, 1000000, 41)

gamma_pt_weights = flattened_pt_weighted(gamma['ditau_pt'], pt_bins, gamma_evt_weights)
graviton_pt_weights = flattened_pt_weighted(graviton['ditau_pt'], pt_bins, graviton_evt_weights)
bkg_pt_weights = flattened_pt_weighted(ditau['ditau_pt'], pt_bins, bkg_evt_weights)

graviton_gamma = ak.concatenate([graviton, gamma])

trainig_vars = ["f_core_lead", 
                "f_core_subl", 
                "f_subjet_subl", 
                "f_subjets", 
                "f_isotracks", 
                "R_max_lead",
                "R_max_subl", 
                "R_isotrack", 
                "R_tracks_subl",
                "m_core_lead",
                "m_core_subl", 
                "m_tracks_lead",
                "m_tracks_subl",
                "d0_leadtrack_lead",
                "d0_leadtrack_subl", 
                "n_track", 
                "n_tracks_lead"]

graviton_gamma_train = graviton_gamma[trainig_vars]
jz_train = ditau[trainig_vars]

log_vars = ["f_isotracks", "m_core_lead", "m_core_subl", "m_tracks_lead", "m_tracks_subl"]
abs_log_var = ["d0_leadtrack_lead", "d0_leadtrack_subl"]

# create signal and background labels 
signal_labels = np.ones(len(graviton_gamma_train['n_tracks_lead']))
qcd_labels = np.zeros(len(jz_train['n_tracks_lead']))

comb_pt_weights = np.concatenate([graviton_pt_weights, gamma_pt_weights, bkg_pt_weights])
comb_evt_weights = np.concatenate([graviton_evt_weights, gamma_evt_weights, bkg_evt_weights])
labels = np.concatenate([signal_labels, qcd_labels])

graviton_gamma_pd = pd.DataFrame(columns=trainig_vars)
jz_pd = pd.DataFrame(columns=trainig_vars)
for i in tqdm(range(len(trainig_vars))):
    #print(graviton_gamma_train[trainig_vars[i]])
    graviton_gamma_pd[trainig_vars[i]] = graviton_gamma_train[trainig_vars[i]].tolist()
    jz_pd[trainig_vars[i]] = jz_train[trainig_vars[i]].tolist()

#combine signal and background dataframes
# weight_pd = pd.DataFrame(comb_pt_weights*(comb_evt_weights+1), columns=['weight'])
weight_pd = pd.DataFrame(comb_pt_weights*(comb_evt_weights), columns=['weight'])
labels_pd = pd.DataFrame(labels, columns=['label']) 

combined_column = pd.concat([graviton_gamma_pd, jz_pd], axis=0, ignore_index=True)
combined_data = pd.concat([combined_column, weight_pd, labels_pd], axis=1)

#log certain variables
# combined_data[log_vars] = np.log(combined_data[log_vars])
# combined_data = combined_data[~combined_data[log_vars].isin([np.inf, -np.inf]).any(axis=1)]
# combined_data[abs_log_var] = np.log(np.abs(combined_data[abs_log_var]))
# combined_data = combined_data[~combined_data[abs_log_var].isin([np.inf, -np.inf]).any(axis=1)]

#shuffle the data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

pd_data = combined_data[trainig_vars]
pd_weights = combined_data["weight"]
pd_labels = combined_data["label"]
print(pd_data)

# change column names to integers 
feature_mapping = {feature: i for i, feature in enumerate(pd_data.columns)}
pd_data.rename(columns=feature_mapping, inplace=True)

print(pd_data)

# split the data into training and testing sets
training_size = 0.8

training_data = pd_data[:int(len(pd_data)*training_size)]
training_labels = pd_labels[:int(len(pd_labels)*training_size)]
training_weights = pd_weights[:int(len(pd_weights)*training_size)]
print("training data: ", training_data.shape)
print("training labels: ", training_labels.shape)
print("training weights: ", training_weights.shape)

testing_data = pd_data[int(len(pd_data)*training_size):]
testing_labels = pd_labels[int(len(pd_labels)*training_size):]
testing_weights = pd_weights[int(len(pd_weights)*training_size):]
print("testing data: ", testing_data.shape)
print("testing labels: ", testing_labels.shape)
print("testing weights: ", testing_weights.shape)

# create the BDT    
params = {
'n_estimators': 150,
'learning_rate': 0.1,
'max_depth': 3,
'eval_metric': 'logloss',
'random_state': 0,
'gamma': 0.001,
'verbosity': 2
}
bdt = XGBClassifier(**params)
# print(type(bdt))

# Train the classifier
# bdt.fit(training_data, training_labels, sample_weight=training_weights)
bdt.fit(training_data, training_labels, eval_set=[(testing_data, testing_labels)], sample_weight=training_weights)

# print("CCCCC: ", bdt.classes_)

# Predict probabilities for the testing data
probs = bdt.predict_proba(testing_data)
probs = probs[:, 1]

fpr_w, tpr_w, thresholds_w = roc_curve(testing_labels, probs, sample_weight=testing_weights)
auc_w = roc_auc_score(testing_labels, probs, sample_weight=testing_weights)
plt.figure(figsize=(10, 10))
plt.plot(fpr_w, tpr_w, label='BDT (area = {:.5f})'.format(auc_w))
plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig("bdt_results/xgb_bdt_roc.png")
plt.clf()

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)
canvas = ROOT.TCanvas("canvas", "bdt_score_new", 800, 500)
canvas.cd()
canvas.Print("bdt_results/xgb_bdt.pdf[")
sig_score_1p3p = plt_to_root_hist_w(probs[testing_labels==1], 100, 0, 1, testing_weights[testing_labels==1], False)
bkg_score_1p3p = plt_to_root_hist_w(probs[testing_labels==0], 100, 0, 1, testing_weights[testing_labels==0], False)
sig_score_1p3p.SetLineColor(ROOT.kBlack)
bkg_score_1p3p.SetLineColor(ROOT.kOrange)
sig_score_1p3p.Draw("hist ")
bkg_score_1p3p.Draw("hist same")
ROOT.gPad.SetLogy()
canvas.Print("bdt_results/xgb_bdt.pdf")
canvas.Clear()

root_1p3p_roc = create_roc_graph(sig_score_1p3p, bkg_score_1p3p, effmin=0.05, name="1p3p", normalize=False, reverse=False)
root_1p3p_roc.Draw("")
ROOT.gPad.SetLogy()
canvas.Print("bdt_results/xgb_bdt.pdf")
canvas.Clear()

canvas.Print("bdt_results/xgb_bdt.pdf]")

bdt.save_model('bdt_results/xgb_model.json')