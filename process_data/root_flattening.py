import uproot, ROOT, glob, os, random
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from sklearn.metrics import roc_curve, roc_auc_score
from joblib import dump, load
from tqdm import tqdm


branches = ['EventInfoAux.eventNumber',
            'DiTauJetsAuxDyn.BDTScore',
            'DiTauJetsAuxDyn.ditau_pt',
            'DiTauJetsAuxDyn.n_subjets',
            'EventInfoAuxDyn.mcEventWeights',
            'DiTauJetsAuxDyn.IsTruthHadronic',
            'DiTauJetsAuxDyn.n_tracks_lead',
            'DiTauJetsAuxDyn.n_tracks_subl',
            'DiTauJetsAuxDyn.R_max_lead', 'DiTauJetsAuxDyn.R_max_subl',
            'DiTauJetsAuxDyn.R_tracks_subl', 'DiTauJetsAuxDyn.R_isotrack', 
            'DiTauJetsAuxDyn.d0_leadtrack_lead', 'DiTauJetsAuxDyn.d0_leadtrack_subl',
            'DiTauJetsAuxDyn.f_core_lead', 'DiTauJetsAuxDyn.f_core_subl', 'DiTauJetsAuxDyn.f_subjet_subl',
            'DiTauJetsAuxDyn.f_subjets', 'DiTauJetsAuxDyn.f_isotracks', 
            'DiTauJetsAuxDyn.m_core_lead', 'DiTauJetsAuxDyn.m_core_subl', 
            'DiTauJetsAuxDyn.m_tracks_lead', 'DiTauJetsAuxDyn.m_tracks_subl', 'DiTauJetsAuxDyn.n_track']

branch_names = [
    "eventNumber",
    "BDTScore",
    "ditau_pt",
    "n_subjets",
    "IsTruthHadronic",
    "n_tracks_lead",
    "n_tracks_subl",
    "R_max_lead",
    "R_max_subl",
    "R_tracks_subl",
    "R_isotrack",
    "d0_leadtrack_lead",
    "d0_leadtrack_subl",
    "f_core_lead",
    "f_core_subl",
    "f_subjet_subl",
    "f_subjets",
    "f_isotracks",
    "m_core_lead",
    "m_core_subl",
    "m_tracks_lead",
    "m_tracks_subl",
    "n_track",
    "event_weight",
]
branch_types = [
    np.int32,    # event_id
    np.float32,  # bdt_score
    np.float32,  # ditau_pt
    np.int32,    # n_subjets
    np.int32,    # IsTruthHadronic
    np.int32,    # n_tracks_lead
    np.int32,    # n_tracks_subl
    np.float32,  # R_max_lead
    np.float32,  # R_max_subl
    np.float32,  # R_tracks_subl
    np.float32,  # R_isotrack
    np.float32,  # d0_leadtrack_lead
    np.float32,  # d0_leadtrack_subl
    np.float32,  # f_core_lead
    np.float32,  # f_core_subl
    np.float32,  # f_subjet_subl
    np.float32,  # f_subjets
    np.float32,  # f_isotracks
    np.float32,  # m_core_lead
    np.float32,  # m_core_subl
    np.float32,  # m_tracks_lead
    np.float32,  # m_tracks_subl
    np.int32,    # n_track
    np.float32,  # event_weight
]


path = '/global/homes/a/agarabag/pscratch/ditdau_samples/'
file_paths =[path+'user.agarabag.DiJetMC20_JZ0.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_v0_output.root/user.*.output.root',
         path+'user.agarabag.DiTauMC20.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_v0_output.root/user.*.output.root']

for index in range(len(file_paths)):
    l1 = glob.glob(os.path.join(file_paths[index]))

    ditau_pt = []
    n_subjets = []
    IsTruthHadronic = []
    n_tracks_lead = []
    n_tracks_subl = []
    R_max_lead = []
    R_max_subl = []
    R_tracks_subl = []
    R_isotrack = []
    d0_leadtrack_lead = []
    d0_leadtrack_subl = []
    f_core_lead = []
    f_core_subl = []
    f_subjet_subl = []
    f_subjets = []
    f_isotracks = []
    m_core_lead = []
    m_core_subl = []
    m_tracks_lead = []
    m_tracks_subl = []
    n_track = []
    event_weight = []
    event_weight_sum = []
    bdt_score = []
    event_id = []
    
    for i in range(len(l1)):
        print("processing: ", l1[i])
        
        f_1 = uproot.open(l1[i]+':CollectionTree')
        events = f_1.arrays(branches, library='ak')

        flatten_event_weight = np.repeat(ak.firsts(events['EventInfoAuxDyn.mcEventWeights']), ak.num(events['DiTauJetsAuxDyn.ditau_pt']))
        flatten_event_id = np.repeat(events['EventInfoAux.eventNumber'], ak.num(events['DiTauJetsAuxDyn.ditau_pt']))
        event_id.append(flatten_event_id)
        bdt_score.append(ak.flatten(events['DiTauJetsAuxDyn.BDTScore']))
        ditau_pt.append(ak.flatten(events['DiTauJetsAuxDyn.ditau_pt']))
        n_subjets.append(ak.flatten(events['DiTauJetsAuxDyn.n_subjets']))
        IsTruthHadronic.append(ak.flatten(events['DiTauJetsAuxDyn.IsTruthHadronic']))
        n_tracks_lead.append(ak.flatten(events['DiTauJetsAuxDyn.n_tracks_lead']))
        n_tracks_subl.append(ak.flatten(events['DiTauJetsAuxDyn.n_tracks_subl']))
        R_max_lead.append(ak.flatten(events['DiTauJetsAuxDyn.R_max_lead']))
        R_max_subl.append(ak.flatten(events['DiTauJetsAuxDyn.R_max_subl']))
        R_tracks_subl.append(ak.flatten(events['DiTauJetsAuxDyn.R_tracks_subl']))
        R_isotrack.append(ak.flatten(events['DiTauJetsAuxDyn.R_isotrack']))
        d0_leadtrack_lead.append(ak.flatten(events['DiTauJetsAuxDyn.d0_leadtrack_lead']))
        d0_leadtrack_subl.append(ak.flatten(events['DiTauJetsAuxDyn.d0_leadtrack_subl']))
        f_core_lead.append(ak.flatten(events['DiTauJetsAuxDyn.f_core_lead']))
        f_core_subl.append(ak.flatten(events['DiTauJetsAuxDyn.f_core_subl']))
        f_subjet_subl.append(ak.flatten(events['DiTauJetsAuxDyn.f_subjet_subl']))
        f_subjets.append(ak.flatten(events['DiTauJetsAuxDyn.f_subjets']))
        f_isotracks.append(ak.flatten(events['DiTauJetsAuxDyn.f_isotracks']))
        m_core_lead.append(ak.flatten(events['DiTauJetsAuxDyn.m_core_lead']))
        m_core_subl.append(ak.flatten(events['DiTauJetsAuxDyn.m_core_subl']))
        m_tracks_lead.append(ak.flatten(events['DiTauJetsAuxDyn.m_tracks_lead']))
        m_tracks_subl.append(ak.flatten(events['DiTauJetsAuxDyn.m_tracks_subl']))
        n_track.append(ak.flatten(events['DiTauJetsAuxDyn.n_track']))
        event_weight.append(flatten_event_weight)
        event_weight_sum.append(ak.sum(ak.firsts(events['EventInfoAuxDyn.mcEventWeights'])))

    # tree = uproot.mktree({branch_name: branch_type for branch_name, branch_type in zip(branch_names, branch_types)})
    # tree.extend(
    #     event_id=ak.to_numpy(ak.concatenate(event_id)),
    #     bdt_score=ak.to_numpy(ak.concatenate(bdt_score)),
    #     ditau_pt=ak.to_numpy(ak.concatenate(ditau_pt)),
    #     n_subjets=ak.to_numpy(ak.concatenate(n_subjets)),
    #     IsTruthHadronic=ak.to_numpy(ak.concatenate(IsTruthHadronic)),
    #     n_tracks_lead=ak.to_numpy(ak.concatenate(n_tracks_lead)),
    #     n_tracks_subl=ak.to_numpy(ak.concatenate(n_tracks_subl)),
    #     R_max_lead=ak.to_numpy(ak.concatenate(R_max_lead)),
    #     R_max_subl=ak.to_numpy(ak.concatenate(R_max_subl)),
    #     R_tracks_subl=ak.to_numpy(ak.concatenate(R_tracks_subl)),
    #     R_isotrack=ak.to_numpy(ak.concatenate(R_isotrack)),
    #     d0_leadtrack_lead=ak.to_numpy(ak.concatenate(d0_leadtrack_lead)),
    #     d0_leadtrack_subl=ak.to_numpy(ak.concatenate(d0_leadtrack_subl)),
    #     f_core_lead=ak.to_numpy(ak.concatenate(f_core_lead)),
    #     f_core_subl=ak.to_numpy(ak.concatenate(f_core_subl)),
    #     f_subjet_subl=ak.to_numpy(ak.concatenate(f_subjet_subl)),
    #     f_subjets=ak.to_numpy(ak.concatenate(f_subjets)),
    #     f_isotracks=ak.to_numpy(ak.concatenate(f_isotracks)),
    #     m_core_lead=ak.to_numpy(ak.concatenate(m_core_lead)),
    #     m_core_subl=ak.to_numpy(ak.concatenate(m_core_subl)),
    #     m_tracks_lead=ak.to_numpy(ak.concatenate(m_tracks_lead)),
    #     m_tracks_subl=ak.to_numpy(ak.concatenate(m_tracks_subl)),
    #     n_track=ak.to_numpy(ak.concatenate(n_track)),
    #     event_weight=ak.to_numpy(ak.concatenate(event_weight) / ak.sum(event_weight_sum))
    # )
    
    # tree.write()
    # file.close()

    file = uproot.recreate("ditau_unskimmed"+str(index)+"_ak.root")
    file['CollectionTree'] = {"eventNumber": ak.to_numpy(ak.concatenate(event_id)),
                              "BDTScore": ak.to_numpy(ak.concatenate(bdt_score)),
                              "ditau_pt": ak.to_numpy(ak.concatenate(ditau_pt)),
                              "n_subjets": ak.to_numpy(ak.concatenate(n_subjets)),
                              "IsTruthHadronic": ak.to_numpy(ak.concatenate(IsTruthHadronic)),
                              "n_tracks_lead": ak.to_numpy(ak.concatenate(n_tracks_lead)),
                              "n_tracks_subl": ak.to_numpy(ak.concatenate(n_tracks_subl)),
                              "R_max_lead": ak.to_numpy(ak.concatenate(R_max_lead)),
                              "R_max_subl": ak.to_numpy(ak.concatenate(R_max_subl)),
                              "R_tracks_subl": ak.to_numpy(ak.concatenate(R_tracks_subl)),
                              "R_isotrack": ak.to_numpy(ak.concatenate(R_isotrack)),
                              "d0_leadtrack_lead": ak.to_numpy(ak.concatenate(d0_leadtrack_lead)),
                              "d0_leadtrack_subl": ak.to_numpy(ak.concatenate(d0_leadtrack_subl)),
                              "f_core_lead": ak.to_numpy(ak.concatenate(f_core_lead)),
                              "f_core_subl": ak.to_numpy(ak.concatenate(f_core_subl)),
                              "f_subjet_subl": ak.to_numpy(ak.concatenate(f_subjet_subl)),
                              "f_subjets": ak.to_numpy(ak.concatenate(f_subjets)),
                              "f_isotracks": ak.to_numpy(ak.concatenate(f_isotracks)),
                              "m_core_lead": ak.to_numpy(ak.concatenate(m_core_lead)),
                              "m_core_subl": ak.to_numpy(ak.concatenate(m_core_subl)),
                              "m_tracks_lead": ak.to_numpy(ak.concatenate(m_tracks_lead)),
                              "m_tracks_subl": ak.to_numpy(ak.concatenate(m_tracks_subl)),
                              "n_track": ak.to_numpy(ak.concatenate(n_track)),
                              "event_weight": ak.to_numpy(ak.concatenate(event_weight)/ak.sum(event_weight_sum))}
