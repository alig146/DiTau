import uproot, ROOT, glob, os, random
import numpy as np
import awkward as ak
from tqdm import tqdm
import h5py

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


path = '/global/homes/a/agarabag/pscratch/ditdau_samples/'
file_paths = [
# path+'user.agarabag.G_hh_4tau_MC20.425101.MadGraphPythia8EvtGen_A14NNPDF23LO_RS_G_hh_4tau_c10_M1750_v0_output.root/user.*.output.root',
# path+'user.agarabag.G_hh_4tau_MC20.425104.MadGraphPythia8EvtGen_A14NNPDF23LO_RS_G_hh_4tau_c10_M2500_v0_output.root/user.*.output.root',
# path+'user.agarabag.G_hh_4tau_MC20.425107.MadGraphPythia8EvtGen_A14NNPDF23LO_RS_G_hh_4tau_c10_M5000_v0_output.root/user.*.output.root',
# path+'user.agarabag.G_hh_4tau_MC20.425108.MadGraphPythia8EvtGen_A14NNPDF23LO_RS_G_hh_4tau_c10_M1000_v0_output.root/user.*.output.root',
path+'MC20_Gammatautau.root']

# file_paths = [
#         path+'user.agarabag.DiJetMC20_JZ0.364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_v0_output.root/user.*.output.root',
#         path+'user.agarabag.DiTauMC20.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_v0_output.root/user.*.output.root',
#         path+'user.agarabag.DiTauMC20.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root/user.*.output.root']
         # path+'user.agarabag.DiTauMC20.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364709.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364710.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364711.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_v0_output.root/user.*.output.root',
         # path+'user.agarabag.DiTauMC20.364712.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_v0_output.root/user.*.output.root']

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
        
        # print(events['DiTauJetsAuxDyn.ditau_pt'], len(events['DiTauJetsAuxDyn.ditau_pt']))
        # print(type(events['EventInfoAux.eventNumber']), len(events['EventInfoAux.eventNumber']))
        # non_empty_mask = ak.num(events['DiTauJetsAuxDyn.ditau_pt']) > 0
        # print(type(events['DiTauJetsAuxDyn.ditau_pt'][non_empty_mask]), len(events['DiTauJetsAuxDyn.ditau_pt'][non_empty_mask]))

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
        

    # Create an H5 file
    h5_file = h5py.File(f'gamma_flattened_{index}.h5', 'w')
    # Create datasets in the H5 file
    h5_file.create_dataset('event_id', data=ak.to_numpy(ak.concatenate(event_id)), compression='gzip')
    h5_file.create_dataset('bdt_score', data=ak.to_numpy(ak.concatenate(bdt_score)), compression='gzip')
    h5_file.create_dataset('ditau_pt', data=ak.to_numpy(ak.concatenate(ditau_pt)), compression='gzip')
    h5_file.create_dataset('n_subjets', data=ak.to_numpy(ak.concatenate(n_subjets)), compression='gzip') ####was not compressed 
    h5_file.create_dataset('IsTruthHadronic', data=ak.to_numpy(ak.concatenate(IsTruthHadronic)), compression='gzip')
    h5_file.create_dataset('n_tracks_lead', data=ak.to_numpy(ak.concatenate(n_tracks_lead)), compression='gzip')
    h5_file.create_dataset('n_tracks_subl', data=ak.to_numpy(ak.concatenate(n_tracks_subl)), compression='gzip')
    h5_file.create_dataset('R_max_lead', data=ak.to_numpy(ak.concatenate(R_max_lead)), compression='gzip')
    h5_file.create_dataset('R_max_subl', data=ak.to_numpy(ak.concatenate(R_max_subl)), compression='gzip')
    h5_file.create_dataset('R_tracks_subl', data=ak.to_numpy(ak.concatenate(R_tracks_subl)), compression='gzip')
    h5_file.create_dataset('R_isotrack', data=ak.to_numpy(ak.concatenate(R_isotrack)), compression='gzip')
    h5_file.create_dataset('d0_leadtrack_lead', data=ak.to_numpy(ak.concatenate(d0_leadtrack_lead)), compression='gzip')
    h5_file.create_dataset('d0_leadtrack_subl', data=ak.to_numpy(ak.concatenate(d0_leadtrack_subl)), compression='gzip')
    h5_file.create_dataset('f_core_lead', data=ak.to_numpy(ak.concatenate(f_core_lead)), compression='gzip')
    h5_file.create_dataset('f_core_subl', data=ak.to_numpy(ak.concatenate(f_core_subl)), compression='gzip')
    h5_file.create_dataset('f_subjet_subl', data=ak.to_numpy(ak.concatenate(f_subjet_subl)), compression='gzip')
    h5_file.create_dataset('f_subjets', data=ak.to_numpy(ak.concatenate(f_subjets)), compression='gzip')
    h5_file.create_dataset('f_isotracks', data=ak.to_numpy(ak.concatenate(f_isotracks)), compression='gzip')
    h5_file.create_dataset('m_core_lead', data=ak.to_numpy(ak.concatenate(m_core_lead)), compression='gzip')
    h5_file.create_dataset('m_core_subl', data=ak.to_numpy(ak.concatenate(m_core_subl)), compression='gzip')
    h5_file.create_dataset('m_tracks_lead', data=ak.to_numpy(ak.concatenate(m_tracks_lead)), compression='gzip')
    h5_file.create_dataset('m_tracks_subl', data=ak.to_numpy(ak.concatenate(m_tracks_subl)), compression='gzip')
    h5_file.create_dataset('n_track', data=ak.to_numpy(ak.concatenate(n_track)), compression='gzip')
    h5_file.create_dataset('event_weight', data=ak.to_numpy(ak.concatenate(event_weight)/ak.sum(event_weight_sum)), compression='gzip')
    # Close the H5 file
    h5_file.close()
    

