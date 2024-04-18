import uproot, ROOT, glob, os, random
import numpy as np
import awkward as ak
from tqdm import tqdm
import h5py

            
branches = ['mcEventNumber',
            'ditau_BDTScore',
            'ditau_BDTScoreNew',
            'ditau_ditau_pt',
            'ditau_n_subjets',
            'mcEventWeights',
            'ditau_IsTruthHadronic',
            'ditau_n_tracks_lead',
            'ditau_n_tracks_subl',
            'ditau_R_max_lead', 'ditau_R_max_subl',
            'ditau_R_tracks_subl', 'ditau_R_isotrack', 
            'ditau_d0_leadtrack_lead', 'ditau_d0_leadtrack_subl',
            'ditau_f_core_lead', 'ditau_f_core_subl', 'ditau_f_subjet_subl',
            'ditau_f_subjets', 'ditau_f_isotracks', 
            'ditau_m_core_lead', 'ditau_m_core_subl', 
            'ditau_m_tracks_lead', 'ditau_m_tracks_subl', 'ditau_n_track', 'averageInteractionsPerCrossing', 'ditau_eta']

# path = '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag'
# file_paths = [
# path+'/*/*/user.agarabag.37404064*.root',
# path+'/*/*/user.agarabag.37404066*.root',
# path+'/*/*/user.agarabag.37449714*.root',
# path+'/*/*/user.agarabag.37404075*.root',
# path+'/*/*/user.agarabag.37404080*.root',
# path+'/*/*/user.agarabag.37404088*.root',
# path+'/*/*/user.agarabag.37404090*.root',
# path+'/*/*/user.agarabag.37404093*.root',
# path+'/*/*/user.agarabag.37404096*.root',
# path+'/*/*/user.agarabag.37404098*.root',   
# path+'/*/*/user.agarabag.37404100*.root',
# path+'/*/*/user.agarabag.37404103*.root']

path = '/global/homes/a/agarabag/pscratch/ditdau_samples/user.agarabag.VHTauTau.802168.Py8EG_A14NNPDF23LO_VHtautau_flatmasspTFilt_hadhad_v1_ntuple.root/'
file_paths =[path+'user.*.ntuple.root']



for index in range(len(file_paths)):
    l1 = glob.glob(os.path.join(file_paths[index]))
    print("processing: ", l1)

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
    bdt_score_new = []
    event_id = []
    eta = []
    average_mu = []
    
    for i in range(len(l1)):
        f_1 = uproot.open(l1[i]+':CollectionTree')
        events = f_1.arrays(branches, library='ak')
        
        # print(events['ditau_pt'], len(events['ditau_pt']))
        # print(type(events['mcEventNumber']), len(events['mcEventNumber']))
        # non_empty_mask = ak.num(events['ditau_pt']) > 0
        # print(type(events['ditau_pt'][non_empty_mask]), len(events['ditau_pt'][non_empty_mask]))

        flatten_event_weight = np.repeat(ak.firsts(events['mcEventWeights']), ak.num(events['ditau_ditau_pt']))
        flatten_event_id = np.repeat(events['mcEventNumber'], ak.num(events['ditau_ditau_pt']))
        flatten_avg_mu = np.repeat(events['averageInteractionsPerCrossing'], ak.num(events['ditau_ditau_pt']))
        
        event_id.append(flatten_event_id)
        average_mu.append(flatten_avg_mu)
        eta.append(ak.flatten(events['ditau_eta']))
        bdt_score.append(ak.flatten(events['ditau_BDTScore']))
        bdt_score_new.append(ak.flatten(events['ditau_BDTScoreNew']))
        ditau_pt.append(ak.flatten(events['ditau_ditau_pt']))
        n_subjets.append(ak.flatten(events['ditau_n_subjets']))
        IsTruthHadronic.append(ak.flatten(events['ditau_IsTruthHadronic']))
        n_tracks_lead.append(ak.flatten(events['ditau_n_tracks_lead']))
        n_tracks_subl.append(ak.flatten(events['ditau_n_tracks_subl']))
        R_max_lead.append(ak.flatten(events['ditau_R_max_lead']))
        R_max_subl.append(ak.flatten(events['ditau_R_max_subl']))
        R_tracks_subl.append(ak.flatten(events['ditau_R_tracks_subl']))
        R_isotrack.append(ak.flatten(events['ditau_R_isotrack']))
        d0_leadtrack_lead.append(ak.flatten(events['ditau_d0_leadtrack_lead']))
        d0_leadtrack_subl.append(ak.flatten(events['ditau_d0_leadtrack_subl']))
        f_core_lead.append(ak.flatten(events['ditau_f_core_lead']))
        f_core_subl.append(ak.flatten(events['ditau_f_core_subl']))
        f_subjet_subl.append(ak.flatten(events['ditau_f_subjet_subl']))
        f_subjets.append(ak.flatten(events['ditau_f_subjets']))
        f_isotracks.append(ak.flatten(events['ditau_f_isotracks']))
        m_core_lead.append(ak.flatten(events['ditau_m_core_lead']))
        m_core_subl.append(ak.flatten(events['ditau_m_core_subl']))
        m_tracks_lead.append(ak.flatten(events['ditau_m_tracks_lead']))
        m_tracks_subl.append(ak.flatten(events['ditau_m_tracks_subl']))
        n_track.append(ak.flatten(events['ditau_n_track']))
        event_weight.append(flatten_event_weight)
        event_weight_sum.append(ak.sum(ak.firsts(events['mcEventWeights'])))
        

    # Create an H5 file
    h5_file = h5py.File(f'/global/homes/a/agarabag/pscratch/ditdau_samples/VHtautau_ntuple_flattened_jz{index+1}.h5', 'w')
    # Create datasets in the H5 file
    h5_file.create_dataset('event_id', data=ak.to_numpy(ak.concatenate(event_id)), compression='gzip')
    h5_file.create_dataset('eta', data=ak.to_numpy(ak.concatenate(eta)), compression='gzip')
    h5_file.create_dataset('average_mu', data=ak.to_numpy(ak.concatenate(average_mu)), compression='gzip')
    h5_file.create_dataset('bdt_score', data=ak.to_numpy(ak.concatenate(bdt_score)), compression='gzip')
    h5_file.create_dataset('bdt_score_new', data=ak.to_numpy(ak.concatenate(bdt_score_new)), compression='gzip')
    h5_file.create_dataset('ditau_pt', data=ak.to_numpy(ak.concatenate(ditau_pt)), compression='gzip')
    h5_file.create_dataset('n_subjets', data=ak.to_numpy(ak.concatenate(n_subjets)), compression='gzip')
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
    

