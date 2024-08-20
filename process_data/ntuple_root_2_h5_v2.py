import uproot, ROOT, glob, os, random
import numpy as np
import awkward as ak
from tqdm import tqdm
import h5py

            
branches = ['d0TJVA', 'z0TJVA', 'trackPt', 'trackEta', 'trackPhi', 'numberOfInnermostPixelLayerHits', 'numberOfPixelHits', 'numberOfSCTHits', 'qOverP', 
            'mcEventNumber',
            'ditau_BDTScore',
            'ditau_BDTScoreNew',
            'ditau_ditau_pt',
            'lead_subjet_pt',
            'sublead_subjet_pt',
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

path = '/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn_no_cuts'
file_paths = [
path+'/user.agarabag*JZ0*.root',
path+'/user.agarabag*JZ1*.root',
path+'/user.agarabag*JZ2*.root',
path+'/user.agarabag*JZ3*.root',
path+'/user.agarabag*JZ4*.root',
path+'/user.agarabag*JZ5*.root',
path+'/user.agarabag*JZ6*.root',
path+'/user.agarabag*JZ7*.root',
path+'/user.agarabag*JZ8*.root',
path+'/user.agarabag*JZ9incl*.root']

# file_paths = [path+'/user.agarabag.VHTauTauNTuple.802168.Py8EG_A14NNPDF23LO_VHtautau_flatmasspTFilt_hadhad_v2_ntuple.root']

for index in range(len(file_paths)):
    # l1 = glob.glob(os.path.join(file_paths[index]))
    l1 = glob.glob(os.path.join(file_paths[index], '*.root'))
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
    # ditau_subjet_pt = []
    lead_subjet_pt = []
    sublead_subjet_pt = []

    #track variables
    trackPt = []
    trackEta = []
    trackPhi = []
    numberOfInnermostPixelLayerHits = []
    numberOfPixelHits = []
    numberOfSCTHits = []
    qOverP = []
    d0TJVA = []
    z0TJVA = []

    for i in range(len(l1)):
        f_1 = uproot.open(l1[i]+':CollectionTree')
        
        events = f_1.arrays(branches, library='ak')

        # try:
        #     events_nested = f_1.arrays(['ditau_subjet_pt'], library='ak')
        #     for pt_list in events_nested['ditau_subjet_pt']:
        #         flat_pt_list = ak.flatten(pt_list)
        #         if len(flat_pt_list) > 6:
        #             ditau_subjet_pt.append([0])
        #         else:
        #             ditau_subjet_pt.append(flat_pt_list.tolist())
        # except Exception as e:
        #     print(f"Error reading ditau_subjet_pt in file {l1[i]}")
        #     ditau_subjet_pt.append([0])
        
        # print(type(events['mcEventNumber']), len(events['mcEventNumber']))
        # non_empty_mask = ak.num(events['ditau_pt']) > 0
        # print(type(events['ditau_pt'][non_empty_mask]), len(events['ditau_pt'][non_empty_mask]))

        event_weights = events['mcEventWeights'][:, 0, :]

        flatten_event_weight = np.repeat(ak.firsts(event_weights), ak.num(events['ditau_ditau_pt']))
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
        lead_subjet_pt.append(ak.flatten(events['lead_subjet_pt']))
        sublead_subjet_pt.append(ak.flatten(events['sublead_subjet_pt']))

        #track variables
        trackPt.append(ak.flatten(events['trackPt']))
        trackEta.append(ak.flatten(events['trackEta']))
        trackPhi.append(ak.flatten(events['trackPhi']))
        numberOfInnermostPixelLayerHits.append(ak.flatten(events['numberOfInnermostPixelLayerHits']))
        numberOfPixelHits.append(ak.flatten(events['numberOfPixelHits']))
        numberOfSCTHits.append(ak.flatten(events['numberOfSCTHits']))
        qOverP.append(ak.flatten(events['qOverP']))
        d0TJVA.append(ak.flatten(events['d0TJVA']))
        z0TJVA.append(ak.flatten(events['z0TJVA']))

    # max_length = max(len(sublist) for sublist in ditau_subjet_pt)
    # padded_ditau_subjet_pt = np.array([np.pad(sublist, (0, max_length - len(sublist)), 'constant') for sublist in ditau_subjet_pt])

    # Create an H5 file
    h5_file = h5py.File(f'/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn_no_cuts/ntuple_flattened_v2_jz{index}.h5', 'w')
    # h5_file = h5py.File(f'/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn_no_cuts/ntuple_flattened_VHtautau.h5', 'w')

    # Create datasets in the H5 file
    # h5_file.create_dataset('ditau_subjet_pt', data=padded_ditau_subjet_pt, compression='gzip')
    h5_file.create_dataset('lead_subjet_pt', data=ak.to_numpy(ak.concatenate(lead_subjet_pt)), compression='gzip')
    h5_file.create_dataset('sublead_subjet_pt', data=ak.to_numpy(ak.concatenate(sublead_subjet_pt)), compression='gzip')
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

    #track variables
    h5_file.create_dataset('trackPt', data=ak.to_numpy(ak.concatenate(trackPt)), compression='gzip')
    h5_file.create_dataset('trackEta', data=ak.to_numpy(ak.concatenate(trackEta)), compression='gzip')
    h5_file.create_dataset('trackPhi', data=ak.to_numpy(ak.concatenate(trackPhi)), compression='gzip')
    h5_file.create_dataset('numberOfInnermostPixelLayerHits', data=ak.to_numpy(ak.concatenate(numberOfInnermostPixelLayerHits)), compression='gzip')
    h5_file.create_dataset('numberOfPixelHits', data=ak.to_numpy(ak.concatenate(numberOfPixelHits)), compression='gzip')
    h5_file.create_dataset('numberOfSCTHits', data=ak.to_numpy(ak.concatenate(numberOfSCTHits)), compression='gzip')
    h5_file.create_dataset('qOverP', data=ak.to_numpy(ak.concatenate(qOverP)), compression='gzip')
    h5_file.create_dataset('d0TJVA', data=ak.to_numpy(ak.concatenate(d0TJVA)), compression='gzip')
    h5_file.create_dataset('z0TJVA', data=ak.to_numpy(ak.concatenate(z0TJVA)), compression='gzip')

    # Close the H5 file
    h5_file.close()
    

