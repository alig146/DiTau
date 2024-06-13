import uproot, glob, os
import numpy as np
import awkward as ak
from concurrent.futures import ProcessPoolExecutor
import h5py

branches = [
    'mcEventNumber', 'ditau_BDTScore', 'ditau_BDTScoreNew', 'ditau_ditau_pt',
    'ditau_n_subjets', 'mcEventWeights', 'ditau_IsTruthHadronic', 'ditau_n_tracks_lead',
    'ditau_n_tracks_subl', 'ditau_R_max_lead', 'ditau_R_max_subl', 'ditau_R_tracks_subl',
    'ditau_R_isotrack', 'ditau_d0_leadtrack_lead', 'ditau_d0_leadtrack_subl',
    'ditau_f_core_lead', 'ditau_f_core_subl', 'ditau_f_subjet_subl', 'ditau_f_subjets',
    'ditau_f_isotracks', 'ditau_m_core_lead', 'ditau_m_core_subl', 'ditau_m_tracks_lead',
    'ditau_m_tracks_subl', 'ditau_n_track', 'averageInteractionsPerCrossing', 'ditau_eta'
]

path = '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag'
file_paths = [
    path+'/*/*/user.agarabag.37404064*.root',
    path+'/*/*/user.agarabag.37404066*.root',
    path+'/*/*/user.agarabag.37449714*.root',
    path+'/*/*/user.agarabag.37404075*.root',
    path+'/*/*/user.agarabag.37404080*.root',
    path+'/*/*/user.agarabag.37404088*.root',
    path+'/*/*/user.agarabag.37404090*.root',
    path+'/*/*/user.agarabag.37404093*.root',
    path+'/*/*/user.agarabag.37404096*.root',
    path+'/*/*/user.agarabag.37404098*.root',
    path+'/*/*/user.agarabag.37404100*.root',
    path+'/*/*/user.agarabag.37404103*.root'
]

def process_file(file):
    data = {
        'ditau_subjet_pt': [], 'event_id': [], 'eta': [], 'average_mu': [], 'bdt_score': [],
        'bdt_score_new': [], 'ditau_pt': [], 'n_subjets': [], 'IsTruthHadronic': [],
        'n_tracks_lead': [], 'n_tracks_subl': [], 'R_max_lead': [], 'R_max_subl': [],
        'R_tracks_subl': [], 'R_isotrack': [], 'd0_leadtrack_lead': [], 'd0_leadtrack_subl': [],
        'f_core_lead': [], 'f_core_subl': [], 'f_subjet_subl': [], 'f_subjets': [],
        'f_isotracks': [], 'm_core_lead': [], 'm_core_subl': [], 'm_tracks_lead': [],
        'm_tracks_subl': [], 'n_track': [], 'event_weight': [], 'event_weight_sum': []
    }
    
    try:
        with uproot.open(file+':CollectionTree') as f_1:
            events = f_1.arrays(branches, library='ak')
            event_weights = events['mcEventWeights'][:, 0]

            flatten_event_weight = np.repeat(ak.firsts(event_weights), ak.num(events['ditau_ditau_pt']))
            flatten_event_id = np.repeat(events['mcEventNumber'], ak.num(events['ditau_ditau_pt']))
            flatten_avg_mu = np.repeat(events['averageInteractionsPerCrossing'], ak.num(events['ditau_ditau_pt']))

            data['event_id'].append(flatten_event_id)
            data['average_mu'].append(flatten_avg_mu)
            data['eta'].append(ak.flatten(events['ditau_eta']))
            data['bdt_score'].append(ak.flatten(events['ditau_BDTScore']))
            data['bdt_score_new'].append(ak.flatten(events['ditau_BDTScoreNew']))
            data['ditau_pt'].append(ak.flatten(events['ditau_ditau_pt']))
            data['n_subjets'].append(ak.flatten(events['ditau_n_subjets']))
            data['IsTruthHadronic'].append(ak.flatten(events['ditau_IsTruthHadronic']))
            data['n_tracks_lead'].append(ak.flatten(events['ditau_n_tracks_lead']))
            data['n_tracks_subl'].append(ak.flatten(events['ditau_n_tracks_subl']))
            data['R_max_lead'].append(ak.flatten(events['ditau_R_max_lead']))
            data['R_max_subl'].append(ak.flatten(events['ditau_R_max_subl']))
            data['R_tracks_subl'].append(ak.flatten(events['ditau_R_tracks_subl']))
            data['R_isotrack'].append(ak.flatten(events['ditau_R_isotrack']))
            data['d0_leadtrack_lead'].append(ak.flatten(events['ditau_d0_leadtrack_lead']))
            data['d0_leadtrack_subl'].append(ak.flatten(events['ditau_d0_leadtrack_subl']))
            data['f_core_lead'].append(ak.flatten(events['ditau_f_core_lead']))
            data['f_core_subl'].append(ak.flatten(events['ditau_f_core_subl']))
            data['f_subjet_subl'].append(ak.flatten(events['ditau_f_subjet_subl']))
            data['f_subjets'].append(ak.flatten(events['ditau_f_subjets']))
            data['f_isotracks'].append(ak.flatten(events['ditau_f_isotracks']))
            data['m_core_lead'].append(ak.flatten(events['ditau_m_core_lead']))
            data['m_core_subl'].append(ak.flatten(events['ditau_m_core_subl']))
            data['m_tracks_lead'].append(ak.flatten(events['ditau_m_tracks_lead']))
            data['m_tracks_subl'].append(ak.flatten(events['ditau_m_tracks_subl']))
            data['n_track'].append(ak.flatten(events['ditau_n_track']))
            data['event_weight'].append(flatten_event_weight)
            data['event_weight_sum'].append(ak.sum(ak.firsts(event_weights)))

            try:
                events_nested = f_1['ditau_subjet_pt'].array(library='ak')
                for pt_list in events_nested:
                    flat_pt_list = ak.flatten(pt_list)
                    data['ditau_subjet_pt'].append(flat_pt_list.tolist() if len(flat_pt_list) <= 6 else [0])
            except Exception as e:
                print(f"Error reading ditau_subjet_pt in file {file}")
                data['ditau_subjet_pt'].append([0])
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    
    return data

def merge_data(all_data, new_data):
    for key in all_data:
        if key != 'event_weight_sum':
            all_data[key].extend(new_data[key])
    return all_data

# Initialize empty data dictionary
all_data = {
    'ditau_subjet_pt': [], 'event_id': [], 'eta': [], 'average_mu': [], 'bdt_score': [],
    'bdt_score_new': [], 'ditau_pt': [], 'n_subjets': [], 'IsTruthHadronic': [],
    'n_tracks_lead': [], 'n_tracks_subl': [], 'R_max_lead': [], 'R_max_subl': [],
    'R_tracks_subl': [], 'R_isotrack': [], 'd0_leadtrack_lead': [], 'd0_leadtrack_subl': [],
    'f_core_lead': [], 'f_core_subl': [], 'f_subjet_subl': [], 'f_subjets': [],
    'f_isotracks': [], 'm_core_lead': [], 'm_core_subl': [], 'm_tracks_lead': [],
    'm_tracks_subl': [], 'n_track': [], 'event_weight': [], 'event_weight_sum': []
}

# Collect all file paths
all_files = [file for pattern in file_paths for file in glob.glob(pattern)]
print("Total files to process:", len(all_files))

# Process files in parallel
with ProcessPoolExecutor() as executor:
    for result in executor.map(process_file, all_files):
        all_data = merge_data(all_data, result)

# Flatten and concatenate all arrays
for key in all_data:
    if key != 'event_weight_sum':
        all_data[key] = ak.to_numpy(ak.concatenate(all_data[key]))

max_length = max(len(sublist) for sublist in all_data['ditau_subjet_pt'])
padded_ditau_subjet_pt = np.array([np.pad(sublist, (0, max_length - len(sublist)), 'constant') for sublist in all_data['ditau_subjet_pt']])

# Create an H5 file and save the datasets
with h5py.File(f'/global/homes/a/agarabag/pscratch/ditdau_samples/ntuple_flattened_v2.h5', 'w') as h5_file:
    h5_file.create_dataset('ditau_subjet_pt', data=padded_ditau_subjet_pt, compression='gzip')
    for key in all_data:
        if key != 'ditau_subjet_pt':
            h5_file.create_dataset(key, data=all_data[key], compression='gzip')

print("All done!")
