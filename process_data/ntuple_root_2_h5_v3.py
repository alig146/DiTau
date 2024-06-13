import uproot, glob, os
import numpy as np
import awkward as ak
from tqdm import tqdm
import h5py
from concurrent.futures import ThreadPoolExecutor

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

path = '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag'
file_patterns = [
    'user.agarabag.37404064*.root',
    'user.agarabag.37404066*.root',
    'user.agarabag.37449714*.root',
    'user.agarabag.37404075*.root',
    'user.agarabag.37404080*.root',
    'user.agarabag.37404088*.root',
    'user.agarabag.37404090*.root',
    'user.agarabag.37404093*.root',
    'user.agarabag.37404096*.root',
    'user.agarabag.37404098*.root',   
    'user.agarabag.37404100*.root',
    'user.agarabag.37404103*.root'
]

file_paths = []
for pattern in file_patterns:
    file_paths.extend(glob.glob(os.path.join(path, '**', pattern), recursive=True))

def process_file(file_path):
    try:
        with uproot.open(file_path + ':CollectionTree') as f_1:
            events = f_1.arrays(branches, library='ak')

            # Handle ditau_subjet_pt separately
            try:
                events_nested = f_1.arrays(['ditau_subjet_pt'], library='ak')
                ditau_subjet_pt = [ak.flatten(pt_list).tolist() if len(ak.flatten(pt_list)) <= 6 else [0] for pt_list in events_nested['ditau_subjet_pt']]
            except Exception:
                ditau_subjet_pt = [[0]]
            
            # Flatten necessary fields
            flatten_event_weight = np.repeat(ak.firsts(events['mcEventWeights']), ak.num(events['ditau_ditau_pt']))
            flatten_event_id = np.repeat(events['mcEventNumber'], ak.num(events['ditau_ditau_pt']))
            flatten_avg_mu = np.repeat(events['averageInteractionsPerCrossing'], ak.num(events['ditau_ditau_pt']))

            return {
                'ditau_subjet_pt': ditau_subjet_pt,
                'event_id': ak.to_numpy(flatten_event_id),
                'average_mu': ak.to_numpy(flatten_avg_mu),
                'eta': ak.to_numpy(ak.flatten(events['ditau_eta'])),
                'bdt_score': ak.to_numpy(ak.flatten(events['ditau_BDTScore'])),
                'bdt_score_new': ak.to_numpy(ak.flatten(events['ditau_BDTScoreNew'])),
                'ditau_pt': ak.to_numpy(ak.flatten(events['ditau_ditau_pt'])),
                'n_subjets': ak.to_numpy(ak.flatten(events['ditau_n_subjets'])),
                'IsTruthHadronic': ak.to_numpy(ak.flatten(events['ditau_IsTruthHadronic'])),
                'n_tracks_lead': ak.to_numpy(ak.flatten(events['ditau_n_tracks_lead'])),
                'n_tracks_subl': ak.to_numpy(ak.flatten(events['ditau_n_tracks_subl'])),
                'R_max_lead': ak.to_numpy(ak.flatten(events['ditau_R_max_lead'])),
                'R_max_subl': ak.to_numpy(ak.flatten(events['ditau_R_max_subl'])),
                'R_tracks_subl': ak.to_numpy(ak.flatten(events['ditau_R_tracks_subl'])),
                'R_isotrack': ak.to_numpy(ak.flatten(events['ditau_R_isotrack'])),
                'd0_leadtrack_lead': ak.to_numpy(ak.flatten(events['ditau_d0_leadtrack_lead'])),
                'd0_leadtrack_subl': ak.to_numpy(ak.flatten(events['ditau_d0_leadtrack_subl'])),
                'f_core_lead': ak.to_numpy(ak.flatten(events['ditau_f_core_lead'])),
                'f_core_subl': ak.to_numpy(ak.flatten(events['ditau_f_core_subl'])),
                'f_subjet_subl': ak.to_numpy(ak.flatten(events['ditau_f_subjet_subl'])),
                'f_subjets': ak.to_numpy(ak.flatten(events['ditau_f_subjets'])),
                'f_isotracks': ak.to_numpy(ak.flatten(events['ditau_f_isotracks'])),
                'm_core_lead': ak.to_numpy(ak.flatten(events['ditau_m_core_lead'])),
                'm_core_subl': ak.to_numpy(ak.flatten(events['ditau_m_core_subl'])),
                'm_tracks_lead': ak.to_numpy(ak.flatten(events['ditau_m_tracks_lead'])),
                'm_tracks_subl': ak.to_numpy(ak.flatten(events['ditau_m_tracks_subl'])),
                'n_track': ak.to_numpy(ak.flatten(events['ditau_n_track'])),
                'event_weight': ak.to_numpy(flatten_event_weight),
                'event_weight_sum': ak.sum(ak.firsts(events['mcEventWeights']))
            }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main():
    all_results = {
        'ditau_subjet_pt': [],
        'event_id': [],
        'average_mu': [],
        'eta': [],
        'bdt_score': [],
        'bdt_score_new': [],
        'ditau_pt': [],
        'n_subjets': [],
        'IsTruthHadronic': [],
        'n_tracks_lead': [],
        'n_tracks_subl': [],
        'R_max_lead': [],
        'R_max_subl': [],
        'R_tracks_subl': [],
        'R_isotrack': [],
        'd0_leadtrack_lead': [],
        'd0_leadtrack_subl': [],
        'f_core_lead': [],
        'f_core_subl': [],
        'f_subjet_subl': [],
        'f_subjets': [],
        'f_isotracks': [],
        'm_core_lead': [],
        'm_core_subl': [],
        'm_tracks_lead': [],
        'm_tracks_subl': [],
        'n_track': [],
        'event_weight': [],
        'event_weight_sum': []
    }

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))

    for result in results:
        if result:
            for key in all_results:
                all_results[key].extend(result[key])

    max_length = max(len(sublist) for sublist in all_results['ditau_subjet_pt'])
    padded_ditau_subjet_pt = np.array([np.pad(sublist, (0, max_length - len(sublist)), 'constant') for sublist in all_results['ditau_subjet_pt']])

    # Create an H5 file
    with h5py.File('/global/homes/a/agarabag/pscratch/ditdau_samples/ntuple_flattened_v2.h5', 'w') as h5_file:
        # Create datasets in the H5 file
        h5_file.create_dataset('ditau_subjet_pt', data=padded_ditau_subjet_pt, compression='gzip')
        for key in all_results:
            if key != 'ditau_subjet_pt':
                h5_file.create_dataset(key, data=np.array(all_results[key]), compression='gzip')
        event_weight = np.array(all_results['event_weight'])
        event_weight_sum = np.array(all_results['event_weight_sum'])
        h5_file.create_dataset('event_weight', data=event_weight / np.sum(event_weight_sum), compression='gzip')

if __name__ == "__main__":
    main()
