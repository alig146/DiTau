import uproot
import glob
import os
import sys
import numpy as np
import awkward as ak
from tqdm import tqdm
import h5py
import multiprocessing
from functools import partial

sys.path.append("..")
from utils.utils import flattened_pt_weighted, getXS

# Define the branches to be extracted
branches = ['d0TJVA', 'z0TJVA', 'trackPt', 'trackEta', 'trackPhi', 'mcEventWeights', 'mcEventNumber', 'ditau_ditau_pt', 'ditau_eta', 'ditau_phi',
            'numberOfInnermostPixelLayerHits', 'numberOfPixelHits', 'numberOfSCTHits', 'charge',
            'ditau_R_max_lead', 'ditau_R_max_subl', 'ditau_R_tracks_subl', 'ditau_R_isotrack', 
            'ditau_d0_leadtrack_lead', 'ditau_d0_leadtrack_subl', 'ditau_f_core_lead', 
            'ditau_f_core_subl', 'ditau_f_subjet_subl', 'ditau_f_subjets', 'ditau_f_isotracks', 
            'ditau_m_core_lead', 'ditau_m_core_subl', 'ditau_m_tracks_lead', 'ditau_n_track']

path = '/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn_onnx'

bkg_file_patterns = ['/user.agarabag*JZ{}_v2*.root'.format(i) for i in range(10)]
signal_file_pattern = '/user.agarabag.VHTauTauOmni.802168.Py8EG_A14NNPDF23LO_VHtautau_flatmasspTFilt_hadhad_v18_ntuple.root'

file_patterns = bkg_file_patterns + [signal_file_pattern]
labels = [0] * len(bkg_file_patterns) + [1]
xs_ids = list(range(801165, 801175)) + [802168]

MAX_TRACKS = 10

def process_file(file, label, xs_id):
    try:
        f_1 = uproot.open(f"{file}:CollectionTree")
        if f_1.num_entries == 0:
            print(f"File {file} is empty, skipping...")
            return None

        events = f_1.arrays(branches, library='ak')
        
        event_weights_raw = ak.firsts(events['mcEventWeights'][:, 0, :])
        event_weight_sum = ak.sum(ak.firsts(events['mcEventWeights']))

        jet_vars = [events[var] for var in ['ditau_ditau_pt', 'ditau_eta', 'ditau_phi', 'mcEventNumber', 
                                            'ditau_R_max_lead', 'ditau_R_max_subl', 'ditau_R_tracks_subl', 'ditau_R_isotrack',
                                            'ditau_d0_leadtrack_lead', 'ditau_d0_leadtrack_subl', 'ditau_f_core_lead', 
                                            'ditau_f_core_subl', 'ditau_f_subjet_subl', 'ditau_f_subjets', 'ditau_f_isotracks', 
                                            'ditau_m_core_lead', 'ditau_m_core_subl', 'ditau_m_tracks_lead', 'ditau_n_track']]
        jet_vars = [ak.to_numpy(var) for var in jet_vars]
        padded_jets = np.column_stack(jet_vars)

        track_features = [events[var] for var in ['trackEta', 'trackPhi', 'trackPt', 'd0TJVA', 'z0TJVA',
                                                  'numberOfInnermostPixelLayerHits', 'numberOfPixelHits', 
                                                  'numberOfSCTHits', 'charge']]

        num_track_features = len(track_features)
        padded_tracks = np.zeros((len(events), MAX_TRACKS, num_track_features))

        for i in range(len(events)):
            num_tracks = min(len(track_features[0][i]), MAX_TRACKS)
            for j in range(num_tracks):
                for k in range(num_track_features):
                    if len(track_features[k][i]) > 0:
                        padded_tracks[i, j, k] = track_features[k][i][j]

        # Calculate delta_eta, delta_phi, pt_ratio_log, and dR
        jet_eta, jet_phi, jet_pt = padded_jets[:, 1:4].T
        mask = padded_tracks[:, :, 0] != 0

        padded_tracks[:, :, 0] = np.where(mask, padded_tracks[:, :, 0] - jet_eta[:, np.newaxis], 0)
        
        phi_difference = np.where(mask, padded_tracks[:, :, 1] - jet_phi[:, np.newaxis], 0)
        phi_difference = np.where(phi_difference > np.pi, phi_difference - 2*np.pi, phi_difference)
        phi_difference = np.where(phi_difference <= -np.pi, phi_difference + 2*np.pi, phi_difference)
        padded_tracks[:, :, 1] = phi_difference

        epsilon = 1e-8
        pt_ratio = np.where(mask, padded_tracks[:, :, 2] / jet_pt[:, np.newaxis], 0)
        pt_ratio = np.nan_to_num(pt_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        pt_ratio_subtracted = np.where(mask, 1 - pt_ratio + epsilon, epsilon)
        pt_ratio_log = np.log(pt_ratio_subtracted)
        pt_ratio_log = np.nan_to_num(pt_ratio_log, nan=0.0, posinf=0.0, neginf=0.0)
        pt_ratio_log = np.where(pt_ratio_subtracted == epsilon, 0, pt_ratio_log)
        
        dR = np.hypot(padded_tracks[:, :, 0], padded_tracks[:, :, 1])

        padded_tracks = np.insert(padded_tracks, 4, pt_ratio_log, axis=2)
        padded_tracks = np.insert(padded_tracks, 6, dR, axis=2)
        padded_tracks[:, :, 2] = np.ma.log(padded_tracks[:, :, 2]).filled(0)

        pids = np.full(len(events), label)
        
        return padded_jets, padded_tracks, pids, event_weights_raw, event_weight_sum

    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        return None

def process_file_pattern(file_pattern, label, xs_id):
    file_path = path + file_pattern
    files = glob.glob(os.path.join(file_path, '*.root'))

    print(f"Processing {len(files)} files from {file_pattern} with label {label}")
    print("getting xs for", xs_id, "with value", getXS(xs_id))

    jet_data, track_data, pids, event_weights = [], [], [], []
    event_weight_sum = 0
    ditau_pt = []

    for file in tqdm(files, desc="Files"):
        result = process_file(file, label, xs_id)
        if result is not None:
            padded_jets, padded_tracks, file_pids, file_weights, file_weight_sum = result
            jet_data.append(padded_jets)
            track_data.append(padded_tracks)
            pids.append(file_pids)
            event_weights.append(file_weights)
            event_weight_sum += file_weight_sum
            ditau_pt.append(padded_jets[:, 0])

    if not jet_data:
        return None

    jet_data = np.concatenate(jet_data)
    track_data = np.concatenate(track_data)
    pids = np.concatenate(pids)
    event_weights = ak.concatenate(event_weights)
    ditau_pt = np.concatenate(ditau_pt)

    event_weights_norm = ak.to_numpy(event_weights / event_weight_sum)
    pt_bins = np.linspace(ditau_pt.min(), ditau_pt.max(), 41)
    pt_flattening_weights = flattened_pt_weighted(ditau_pt, pt_bins, event_weights_norm)
    weights = event_weights_norm * getXS(xs_id) * pt_flattening_weights

    return jet_data, track_data, pids, weights

def process_files(file_patterns, labels, xs_ids):
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.starmap(process_file_pattern, zip(file_patterns, labels, xs_ids)), total=len(file_patterns), desc="File patterns"))

    jet_data, track_data, pids, weights = [], [], [], []
    for result in results:
        if result is not None:
            jet_data.append(result[0])
            track_data.append(result[1])
            pids.append(result[2])
            weights.append(result[3])

    return np.concatenate(jet_data), np.concatenate(track_data), np.concatenate(pids), np.concatenate(weights)

if __name__ == "__main__":
    jet_data, track_data, pids, weights = process_files(file_patterns, labels, xs_ids)

    print("Jet data shape:", jet_data.shape)
    print("Track data shape:", track_data.shape)
    print("PIDs shape:", pids.shape)
    print("Event weights shape:", weights.shape)

    np.random.seed(42)
    permutation = np.random.permutation(jet_data.shape[0])
    jet_data = jet_data[permutation]
    track_data = track_data[permutation]
    pids = pids[permutation]
    weights = weights[permutation]

    event_ids = jet_data[:, 3].astype(int)
    eventID_mod = event_ids % 100

    train_indices = np.where(eventID_mod < 60)[0]
    val_indices = np.where((eventID_mod >= 60) & (eventID_mod < 80))[0]
    test_indices = np.where(eventID_mod >= 80)[0]

    print("Train data count:", len(train_indices))
    print("Validation data count:", len(val_indices))
    print("Test data count:", len(test_indices))

    train_jet, train_track, train_pid, train_weights = jet_data[train_indices], track_data[train_indices], pids[train_indices], weights[train_indices]
    val_jet, val_track, val_pid, val_weights = jet_data[val_indices], track_data[val_indices], pids[val_indices], weights[val_indices]
    test_jet, test_track, test_pid, test_weights = jet_data[test_indices], track_data[test_indices], pids[test_indices], weights[test_indices]

    train_jet = train_jet[:, 3:]
    val_jet = val_jet[:, 3:]
    test_jet = test_jet[:, 3:]

    print("Jet Train data shape:", train_jet.shape)
    print("Track Train data shape:", train_track.shape)
    print("PIDs Train shape:", train_pid.shape)
    print("Weights Train shape:", train_weights.shape)

    output_dir = '/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn_onnx/test/'

    def write_h5(filename, jet_data, track_data, pid_data, weight_data):
        with h5py.File(os.path.join(output_dir, filename), 'w') as f:
            f.create_dataset('data', data=track_data)
            f.create_dataset('jet', data=jet_data)
            f.create_dataset('pid', data=pid_data)
            f.create_dataset('weights', data=weight_data)

    write_h5('train_tau.h5', train_jet, train_track, train_pid, train_weights)
    write_h5('test_tau.h5', test_jet, test_track, test_pid, test_weights)
    write_h5('val_tau.h5', val_jet, val_track, val_pid, val_weights)