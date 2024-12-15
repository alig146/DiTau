import uproot
import glob
import os, sys
import numpy as np
import awkward as ak
from tqdm import tqdm
from sklearn.utils import shuffle
import h5py
sys.path.append("..")
from utils.utils import flattened_pt_weighted, getXS, flattened_pt

# Define the branches to be extracted
branches = ['d0TJVA', 'z0TJVA', 'trackPt', 'trackEta', 'trackPhi', 'mcEventWeights', 'omni_score', 'delta_eta', 'delta_phi', 'pt_log', 'pt_ratio_log', 'delta_R',
            'numberOfInnermostPixelLayerHits', 'numberOfPixelHits', 'numberOfSCTHits', 'charge', 'ditau_idx',
            #'qOverP', 'lead_subjet_pt', 'sublead_subjet_pt', 'ditau_n_subjets', 'ditau_n_tracks_lead', 'ditau_n_tracks_subl',
            #'numberOfPixelSharedHits', 'numberOfPixelDeadSensors',
            # 'numberOfSCTSharedHits', 'numberOfSCTDeadSensors', 'numberOfTRTHighThresholdHits', 
            # 'numberOfTRTHits', 'expectInnermostPixelLayerHit', 'expectNextToInnermostPixelLayerHit',
            # 'numberOfContribPixelLayers', 'numberOfPixelHoles', 'numberOfSCTHoles'
            'mcEventNumber', 'ditau_ditau_pt',   
            'ditau_R_max_lead', 'ditau_R_max_subl', 'ditau_R_tracks_subl', 'ditau_R_isotrack', 
            'ditau_d0_leadtrack_lead', 'ditau_d0_leadtrack_subl', 'ditau_f_core_lead', 
            'ditau_f_core_subl', 'ditau_f_subjet_subl', 'ditau_f_subjets', 'ditau_f_isotracks', 
            'ditau_m_core_lead', 'ditau_m_core_subl', 'ditau_m_tracks_lead', 'ditau_m_tracks_subl', 
            'ditau_n_track', 'ditau_eta', 'ditau_phi']

# path = '/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn'
path = '/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn_onnx'

bkg_file_paths = [path + pattern for pattern in [
    '/user.agarabag*JZ0_v2*.root', '/user.agarabag*JZ1_v2*.root', '/user.agarabag*JZ2_v2*.root', 
    '/user.agarabag*JZ3_v2*.root', '/user.agarabag*JZ4_v2*.root', '/user.agarabag*JZ5_v2*.root', 
    '/user.agarabag*JZ6_v2*.root', '/user.agarabag*JZ7_v2*.root', '/user.agarabag*JZ8_v2*.root', 
    '/user.agarabag*JZ9incl_v2*.root']]
signal_file_paths = [path + '/user.agarabag.VHTauTauOmni.802168.Py8EG_A14NNPDF23LO_VHtautau_flatmasspTFilt_hadhad_v18_ntuple.root']
xs_ids = [801165, 801166, 801167, 801168, 801169, 801170, 801171, 801172, 801173, 801174, 802168] #last one is signal

# Combine background and signal file paths
file_paths = bkg_file_paths + signal_file_paths
labels = [0] * len(bkg_file_paths) + [1] * len(signal_file_paths)

# Initialize lists to store the data
jet_data = []
track_data = []
pids = []
total_weight = []

# Define maximum number of tracks per event
MAX_TRACKS = 10  # You may need to adjust this based on your data

for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
    l1 = glob.glob(os.path.join(file_path, '*.root'))

    print(f"Processing {len(l1)} files from {file_path} with label {label}")
    print("getting xs for ", xs_ids[idx], "with value ", getXS(xs_ids[idx]))

    event_weights = []
    event_weight_sum = []
    ditau_pt = []

    for file in tqdm(l1, desc="Files"):
        print("processing file: ", file)
        f_1 = uproot.open(f"{file}:CollectionTree")
        # Check if the tree exists and has entries
        if f_1.num_entries == 0:
            print(f"File {file} is empty, skipping...")
            continue
        f_1 = uproot.open(f"{file}:CollectionTree")
        events = f_1.arrays(branches, library='ak')
        print("# events: ", len(events))

        event_weights_raw = ak.firsts(events['mcEventWeights'][:, 0, :])
        ditau_pt.append(events['ditau_ditau_pt'])

        ## save jet features
        jet_vars = [
            events['ditau_ditau_pt'],
            events['ditau_eta'],
            events['ditau_phi'],
            events['mcEventNumber'],
            events['ditau_R_max_lead'],
            events['ditau_R_max_subl'],
            events['ditau_R_tracks_subl'],
            events['ditau_R_isotrack'],
            events['ditau_d0_leadtrack_lead'],
            events['ditau_d0_leadtrack_subl'],
            events['ditau_f_core_lead'],
            events['ditau_f_core_subl'],
            events['ditau_f_subjet_subl'],
            events['ditau_f_subjets'],
            events['ditau_f_isotracks'],
            events['ditau_m_core_lead'],
            events['ditau_m_core_subl'],
            events['ditau_m_tracks_lead'],
            events['ditau_n_track'],
        ]
     
        jet_vars = [ak.to_numpy(var) for var in jet_vars]
        padded_jets = np.column_stack(jet_vars)

        ## save track features
        track_features = [
            events['trackEta'], 
            events['trackPhi'], 
            events['trackPt'], 
            events['d0TJVA'],
            events['z0TJVA'],
            events['numberOfInnermostPixelLayerHits'], 
            events['numberOfPixelHits'], 
            events['numberOfSCTHits'], 
            events['charge']
        ]

        num_track_features = len(track_features)
        padded_tracks = np.zeros((len(events), MAX_TRACKS, num_track_features))

        for i in range(len(events)):
            num_tracks = min(len(track_features[0][i]), MAX_TRACKS)
            for j in range(num_tracks):
                for k in range(num_track_features):
                    if len(track_features[k][i]) > 0:  # Check if the array is not empty
                        padded_tracks[i, j, k] = track_features[k][i][j]
                    else:
                        padded_tracks[i, j, k] = 0  # Assign a default value (e.g., 0) if the array is empty

        #cal delta_eta
        jet_eta = padded_jets[:, 1]
        jet_eta = jet_eta[:, np.newaxis, np.newaxis]
        jet_eta = np.repeat(jet_eta, MAX_TRACKS, axis=1)
        jet_eta = np.repeat(jet_eta, num_track_features, axis=2)
        mask = padded_tracks[:, :, 0] != 0
        eta_difference = np.where(mask, padded_tracks[:, :, 0] - jet_eta[:, :, 0], 0)
        # for i in range(len(eta_difference)):
        #     if np.any(eta_difference[i] > 1.5):
        #         print("eta_difference: ", eta_difference[i])
        #         print("jet_eta: ", jet_eta[i, :, 0])
        #         print("padded_tracks: ", padded_tracks[i, :, 0])
        padded_tracks[:, :, 0] = eta_difference

        #calculate delta_phi
        jet_phi = padded_jets[:, 2]
        jet_phi = jet_phi[:, np.newaxis, np.newaxis]
        jet_phi = np.repeat(jet_phi, MAX_TRACKS, axis=1)
        jet_phi = np.repeat(jet_phi, num_track_features, axis=2)
        mask = padded_tracks[:, :, 1] != 0
        phi_difference = np.where(mask, padded_tracks[:, :, 1] - jet_phi[:, :, 0], 0)
        phi_difference = np.where(phi_difference > np.pi, phi_difference - 2*np.pi, phi_difference)
        phi_difference = np.where(phi_difference <= -np.pi, phi_difference + 2*np.pi, phi_difference)
        padded_tracks[:, :, 1] = phi_difference
        # print("PPPPiiii: ", padded_tracks[:, :, 1][0])

        epsilon = 1e-8
        jet_pt = padded_jets[:, 0]
        jet_pt = jet_pt[:, np.newaxis, np.newaxis]
        jet_pt = np.repeat(jet_pt, MAX_TRACKS, axis=1)
        jet_pt = np.repeat(jet_pt, num_track_features, axis=2)
        # jet_pt = np.maximum(jet_pt, epsilon)
        mask = padded_tracks[:, :, 2] != 0
        pt_ratio = np.where(mask, padded_tracks[:, :, 2] / jet_pt[:, :, 0], 0)
        pt_ratio = np.nan_to_num(pt_ratio, nan=0.0, posinf=1.0, neginf=0.0)
        pt_ratio_subtracted = np.where(mask, 1 - pt_ratio + epsilon, epsilon)
        # pt_ratio_subtracted = np.maximum(pt_ratio_subtracted, epsilon)
        pt_ratio_log = np.log(pt_ratio_subtracted)
        pt_ratio_log = np.nan_to_num(pt_ratio_log, nan=0.0, posinf=0.0, neginf=0.0)
        pt_ratio_log = np.where(pt_ratio_subtracted == epsilon, 0, pt_ratio_log)
        padded_tracks = np.insert(padded_tracks, 4, pt_ratio_log, axis=2)
        # print("PTPT: ", pt_ratio_log[0])

        #calculate deta_R
        deta = padded_tracks[:, :, 0]
        dphi = padded_tracks[:, :, 1]
        dR = np.hypot(deta, dphi)
        padded_tracks = np.insert(padded_tracks, 6, dR, axis=2)
        # print("DRDR: ", dR)

        #take the log of the track pt 
        # padded_tracks[:, :, 2] = np.ma.log(padded_tracks[:, :, 2] + epsilon)
        mask = padded_tracks[:, :, 2] != 0
        padded_tracks[:, :, 2][mask] = np.ma.log(padded_tracks[:, :, 2][mask] + epsilon)
        # print("PPPPPPP: ", padded_tracks[:, :, 2][0])

        jet_data.append(padded_jets)
        track_data.append(padded_tracks)
        pids.append(np.full(len(events), label))
        event_weights.append(event_weights_raw)
        event_weight_sum.append(ak.sum(ak.firsts(events['mcEventWeights'])))

    event_weights_norm = ak.to_numpy(ak.concatenate(event_weights)/ak.sum(event_weight_sum))
    # pt_bins = np.linspace(200000, 1000000, 41)
    pt_numpy = ak.to_numpy(ak.concatenate(ditau_pt)).flatten()
    pt_bins = np.linspace(pt_numpy.min(), pt_numpy.max(), 41)
    # pt_flattening_weights = flattened_pt(pt_numpy, pt_bins)
    pt_flattening_weights = flattened_pt_weighted(pt_numpy, pt_bins, event_weights_norm)
    total_weight.append(event_weights_norm * getXS(xs_ids[idx]) * pt_flattening_weights)


# Convert lists to numpy arrays
jet_data = np.concatenate(jet_data)
track_data = np.concatenate(track_data)
pids = np.concatenate(pids)
weigts = np.concatenate(total_weight)
print("Jet data shape: ", jet_data.shape)
print("Track data shape: ", track_data.shape)
print("PIDs shape: ", pids.shape)
print("Event weights shape: ", weigts.shape)

np.random.seed(42)
permutation = np.random.permutation(jet_data.shape[0])
jet_data = jet_data[permutation]
track_data = track_data[permutation]
pids = pids[permutation]
weigts = weigts[permutation]

event_ids = jet_data[:, 3].astype(int)
eventID_mod = event_ids % 100

train_indices = np.where((eventID_mod < 60))[0]  # 60%
val_indices = np.where((eventID_mod >= 60) & (eventID_mod < 80))[0]  # 20%
test_indices = np.where(eventID_mod >= 80)[0]  # 20%

print("Train data count: ", len(train_indices))
print("Validation data count: ", len(val_indices))
print("Test data count: ", len(test_indices))

train_jet, train_track, train_pid, train_weigts = jet_data[train_indices], track_data[train_indices], pids[train_indices], weigts[train_indices]
val_jet, val_track, val_pid, val_weigts = jet_data[val_indices], track_data[val_indices], pids[val_indices], weigts[val_indices]
test_jet, test_track, test_pid, test_weigts = jet_data[test_indices], track_data[test_indices], pids[test_indices], weigts[test_indices]

#take out jet_data elemts 0, 1, 2, 3
train_jet = train_jet[:, 3:] #use 4 if you don't want to save event id's
val_jet = val_jet[:, 3:]
test_jet = test_jet[:, 3:]
    
print("Jet Train data shape: ", train_jet.shape)
print("Track Train data shape: ", train_track.shape)
print("PIDs Train shape: ", train_pid.shape)
print("Weights Train shape: ", train_weigts.shape)

# Write to H5 files
output_dir = '/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn_onnx/for_training/'

with h5py.File(os.path.join(output_dir, 'train_tau.h5'), 'w') as train_file:
    train_file.create_dataset('data', data=train_track)
    train_file.create_dataset('jet', data=train_jet)
    train_file.create_dataset('pid', data=train_pid)
    train_file.create_dataset('weights', data=train_weigts)

with h5py.File(os.path.join(output_dir, 'test_tau.h5'), 'w') as test_file:
    test_file.create_dataset('data', data=test_track)
    test_file.create_dataset('jet', data=test_jet)
    test_file.create_dataset('pid', data=test_pid)
    test_file.create_dataset('weights', data=test_weigts)

with h5py.File(os.path.join(output_dir, 'val_tau.h5'), 'w') as val_file:
    val_file.create_dataset('data', data=val_track)
    val_file.create_dataset('jet', data=val_jet)
    val_file.create_dataset('pid', data=val_pid)
    val_file.create_dataset('weights', data=val_weigts)
