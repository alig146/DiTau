import uproot
import glob
import os
import numpy as np
import awkward as ak
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split

# def get_first_item(array):
#     return array[:, 0] if len(array) > 0 else array

def get_first(array):
    if len(array) > 0:
        return array[:, 0]
    else:
        return ak.Array([])

# Define the branches to be extracted
branches = ['d0TJVA', 'z0TJVA', 'trackPt', 'trackEta', 'trackPhi', 
            'numberOfInnermostPixelLayerHits', 'numberOfPixelHits', 'numberOfSCTHits', 'qOverP', 
            'mcEventNumber', 'ditau_ditau_pt',
            'lead_subjet_pt', 'sublead_subjet_pt', 'ditau_n_subjets', 
            'ditau_IsTruthHadronic', 'ditau_n_tracks_lead', 'ditau_n_tracks_subl', 
            'ditau_R_max_lead', 'ditau_R_max_subl', 'ditau_R_tracks_subl', 'ditau_R_isotrack', 
            'ditau_d0_leadtrack_lead', 'ditau_d0_leadtrack_subl', 'ditau_f_core_lead', 
            'ditau_f_core_subl', 'ditau_f_subjet_subl', 'ditau_f_subjets', 'ditau_f_isotracks', 
            'ditau_m_core_lead', 'ditau_m_core_subl', 'ditau_m_tracks_lead', 'ditau_m_tracks_subl', 
            'ditau_n_track', 'ditau_eta', 'ditau_phi', 'charge', 'numberOfPixelSharedHits', 'numberOfPixelDeadSensors',
            'numberOfSCTSharedHits', 'numberOfSCTDeadSensors', 'numberOfTRTHighThresholdHits', 
            'numberOfTRTHits', 'expectInnermostPixelLayerHit', 'expectNextToInnermostPixelLayerHit',
            'numberOfContribPixelLayers', 'numberOfPixelHoles', 'numberOfSCTHoles']

path = '/global/homes/a/agarabag/pscratch/ditdau_samples/samples_for_gnn'

bkg_file_paths = [path + pattern for pattern in [
    '/user.agarabag*JZ0*.root', '/user.agarabag*JZ1*.root', '/user.agarabag*JZ2*.root', 
    '/user.agarabag*JZ3*.root', '/user.agarabag*JZ4*.root', '/user.agarabag*JZ5*.root', 
    '/user.agarabag*JZ6*.root', '/user.agarabag*JZ7*.root', '/user.agarabag*JZ8*.root', 
    '/user.agarabag*JZ9incl*.root']]
signal_file_paths = [path + '/user.agarabag.VHTauTauNTuple.802168.Py8EG_A14NNPDF23LO_VHtautau_flatmasspTFilt_hadhad_v2_ntuple.root']

# Combine background and signal file paths
file_paths = bkg_file_paths + signal_file_paths
labels = [0] * len(bkg_file_paths) + [1] * len(signal_file_paths)

# Initialize lists to store the data
jet_data = []
track_data = []
pids = []

# Define maximum number of tracks per event
MAX_TRACKS = 10  # You may need to adjust this based on your data

for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
    l1 = glob.glob(os.path.join(file_path, '*.root'))
    print(f"Processing {len(l1)} files from {file_path} with label {label}")

    for file in tqdm(l1, desc="Files"):
        f_1 = uproot.open(f"{file}:CollectionTree")
        events = f_1.arrays(branches, library='ak')
        print("processing file: ", file)
        print("# events: ", len(events))

        ## save jet features
        jet_vars = [
                get_first(events['ditau_ditau_pt']),
                get_first(events['ditau_eta']),
                get_first(events['ditau_phi']),
                get_first(events['ditau_n_tracks_lead']),
                get_first(events['ditau_n_tracks_subl']),
                get_first(events['ditau_R_max_lead']),
                get_first(events['ditau_R_max_subl']),
                get_first(events['ditau_R_tracks_subl']),
                get_first(events['ditau_R_isotrack']),
                get_first(events['ditau_d0_leadtrack_lead']),
                get_first(events['ditau_d0_leadtrack_subl']),
                get_first(events['ditau_f_core_lead']),
                get_first(events['ditau_f_core_subl']),
                get_first(events['ditau_f_subjet_subl']),
                get_first(events['ditau_f_subjets']),
                get_first(events['ditau_f_isotracks']),
                get_first(events['ditau_m_core_lead']),
                get_first(events['ditau_m_core_subl']),
                get_first(events['ditau_m_tracks_lead']),
                get_first(events['ditau_m_tracks_subl']),
                get_first(events['ditau_n_track']),
                # get_first(events['lead_subjet_pt']),
                # get_first(events['sublead_subjet_pt'])
            ]

        jet_vars = [ak.to_numpy(var) for var in jet_vars]
        padded_jets = np.column_stack(jet_vars)

        ## save track features
        track_features = [
            events['trackEta'], events['trackPhi'], events['trackPt'], events['d0TJVA'], events['z0TJVA'],
            events['numberOfInnermostPixelLayerHits'], events['numberOfPixelHits'], 
            events['numberOfSCTHits'], events['charge']
        ]

        num_track_features = len(track_features)
        padded_tracks = np.zeros((len(events), MAX_TRACKS, num_track_features))

        for i in range(len(events)):
            num_tracks = min(len(track_features[0][i]), MAX_TRACKS)
            for j in range(num_tracks):
                for k in range(num_track_features):
                    padded_tracks[i, j, k] = track_features[k][i][j]

        #cal delta_eta
        jet_eta = padded_jets[:, 1]
        jet_eta = jet_eta[:, np.newaxis, np.newaxis]
        jet_eta = np.repeat(jet_eta, MAX_TRACKS, axis=1)
        jet_eta = np.repeat(jet_eta, num_track_features, axis=2)
        mask = padded_tracks[:, :, 0] != 0
        eta_difference = np.where(mask, padded_tracks[:, :, 0] - jet_eta[:, :, 0], 0)
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

        #calculate the ratio between track pt and jet pt
        jet_pt = padded_jets[:, 0]
        jet_pt = jet_pt[:, np.newaxis, np.newaxis]
        jet_pt = np.repeat(jet_pt, MAX_TRACKS, axis=1)
        jet_pt = np.repeat(jet_pt, num_track_features, axis=2)
        mask = padded_tracks[:, :, 2] != 0
        pt_ratio = np.where(mask, padded_tracks[:, :, 2] / jet_pt[:, :, 0], 0)
        pt_ratio_subtracted = np.where(mask, 1 - pt_ratio + 1e-8, 0)
        pt_ratio_log = np.log(np.where(pt_ratio_subtracted != 0, pt_ratio_subtracted, 1))
        # print("JJJJ: ", padded_jets[3, 0])
        # print("TTT: ", padded_tracks[3, :, 2])
        # print("RRR: ", pt_ratio_log[3, :])
        # padded_tracks = np.concatenate((padded_tracks, pt_ratio[:, :, np.newaxis]), axis=2)
        padded_tracks = np.insert(padded_tracks, 3, pt_ratio_log, axis=2)

        #calculate deta_R
        deta = padded_tracks[:, :, 0]
        dphi = padded_tracks[:, :, 1]
        dR = np.hypot(deta, dphi)
        padded_tracks = np.insert(padded_tracks, 6, dR, axis=2)

        #take the log of the track pt 
        padded_tracks[:, :, 2] = np.log(padded_tracks[:, :, 2] + 1e-8)

        print("track shape: ", padded_tracks.shape)
        print("jet shape: ", padded_jets.shape)
        jet_data.append(padded_jets)
        track_data.append(padded_tracks)
        pids.append(np.full(len(events), label))

# Convert lists to numpy arrays
jet_data = np.concatenate(jet_data)
track_data = np.concatenate(track_data)
pids = np.concatenate(pids)

# Split data into train, test, and validation sets
train_ratio = 0.6
test_ratio = 0.2
val_ratio = 0.2

train_jet, testval_jet, train_track, testval_track, train_pid, testval_pid = train_test_split(
    jet_data, track_data, pids, test_size=test_ratio + val_ratio, random_state=42)
test_jet, val_jet, test_track, val_track, test_pid, val_pid = train_test_split(
    testval_jet, testval_track, testval_pid, test_size=val_ratio/(test_ratio + val_ratio), random_state=42)

# Write to H5 files
output_dir = '/global/homes/a/agarabag/pscratch/ditdau_samples/'

with h5py.File(os.path.join(output_dir, 'train_tau.h5'), 'w') as train_file:
    train_file.create_dataset('data', data=train_track)
    train_file.create_dataset('jet', data=train_jet)
    train_file.create_dataset('pid', data=train_pid)

with h5py.File(os.path.join(output_dir, 'test_tau.h5'), 'w') as test_file:
    test_file.create_dataset('data', data=test_track)
    test_file.create_dataset('jet', data=test_jet)
    test_file.create_dataset('pid', data=test_pid)

with h5py.File(os.path.join(output_dir, 'val_tau.h5'), 'w') as val_file:
    val_file.create_dataset('data', data=val_track)
    val_file.create_dataset('jet', data=val_jet)
    val_file.create_dataset('pid', data=val_pid)
