import uproot, ROOT, glob, os, random
import numpy as np
import awkward as ak
from tqdm import tqdm
import h5py

branches = ['EventInfoAux.eventNumber',
            'DiTauJetsAuxDyn.BDTScore',
            'DiTauJetsAuxDyn.BDTScoreNew',
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

# file_path = "/eos/user/j/jlai/ditau/jz1.txt"

# # Pattern to search for
# search_pattern = 'NERSC_LOCALGROUPDISK: ...... global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ ........ / output.root'

# file_paths = []
# # Open the file and search for lines containing 'NERSC_LOCALGROUPDISK' and 'output.root'
# with open(file_path, 'r') as file:
#     for line in file:
#         if 'NERSC_LOCALGROUPDISK' in line and 'output.root' in line:
#             # Find the index of '/global' in the line
#             start_index = line.find('/global')
#             end_index = line.find('|')
#             if start_index != -1:  # Make sure '/global' was found in the line
#                 print(line[start_index:-3].strip())
#                 file_paths.append(line[start_index:-3].strip())


file_paths = ['/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/95/33/user.agarabag.37271399._000001.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/62/d7/user.agarabag.37271399._000002.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/97/ea/user.agarabag.37271399._000003.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/28/34/user.agarabag.37271399._000004.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8e/d2/user.agarabag.37271399._000005.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d3/55/user.agarabag.37271399._000006.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/18/50/user.agarabag.37271399._000007.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/98/63/user.agarabag.37271399._000008.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/33/ff/user.agarabag.37271399._000009.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/35/29/user.agarabag.37271399._000012.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/60/c6/user.agarabag.37271399._000013.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/96/22/user.agarabag.37271399._000014.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c3/9c/user.agarabag.37271399._000015.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f6/02/user.agarabag.37271399._000016.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/fe/ce/user.agarabag.37271399._000017.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/80/f8/user.agarabag.37271399._000018.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/86/67/user.agarabag.37271399._000019.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/4e/ee/user.agarabag.37271399._000020.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/bc/b7/user.agarabag.37271399._000021.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f4/e7/user.agarabag.37271399._000022.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/66/04/user.agarabag.37271399._000023.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8f/5c/user.agarabag.37271399._000024.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/2c/6e/user.agarabag.37271399._000025.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/74/c8/user.agarabag.37271399._000026.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/81/21/user.agarabag.37271399._000027.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/dd/0c/user.agarabag.37271399._000028.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e1/89/user.agarabag.37271399._000029.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/34/e9/user.agarabag.37271399._000030.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/73/b3/user.agarabag.37271399._000031.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b2/5a/user.agarabag.37271399._000032.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/83/fa/user.agarabag.37271399._000033.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/5b/ca/user.agarabag.37271399._000034.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f9/17/user.agarabag.37271399._000035.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d9/60/user.agarabag.37271399._000036.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ba/0a/user.agarabag.37271399._000037.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/2f/45/user.agarabag.37271399._000038.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ad/cb/user.agarabag.37271399._000039.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/22/8b/user.agarabag.37271399._000040.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/04/06/user.agarabag.37271399._000041.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8d/96/user.agarabag.37271399._000042.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/07/b2/user.agarabag.37271399._000043.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d6/c5/user.agarabag.37271399._000044.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/0a/ba/user.agarabag.37271399._000045.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/33/35/user.agarabag.37271399._000046.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/80/0f/user.agarabag.37271399._000047.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/98/5a/user.agarabag.37271399._000048.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/dc/cf/user.agarabag.37271399._000049.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/41/05/user.agarabag.37271399._000050.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8e/c7/user.agarabag.37271399._000051.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/27/b2/user.agarabag.37271399._000052.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ff/6a/user.agarabag.37271399._000053.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/56/ff/user.agarabag.37271399._000054.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/df/d7/user.agarabag.37271399._000055.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c2/24/user.agarabag.37271399._000056.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/28/c2/user.agarabag.37271399._000057.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/81/cb/user.agarabag.37271399._000058.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/96/38/user.agarabag.37271399._000059.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6a/f7/user.agarabag.37271399._000060.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/bf/d4/user.agarabag.37271399._000061.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/9b/95/user.agarabag.37271399._000062.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/7e/a8/user.agarabag.37271399._000063.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/28/69/user.agarabag.37271399._000064.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/5d/b9/user.agarabag.37271399._000065.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/08/c6/user.agarabag.37271399._000066.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/18/b7/user.agarabag.37271399._000067.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/5d/16/user.agarabag.37271399._000068.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/88/20/user.agarabag.37271399._000069.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a1/d4/user.agarabag.37271399._000070.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6c/c2/user.agarabag.37271399._000071.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ce/54/user.agarabag.37271399._000072.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/82/b1/user.agarabag.37271399._000073.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/18/df/user.agarabag.37271399._000074.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/3d/07/user.agarabag.37271399._000075.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/aa/80/user.agarabag.37271399._000076.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/65/e0/user.agarabag.37271399._000077.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/77/fa/user.agarabag.37271399._000078.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c5/c1/user.agarabag.37271399._000079.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/56/87/user.agarabag.37271399._000080.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a7/79/user.agarabag.37271399._000081.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/7e/26/user.agarabag.37271399._000082.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f3/f6/user.agarabag.37271399._000083.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/1b/56/user.agarabag.37271399._000084.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8f/6f/user.agarabag.37271399._000085.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e3/5a/user.agarabag.37271399._000086.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ae/fb/user.agarabag.37271399._000087.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/84/ea/user.agarabag.37271399._000088.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c7/1f/user.agarabag.37271399._000089.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/cf/70/user.agarabag.37271399._000090.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/87/1f/user.agarabag.37271399._000091.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b0/b0/user.agarabag.37271399._000092.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/39/6e/user.agarabag.37271399._000093.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/05/db/user.agarabag.37271399._000094.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6a/5b/user.agarabag.37271399._000095.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/10/3d/user.agarabag.37271399._000096.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/4c/9d/user.agarabag.37271399._000097.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/57/0b/user.agarabag.37271399._000098.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/75/dd/user.agarabag.37271399._000099.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/9d/a4/user.agarabag.37271399._000100.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/24/b8/user.agarabag.37271399._000101.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/9a/43/user.agarabag.37271399._000102.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f6/d4/user.agarabag.37271399._000103.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/64/09/user.agarabag.37271399._000104.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a2/ab/user.agarabag.37271399._000105.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e7/57/user.agarabag.37271399._000106.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/df/68/user.agarabag.37271399._000107.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a8/d0/user.agarabag.37271399._000108.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ef/4b/user.agarabag.37271399._000109.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e9/8d/user.agarabag.37271399._000110.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/2a/f6/user.agarabag.37271399._000111.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/48/f1/user.agarabag.37271399._000112.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/2a/3b/user.agarabag.37271399._000113.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/86/a2/user.agarabag.37271399._000114.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e6/5e/user.agarabag.37271399._000115.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/5f/6b/user.agarabag.37271399._000116.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a3/d7/user.agarabag.37271399._000117.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ab/70/user.agarabag.37271399._000118.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/99/30/user.agarabag.37271399._000119.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ae/be/user.agarabag.37271399._000120.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/83/a3/user.agarabag.37271399._000121.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ae/8a/user.agarabag.37271399._000122.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/55/26/user.agarabag.37271399._000123.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6b/54/user.agarabag.37271399._000124.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8b/04/user.agarabag.37271399._000125.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/29/ce/user.agarabag.37271399._000126.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/1b/b1/user.agarabag.37271399._000127.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/63/ac/user.agarabag.37271399._000128.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/71/fc/user.agarabag.37271399._000129.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/88/7c/user.agarabag.37271399._000130.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/50/cc/user.agarabag.37271399._000131.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/59/34/user.agarabag.37271399._000132.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/18/06/user.agarabag.37271399._000133.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/35/28/user.agarabag.37271399._000134.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/80/75/user.agarabag.37271399._000135.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/0d/70/user.agarabag.37271399._000136.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/69/53/user.agarabag.37271399._000137.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/4b/0d/user.agarabag.37271399._000138.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/00/f5/user.agarabag.37271399._000139.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/38/89/user.agarabag.37271399._000140.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/7c/d8/user.agarabag.37271399._000141.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ff/e0/user.agarabag.37271399._000142.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/0e/48/user.agarabag.37271399._000143.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/16/6b/user.agarabag.37271399._000144.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a4/11/user.agarabag.37271399._000145.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/78/1b/user.agarabag.37271399._000146.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/da/07/user.agarabag.37271399._000147.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/80/55/user.agarabag.37271399._000148.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a9/91/user.agarabag.37271399._000149.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a8/69/user.agarabag.37271399._000150.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6f/f9/user.agarabag.37271399._000151.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/88/d2/user.agarabag.37271399._000152.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/69/68/user.agarabag.37271399._000153.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a0/b2/user.agarabag.37271399._000154.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/10/17/user.agarabag.37271399._000155.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ae/c9/user.agarabag.37271399._000156.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/15/ad/user.agarabag.37271399._000157.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d8/06/user.agarabag.37271399._000158.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/11/a1/user.agarabag.37271399._000159.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/38/3d/user.agarabag.37271399._000160.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/41/40/user.agarabag.37271399._000161.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c4/b4/user.agarabag.37271399._000162.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/2c/91/user.agarabag.37271399._000163.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8a/e2/user.agarabag.37271399._000164.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/27/cd/user.agarabag.37271399._000165.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ab/a4/user.agarabag.37271399._000166.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f8/23/user.agarabag.37271399._000167.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ad/27/user.agarabag.37271399._000168.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/1d/6f/user.agarabag.37271399._000169.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e2/eb/user.agarabag.37271399._000170.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ac/f1/user.agarabag.37271399._000171.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ec/ff/user.agarabag.37271399._000172.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/58/92/user.agarabag.37271399._000173.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e5/f5/user.agarabag.37271399._000174.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/2e/b9/user.agarabag.37271399._000175.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/7d/e5/user.agarabag.37271399._000176.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/3b/64/user.agarabag.37271399._000177.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/44/0d/user.agarabag.37271399._000178.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/bb/8b/user.agarabag.37271399._000179.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/9b/64/user.agarabag.37271399._000181.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/05/26/user.agarabag.37271399._000182.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e6/8a/user.agarabag.37271399._000183.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c9/df/user.agarabag.37271399._000184.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8c/40/user.agarabag.37271399._000185.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d0/31/user.agarabag.37271399._000186.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/4f/ed/user.agarabag.37271399._000187.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/11/3f/user.agarabag.37271399._000188.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/3f/75/user.agarabag.37271399._000189.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e6/12/user.agarabag.37271399._000190.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/87/75/user.agarabag.37271399._000191.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/7e/48/user.agarabag.37271399._000192.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/62/f5/user.agarabag.37271399._000193.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/87/00/user.agarabag.37271399._000194.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b9/50/user.agarabag.37271399._000195.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/17/f8/user.agarabag.37271399._000196.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c1/0c/user.agarabag.37271399._000197.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6b/ec/user.agarabag.37271399._000199.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b1/8d/user.agarabag.37271399._000200.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/98/47/user.agarabag.37271399._000202.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b8/bf/user.agarabag.37271399._000203.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a3/a5/user.agarabag.37271399._000204.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/01/7a/user.agarabag.37271399._000205.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/46/d9/user.agarabag.37271399._000206.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e1/a0/user.agarabag.37271399._000207.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/25/b8/user.agarabag.37271399._000208.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/3c/1b/user.agarabag.37271399._000209.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d2/d4/user.agarabag.37271399._000210.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d9/34/user.agarabag.37271399._000211.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/29/07/user.agarabag.37271399._000212.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/18/e5/user.agarabag.37271399._000213.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/eb/b5/user.agarabag.37271399._000214.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6b/7f/user.agarabag.37271399._000215.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a1/f4/user.agarabag.37271399._000216.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f1/44/user.agarabag.37271399._000217.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e2/8b/user.agarabag.37271399._000218.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c6/0b/user.agarabag.37271399._000219.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/50/0b/user.agarabag.37271399._000220.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/83/c4/user.agarabag.37271399._000221.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/65/e4/user.agarabag.37271399._000222.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e5/53/user.agarabag.37271399._000223.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/50/c1/user.agarabag.37271399._000224.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/dd/fd/user.agarabag.37271399._000225.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/19/db/user.agarabag.37271399._000226.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/97/76/user.agarabag.37271399._000227.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ea/7c/user.agarabag.37271399._000228.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ba/96/user.agarabag.37271399._000229.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e3/cd/user.agarabag.37271399._000230.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b2/b1/user.agarabag.37271399._000231.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/af/ff/user.agarabag.37271399._000232.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a7/03/user.agarabag.37271399._000233.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ba/92/user.agarabag.37271399._000234.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ab/fb/user.agarabag.37271399._000235.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c8/6e/user.agarabag.37271399._000236.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/26/f7/user.agarabag.37271399._000237.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/63/50/user.agarabag.37271399._000238.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e6/cf/user.agarabag.37271399._000239.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/bc/16/user.agarabag.37271399._000240.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/47/95/user.agarabag.37271399._000241.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b0/23/user.agarabag.37271399._000242.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c2/6e/user.agarabag.37271399._000243.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/0d/ee/user.agarabag.37271399._000244.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/eb/bb/user.agarabag.37271399._000245.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/c3/c7/user.agarabag.37271399._000246.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b8/a2/user.agarabag.37271399._000247.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/3f/14/user.agarabag.37271399._000248.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a0/0a/user.agarabag.37271399._000249.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d8/0c/user.agarabag.37271399._000250.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/3d/b2/user.agarabag.37271399._000251.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/00/ef/user.agarabag.37271399._000252.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/60/05/user.agarabag.37271399._000253.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/a2/05/user.agarabag.37271399._000254.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8b/4e/user.agarabag.37271399._000255.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/00/f1/user.agarabag.37271399._000256.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b5/41/user.agarabag.37271399._000257.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/36/fa/user.agarabag.37271399._000258.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/cd/60/user.agarabag.37271399._000259.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/af/b2/user.agarabag.37271399._000260.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/95/b3/user.agarabag.37271399._000261.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/b2/10/user.agarabag.37271399._000262.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/02/e1/user.agarabag.37271399._000263.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/07/23/user.agarabag.37271399._000264.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/8b/28/user.agarabag.37271399._000265.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6f/20/user.agarabag.37271399._000266.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f7/e5/user.agarabag.37271399._000267.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/e3/21/user.agarabag.37271399._000268.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/4b/eb/user.agarabag.37271399._000269.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/0b/9d/user.agarabag.37271399._000270.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/d5/67/user.agarabag.37271399._000271.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/ba/6b/user.agarabag.37271399._000272.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/af/e6/user.agarabag.37271399._000273.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/60/fd/user.agarabag.37271399._000274.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/97/e2/user.agarabag.37271399._000275.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/58/d9/user.agarabag.37271399._000276.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/cc/a8/user.agarabag.37271399._000277.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/34/fa/user.agarabag.37271399._000278.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/79/6a/user.agarabag.37271399._000279.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/6f/54/user.agarabag.37271399._000280.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/60/fa/user.agarabag.37271399._000281.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f3/c4/user.agarabag.37271399._000282.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/5e/5b/user.agarabag.37271399._000283.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/f8/cb/user.agarabag.37271399._000284.output.root',
 '/global/cfs/cdirs/atlas/projecta/atlas/atlaslocalgroupdisk/rucio/user/agarabag/72/e4/user.agarabag.37271399._000285.output.root']

for index in tqdm(range(len(file_paths))):
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
    bdt_score_new = []
    event_id = []
    

    for i in range(len(l1)):
        print("processing: ", l1[i])
        f_1 = uproot.open(l1[i]+':CollectionTree')
        events = f_1.arrays(branches, library='ak')

        flatten_event_weight = np.repeat(ak.firsts(events['EventInfoAuxDyn.mcEventWeights']), ak.num(events['DiTauJetsAuxDyn.ditau_pt']))
        flatten_event_id = np.repeat(events['EventInfoAux.eventNumber'], ak.num(events['DiTauJetsAuxDyn.ditau_pt']))
        event_id.append(flatten_event_id)
        bdt_score.append(ak.flatten(events['DiTauJetsAuxDyn.BDTScore']))
        bdt_score_new.append(ak.flatten(events['DiTauJetsAuxDyn.BDTScoreNew']))
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
    h5_file = h5py.File(f'jz1_flatten_{index}.h5', 'w')
    # Create datasets in the H5 file
    h5_file.create_dataset('event_id', data=ak.to_numpy(ak.concatenate(event_id)), compression='gzip')
    h5_file.create_dataset('bdt_score', data=ak.to_numpy(ak.concatenate(bdt_score)), compression='gzip')
    h5_file.create_dataset('bdt_score_new', data=ak.to_numpy(ak.concatenate(bdt_score_new)), compression='gzip')
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

