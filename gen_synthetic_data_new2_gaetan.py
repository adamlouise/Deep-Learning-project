# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:12:07 2019

Pipeline to create synthetic data with a given protocol lending itself to
easy rotations (here the HCP-MGH multi-HARDI protocol) in a form suitable for
training a (moderately) deep network.

The script also performs the initial NNLS estimation and stores the estimated
weights, possibly in a compact (sparse) form to save disk space.

April 26, 2019: removed normalization to unit sum of NNLS weights

Benchmarking on Rastaban CentOS 7 with prerotated dictionaries:
    10*6 samples: 6734.71s (=1h52m14.71s = 112m14.71s)
Quite longer without using prerotated dictionaries.


Online estimation of peak directions:
    - arbitrarily select single-fascicle response function from dictionary
    - or select subset of dictionary colums and fit DTI model on all of them
        in noiseless scenario
    - or just impose a tensor with diffusivities chosen arbitrarily (although
        guided by dictionary)
    - when only a single voxel is available and it contains a crossing, it is
    impossible to estimate the single-fascicle response in the data driven
    ways proposed in Dipy
    - response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
        response typically (evals, S0mean)
        typically (array([ 0.0014, 0.00029, 0.00029]), 416.206)
        the axial diffusivity of this tensor should be around 5 times
        larger than the radial diffusivity ratio ~0.20
In our 782-atom MGH-HCP hexagonal packing dictionary at D=2.2um^2/ms,
np.mean(tenfit.evals, axis=0) = array([0.00215257, 0.00022733, 0.00022463])
and
np.median(tenfit.evals, axis=0) =  array([0.00213064, 0.00017771, 0.00017323])

@author: rensonnetg
"""
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as scio
import sys
import time
import pickle

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel


# See check_synthetic_data.py for explanations of the lines below
path_to_utils = os.path.join('python_functions')
path_to_utils = os.path.abspath(path_to_utils)
if path_to_utils not in sys.path:
    sys.path.insert(0, path_to_utils)
import mf_utils as util


# ---- Set input parameters here -----
use_prerot = False  # use pre-rotated dictionaries
sparse = True  # store data sparsely to save space
save_res = False  # save mat file containing data
save_DW_image = False
save_DW_noisy = False

SNR_dist = 'uniform'  # 'uniform' or 'triangular'
num_samples = 1000
save_dir = 'synthetic_data'  # destination folder

# Initiate random number generator (to make results reproducible)
rand_seed = 141414
np.random.seed(rand_seed)


# %% Load DW-MRI protocol from Human Connectome Project (HCP)
schemefile = os.path.join('real_data', 'hcp_mgh_1003.scheme1')
sch_mat = np.loadtxt(schemefile, skiprows=1)  # only DWI, no b0s
bvalfile = os.path.join('real_data', 'bvals.txt')
bvecfile = os.path.join('real_data', 'bvecs.txt')
bvals = np.loadtxt(bvalfile)  # NOT in SI units, in s/mm^2
ind_b0 = np.where(bvals <= 1e-16)[0]
ind_b = np.where(bvals > 1e-16)[0]
num_B0 = ind_b0.size
sch_mat_b0 = np.zeros((sch_mat.shape[0] + num_B0, sch_mat.shape[1]))
sch_mat_b0[ind_b0, 4:] = sch_mat[0, 4:]
sch_mat_b0[ind_b, :] = sch_mat
num_mris = sch_mat_b0.shape[0]

# %% Load single-fascicle canonical dictionary
ld_singfasc_dic = util.loadmat('MC_dictionary_hcp.mat')
# The single-fascicle dictionary stored in the matfile contains all the b0
# images first followed by the diffusion-weighted signals. We reorder the
# acquisitions to match the acquisition protocol. This is the canonical
# dictionary.
if not use_prerot:
    dic_sing_fasc = np.zeros(ld_singfasc_dic['dic_fascicle_refdir'].shape)
    dic_sing_fasc[ind_b0,
                  :] = ld_singfasc_dic['dic_fascicle_refdir'][:num_B0, :]
    dic_sing_fasc[ind_b,
                  :] = ld_singfasc_dic['dic_fascicle_refdir'][num_B0:, :]
    refdir = np.array([0.0, 0.0, 1.0])

# Paramètres du protocole
num_atoms = ld_singfasc_dic['dic_fascicle_refdir'].shape[1]
WM_DIFF = ld_singfasc_dic['WM_DIFF']
S0_fasc = ld_singfasc_dic['S0_fascicle']
sig_csf = ld_singfasc_dic['sig_csf']  # already T2-weighted as well
subinfo = ld_singfasc_dic['subinfo']  # just used for displaying results

S0_max = np.max(S0_fasc)
assert num_atoms == len(subinfo['rad']), "Inconsistency dictionaries"


# %% Generate synthetic acquisition
M0 = 500
num_fasc = 2
nu_min = 0.15
nu_max = 1 - nu_min
SNR_min = 50
SNR_max = 100
num_coils = 1
crossangle_min = 30 * pi/180  # ! increased from 15deg otherwise CSD fails
cos_min = np.cos(crossangle_min)

# Estimate RAM requirements
RAM_dense = (num_fasc*num_atoms*num_samples*8  # NNLS weights
             + num_fasc*num_samples*4  # IDs
             + num_fasc*num_samples*8  # nus
             + num_fasc*num_samples*(3-2*use_prerot)*8  # orientations
             + num_samples*8  # SNRs
             )
if RAM_dense > 1e9 and not sparse:
    raise ValueError("At least %5.4f Gb of RAM required. "
                     "Sparse mode should be used." % RAM_dense/1e9)

starttime = time.time()
time_rot_hist = np.zeros(num_samples)
time_nnls_hist = np.zeros(num_samples)
time_est_o = np.zeros(num_samples)

# Prepare memory
IDs = np.zeros((num_samples, num_fasc), dtype=np.int32)
nus = np.zeros((num_samples, num_fasc))
SNRs = np.zeros(num_samples)

DW_image_store = np.zeros((552, num_samples))
DW_noisy_store = np.zeros((552, num_samples))

# memory for storing groundtruth fascicle directions
if use_prerot:
    orientations = np.zeros((num_samples, num_fasc))
else:
    orientations = np.zeros((num_samples, num_fasc, 3))

# memory for storing estimated directions from DW signal
est_orientations = np.zeros((num_samples, num_fasc, 3))

# memory for storing NNLS weights
if sparse:
    sparsity = 0.01  # expected proportion of nnz atom weights per fascicle
    nnz_pred = int(np.ceil(sparsity * num_atoms * num_samples * num_fasc))
    # Store row and column indices of the dense weight matrix
    w_idx = np.zeros((nnz_pred, 2), dtype=np.int64)  # 2 is 2 !
    # Store weights themselves
    w_data = np.zeros(nnz_pred, dtype=np.float64)
else:
    w_store = np.zeros((num_samples, num_fasc*num_atoms), dtype=np.float)

nnz_hist = np.zeros(num_samples)  # always useful even in non sparse mode

# Quantities used repeatedly for CSD peak estimation
# use largest sphere available in Dipy
odf_sphere = get_sphere('repulsion724')

gam = util.get_gyromagnetic_ratio('H')
G = sch_mat_b0[:, 3]
Deltas = sch_mat_b0[:, 4]
deltas = sch_mat_b0[:, 5]
bvals = (gam*G*deltas)**2*(Deltas-deltas/3)  # in SI units s/m^2
bvecs = sch_mat_b0[:, :3]
gtab = gradient_table(bvals/1e6, bvecs)  # bvals in s/mm^2
num_dwi = np.sum(bvals > 0)

MAX_SH_ORDER = 12
sh_max_vals = np.arange(2, MAX_SH_ORDER+1, 2)
# base sizes is the number of free coefficients to estimate, i.e. the
# degrees of freedom of the model
base_sizes = (sh_max_vals+1)*(sh_max_vals+2)//2
# the number of free parameters to estimate must be less than the number
# of data points (measurements) available
i_shmax = np.where(num_dwi >= base_sizes)[0][-1]
sh_max_order = sh_max_vals[i_shmax]


def get_csd_peaks(DWI_image_noisy, sch_mat_b0, num_fasc):
    '''Get peak orientations using constrained spherical deconvolution (CSD)
    '''
    S0mean = np.mean(DW_image_noisy[bvals == 0])
    # mean on dictionary is array([0.00215257, 0.00022733, 0.00022463])
    # median is  array([0.00213064, 0.00017771, 0.00017323])
    sing_fasc_response = (np.array([0.0020, 0.00024, 0.00024]), S0mean)

    csd_mod = ConstrainedSphericalDeconvModel(gtab,
                                              sing_fasc_response,
                                              sh_order=sh_max_order)
    # we cheat a bit by using groundtruth information on the angle to set the
    # minimum separation angle:

    min_sep_angle = 0.9*crossangle_min*180/np.pi  # in degrees !
    peaks = peaks_from_model(model=csd_mod,
                             data=DW_image_noisy,
                             sphere=odf_sphere,
                             relative_peak_threshold=0.9*nu_min/nu_max,
                             min_separation_angle=min_sep_angle,
                             sh_order=sh_max_order,
                             npeaks=num_fasc)
    # direction of peaks, shape (npeaks, 3). pdirs[0] is main peak,
    # peak_dirs[1] secondary peak. peak_dirs[1] could be [0, 0, 0] if no peak
    # was detected.
    return peaks.peak_dirs


# prepare memory for dictionary
dictionary = np.zeros((num_mris, num_fasc * num_atoms), dtype=np.float64)
if not use_prerot:
    # Prefill first part of dictionary in case first direction fixed
    # will be overwritten otherwise
    dictionary[:, :num_atoms] = dic_sing_fasc

nnz_cnt = 0  # non-zero entries (just for sparse case)

plt.close('all')

# Tester une direction 1 constante mais pas l'axe z
# cyldir_1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
# dic_sing_fasc1 = util.rotate_atom(dic_sing_fasc,
#                                  sch_mat_b0, refdir, cyldir_1,  # once and for all
#                                  WM_DIFF, S0_fasc[:, ID_1])

for i in range(num_samples):
    nu1 = nu_min + (nu_max - nu_min) * np.random.rand()
    nu2 = 1 - nu1
    ID_1 = np.random.randint(0, num_atoms)
    ID_2 = np.random.randint(0, num_atoms)
    if SNR_dist == 'triangular':
        SNR = np.random.triangular(SNR_min, SNR_min, SNR_max, 1)
    elif SNR_dist == 'uniform':
        SNR = np.random.uniform(SNR_min, SNR_max, 1)
    else:
        raise ValueError("Unknown SNR distribution %s" % SNR_dist)

    sigma_g = S0_max/SNR

    # NEW way to create groundtruth
    # norm1 = -1
    # while norm1 <= 0:
    #     cyldir_1 = np.random.randn(3)
    #     norm1 = np.linalg.norm(cyldir_1, 2)
    # cyldir_1 = cyldir_1/norm1  # get unit vector

    # generate second direction making sure it's not too close to first
    cyldir_1 = refdir # enlever su premiere direction aleatoire
    cyldir_2 = cyldir_1.copy()
    while np.abs(np.dot(cyldir_1, cyldir_2)) > np.cos(crossangle_min):
        norm2 = -1
        while norm2 <= 0:
            cyldir_2 = np.random.randn(3)
            norm2 = np.sqrt(np.sum(cyldir_2**2))
        cyldir_2 = cyldir_2/norm2
    crossang = np.arccos(np.abs(np.dot(cyldir_1, cyldir_2))) * 180/np.pi

    # sig_fasc1 = util.rotate_atom(dic_sing_fasc[:, ID_1],
    #                              sch_mat_b0, refdir, cyldir_1,
    #                              WM_DIFF, S0_fasc[:, ID_1])
    sig_fasc1 = dic_sing_fasc[:, ID_1]
    sig_fasc2 = util.rotate_atom(dic_sing_fasc[:, ID_2],
                                 sch_mat_b0, refdir, cyldir_2,
                                 WM_DIFF, S0_fasc[:, ID_2])

    DW_image = nu1 * sig_fasc1 + nu2 * sig_fasc2

    # Simulate noise and MRI scanner scaling
    DW_image_store[:, i] = DW_image

    DW_image_noisy = util.gen_SoS_MRI(DW_image, sigma_g, num_coils)
    DW_image_noisy = M0 * DW_image_noisy

    DW_noisy_store[:, i] = DW_image_noisy

    start_est_o = time.time()
    # Estimate peak directions from noisy signal
    peaks = get_csd_peaks(DW_image_noisy, sch_mat_b0, num_fasc)

    # Analyze result of CSD (just for displaying progress). You only need to
    # store the groundtruth and estimated directions to compute all these
    # metrics afterwards.
    num_pk_detected = np.sum(np.sum(np.abs(peaks), axis=1) > 0)

    if num_pk_detected < num_fasc:
        # There should always be at least 1 detected peak because the ODF
        # always has a max.
        # Pick second direction randomly on a cone centered around first
        # direction with angle set to min separation angle above.
        # Using the same direction will lead to problems with NNLS down the
        # line with a poorly conditioned matrix (since the same submatrix
        # will be repeated)
        peaks[1] = peaks[0].copy()
        rot_ax = util.get_perp_vector(peaks[1])
        peaks[1] = util.rotate_vector(peaks[0], rot_ax, crossangle_min)

    # Match detected peaks to groundtruth peaks, i.e. either swap peaks[0]
    # and peaks[1] or don't swap. Rigorously, this should not be done but it
    # considerably eases posterior analyses because there is a greater chance
    # that fascicles match.

    # Compare two possibilities in terms of absolute dot product (inversely
    # proportional to angular separation)
    dp_1 = (np.abs(np.dot(peaks[0], cyldir_1))
            + np.abs(np.dot(peaks[1], cyldir_2)))
    dp_2 = (np.abs(np.dot(peaks[0], cyldir_2))
            + np.abs(np.dot(peaks[1], cyldir_1)))
    if dp_1 > dp_2:
        est_dir1 = peaks[0]
        est_dir2 = peaks[1]
    else:
        est_dir1 = peaks[1]
        est_dir2 = peaks[0]
    cos_11 = np.clip(np.abs(np.dot(est_dir1, cyldir_1)), 0, 1)
    cos_22 = np.clip(np.abs(np.dot(est_dir2, cyldir_2)), 0, 1)
    mean_ang_err = 0.5*(np.arccos(cos_11) + np.arccos(cos_22))*180/np.pi
    crossang_est = np.arccos(np.abs(np.dot(est_dir1, est_dir2))) * 180/np.pi
#    print('Sample %d/%d : SNR=%d, nu1=%gn ang=%gdeg\n\tdetected %d peak(s),'
#          ' mean angular error %g deg, ang sep %gdeg.' %
#          (i+1, num_samples, SNR, nu1, crossang,
#           num_pk_detected, mean_ang_err, crossang_est))

    time_est_o[i]= time.time()- start_est_o

    # Create big dictionary. Rotate dic_sing_fasc along estimated
    # cyldir_1 and cyldir_2.
    # !! In theory we should use peaks[0] and peaks[1] directly from the peak
    # estimation routine.
    # However to simplify the analysis of estimates down the line, we will
    # use est_dir1 and est_dir2 (which is simply a 'smart' swap). That way
    # the estimates of fascicle 1 will match the groundtruth fascicle 1 and
    # we won't have to swap.
    start_rot = time.time()
    dictionary[:, :num_atoms] = util.rotate_atom(dic_sing_fasc,
                                                 sch_mat_b0,
                                                 refdir,
                                                 est_dir1,
                                                 WM_DIFF, S0_fasc)
    dictionary[:, num_atoms:] = util.rotate_atom(dic_sing_fasc,
                                                 sch_mat_b0,
                                                 refdir,
                                                 est_dir2,
                                                 WM_DIFF, S0_fasc)
    time_rot_hist[i] = time.time() - start_rot


    # Solve NNLS
    start_nnls = time.time()
    norm_DW = np.max(DW_image_noisy[sch_mat_b0[:, 3] == 0])
    (w_nnls,
     PP,
     _) = util.nnls_underdetermined(dictionary,
                                    DW_image_noisy/norm_DW)
                                    
    time_nnls_hist[i] = time.time() - start_nnls

    # Store
    IDs[i, :] = np.array([ID_1, ID_2])
    nus[i, :] = np.array([nu1, nu2])
    SNRs[i] = SNR
    nnz_hist[i] = PP.size
    orientations[i, 0, :] = cyldir_1
    orientations[i, 1, :] = cyldir_2
    # Again, rigorously this should be est_orientations[i, ...] = peaks
    est_orientations[i, 0, :] = est_dir1
    est_orientations[i, 1, :] = est_dir2
    if sparse:
        # Check size and double it if needed
        if nnz_cnt + PP.size > w_data.shape[0]:
            w_idx = np.concatenate((w_idx, np.zeros(w_idx.shape)), axis=0)
            w_data = np.concatenate((w_data, np.zeros(w_data.shape)), axis=0)
            print("Doubled size of index and weight arrays after sample %d "
                  "(adding %d non-zero elements to %d, exceeding arrays' "
                  " size of %d)"
                  % (i+1, PP.size, nnz_cnt, w_data.shape[0]))
        w_data[nnz_cnt:nnz_cnt+PP.size] = w_nnls[PP]

        w_idx[nnz_cnt:nnz_cnt+PP.size, 0] = i  # row indices
        w_idx[nnz_cnt:nnz_cnt+PP.size, 1] = PP  # column indices

        nnz_cnt += PP.size
    else:
        w_store[i, :] = w_nnls

    # Log progress
    if i % 1000 == 0:
        print('Generated voxel %d/%d' % (i+1, num_samples))

if sparse:
    # Discard unused memory
    w_idx = w_idx[:nnz_cnt, :]
    w_data = w_data[:nnz_cnt]
time_elapsed = time.time() - starttime
print('%d samples created in %g sec.' % (num_samples, time_elapsed))

print('Time estimation of orientation', print(sum(time_est_o)))
print('Time rotation of dictionary', print(sum(time_rot_hist)))
print('Time nnls', print(sum(time_nnls_hist)))


# %% Save results
# plus d'actualité: Note! The DW-MRI signal is not stored! just the output of NNLS
# --> ca j'ai changé :-)
# en fait non, seulement ajouter pour < 500 000, si c est + matlab ne veut pas

print('--- save dico ----', save_res)

mdict = {'rand_seed': rand_seed,
         'M0': M0,
         'num_fasc': num_fasc,
         'nu_min': nu_min,
         'nu_max': nu_max,
         'SNR_min': SNR_min,
         'SNR_max': SNR_max,
         'SNR_dist': SNR_dist,
         'num_coils': num_coils,
         'crossangle_min': crossangle_min,
         'num_fasc': num_fasc,
         'num_atoms': num_atoms,
         'WM_DIFF': WM_DIFF,
         'S0_fasc': S0_fasc,
         'S0_max': S0_max,
         'sig_csf': sig_csf,
         'subinfo': subinfo,
         'sch_mat_b0': sch_mat_b0,
         'num_samples': num_samples,
         'IDs': IDs,
         'nus': nus,
         'SNRs': SNRs,
         'nnz_hist': nnz_hist,
         'orientations': orientations,
         'est_orientations': est_orientations,  # new !
         'use_prerot': use_prerot,
         'sparse': sparse,
         #'DW_image_store':DW_image_store,
         #'DW_noisy_store':DW_noisy_store
         }
if use_prerot:
    mdict['directions'] = directions
if sparse:
    mdict['w_idx'] = w_idx
    mdict['w_data'] = w_data
else:
    mdict['w_store'] = w_store

if save_res:
    if SNR_dist == 'uniform':
        SNR_str = 'uniform'
    elif SNR_dist == 'triangular':
        SNR_str = '_triangSNR'
    else:
        raise ValueError('Unknown SNR distribution %s' % SNR_dist)
    fname = os.path.join(save_dir, "training_data%s_%d_samples_lou_version8" %
                         (SNR_str, num_samples))
    scio.savemat(fname,
                 mdict,
                 format='5',
                 long_field_names=False,
                 oned_as='column')

''' Assess quality of the estimation of peak directions.
- STRATEGY A
Find for each sample if estimated direction 1 corresponds to groundtruth (gt)
 cylinder direction 1 and est. dir. 2 to gt dir. 2, or if it is the other
 way around (est dir 1 with gt dir 2 and est dir 2 with gt dir 1) based on
 the max absolute dot product (equivalently, minimum total angle separation).

     dp_option1_1 = np.abs(np.sum(orientations[:, 0, :] * est_orientations[:, 0, :], axis=-1))
     dp_option1_2 = np.abs(np.sum(orientations[:, 1, :] * est_orientations[:, 1, :], axis=-1))
     dp_option2_1 = np.abs(np.sum(orientations[:, 0, :] * est_orientations[:, 1, :], axis=-1))
     dp_option2_2 = np.abs(np.sum(orientations[:, 1, :] * est_orientations[:, 0, :], axis=-1))

     # remove numerical round off errors, make sure results are in [0, 1]
     dp_option1_1 = np.clip(dp_option1_1, 0, 1)
     dp_option1_2 = np.clip(dp_option1_2, 0, 1)
     dp_option2_1 = np.clip(dp_option2_1, 0, 1)
     dp_option2_2 = np.clip(dp_option2_2, 0, 1)

     dp_option1 = dp_option1_1 + dp_option1_2
     dp_option2 = dp_option2_1 + dp_option2_2

     is_option1 = dp_option1 >= dp_option2
     # this implies that ~is_option1 corresponds to option 2

     num_samples = orientations.shape[0]
     err_dir1 = np.zeros(num_samples)  # error with respect to gt dir 1
     err_dir1[is_option1] = np.arccos(dp_option1_1[is_option1]) * 180/np.pi
     err_dir1[~is_option1] = np.arccos(dp_option2_1[~is_option1]) * 180/np.pi

     err_dir2 = np.zeros(num_samples)  # error with respect to gt dir 2
     err_dir2[is_option1] = np.arccos(dp_option1_2[is_option1]) * 180/np.pi
     err_dir2[~is_option1] = np.arccos(dp_option2_2[~is_option1]) * 180/np.pi

     dp_est = np.sum(est_orientations[:, 0, :]*est_orientations[:, 1, :], axis=-1)
     est_ang_sep = np.arccos(np.clip(np.abs(dp_est), 0, 1)) * 180/np.pi

     mean_ang_err = (err_dir1 + err_dir2)/2

- STRATEGY B
Give priority to largest detected peak (assumed to match largest groundtruth
peak), assign other detected peak to other groundtruth peak by default

Detect missed fascicles by orientation estimation procedure:
    peaks_L1_norm = np.sum(np.abs(est_orientations), axis=-1) # shape (num_samples, 2)
    num_peaks_det = np.sum(peaks_L1_norm > 0, axis=-1)  # shape (num_samples,)
    peaks_missed = 2 - num_peaks_det  # shape (num_samples,)

'''

#%% Save DW_image with pickle files

if save_DW_image:
    filename1 = os.path.join(save_dir, "DW_image_store_%s_%d__lou_version8_24_5" %
                              (SNR_str, num_samples))
    with open(filename1, 'wb') as f:
        pickle.dump(DW_image_store, f)
        f.close()

if save_DW_noisy:       
    filename2 = os.path.join(save_dir, "DW_noisy_store_%s_%d__lou_version8" %
                              (SNR_str, num_samples))
    with open(filename2, 'wb') as f:
        pickle.dump(DW_noisy_store, f)
        f.close()