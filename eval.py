#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage
-------------

# You must first set appropriately the following variables below: 
# 'output_path', 'chime5_input_path', 'reverberant_librichime5_input_path',
# 'librimix_input_path'.

# run the 'remixit-vad' baseline on CHiME-5, reverberant LibriCHiME-5, Librimix
python eval.py --run-baseline --model remixit-vad

# evaluate the 'remixit-vad' baseline results on CHiME-5, 
# reverberant LibriCHiME-5, and Librimix, and compute the performance scores
# on the unprocessed input signals.
python eval.py --eval-baseline --input-scores --model remixit-vad

-------------

See the description of the functions 'run_baseline' and 'compute_metrics' below
for more information. 

The function 'compute_metrics' is independent of the baseline system, you
may use it to report the results of your system for the UDASE task of CHiME-7.
"""

#%% configuration

"""
You must set the following variables:
    - output_path
    - chime5_input_path
    - reverberant_librichime5_input_path
    - librimix_input_path
"""

# output path to store the results of the baseline
output_path = '/data2/datasets/UDASE-CHiME2023/baseline_results'

######## Config for the CHiME-5 dataset of the UDASE task ########

# path to the input data
chime5_input_path = '/data2/datasets/UDASE-CHiME2023/CHiME-5'
# subsets to process and evaluate
chime5_subsets = ['eval/1']

######## Config for the Reverberant LibriCHiME-5 dataset of the UDASE task ########

# path to the input data
reverberant_librichime5_input_path = '/data2/datasets/UDASE-CHiME2023/reverberant-LibriCHiME-5'
# subsets to process and evaluate
reverberant_librichime5_subsets = ['eval/1', 'eval/2', 'eval/3']

######## Config for the LibriMix dataset ########

# path to the input data
librimix_input_path = '/data/datasets/LibriMix'
# subsets to process and evaluate
librimix_subsets = ['Libri2Mix/wav16k/max/test/mix_single', 
                    'Libri2Mix/wav16k/max/test/mix_both',
                    'Libri3Mix/wav16k/max/test/mix_both']


#%% imports

import numpy as np
import torch
import torchaudio
import baseline.metrics.dnnmos_metric as dnnmos_metric
import baseline.metrics.sisdr_metric as sisdr_metric
import baseline.utils.mixture_consistency as mixture_consistency
import baseline.models.improved_sudormrf as improved_sudormrf
import pandas as pd
import os
import glob
import soundfile as sf
from tqdm import tqdm
import pyloudnorm as pyln
import argparse
import warnings
warnings.filterwarnings("ignore")

#%%

def normalize(x, target_loudness=-30, meter=None, sr=16000):
    """
    LUFS normalization of a signal using pyloudnorm.
    
    Parameters
    ----------
    x : ndarray
        Input signal.
    target_loudness : float, optional
        Target loudness of the output in dB LUFS. The default is -30.
    meter : Meter, optional
        The pyloudnorm BS.1770 meter. The default is None.
    sr : int, optional
        Sampling rate. The default is 16000.

    Returns
    -------
    x_norm : ndarray
        Normalized output signal.
    """
    
    if meter is None:
        meter = pyln.Meter(sr) # create BS.1770 meter
    
    # peak normalize to 0.7 to ensure that the meter does not return -inf
    x = x - np.mean(x)
    x = x/(np.max(np.abs(x)) + 1e-9)*0.7
            
    # measure the loudness first 
    loudness = meter.integrated_loudness(x)
    
    # loudness normalize audio to target_loudness LUFS
    x_norm = pyln.normalize.loudness(x, loudness, target_loudness)  

    return x_norm


def run_baseline(checkpoint, output_path, 
                 chime5_input_path, 
                 chime5_subsets, 
                 reverberant_librichime5_input_path, 
                 reverberant_librichime5_subsets,
                 librimix_input_path,
                 librimix_subsets,
                 save_mix=False, 
                 save_noise=False):
    
    """
    Script to run the baseline on the CHiME-5, Reverberant LibriCHiME-5, and
    LibriMix datasets. 
    
    You must set the variables in configuration section of this file before 
    calling this function.

    Parameters
    ----------
    checkpoint : string
        Path to the baseline model checkpoint.
    output_path : string
        Path to save the results (output signals).
    chime5_input_path : string
        Path to the (preprocessed) CHiME-5 dataset (as provided for the UDASE task).
    chime5_subsets : list of string
        Subsets of the CHiME-5 dataset to process.
    reverberant_librichime5_input_path : string
        Path to the reverberant LibriCHiME-5 dataset.
    reverberant_librichime5_subsets : list of string
        Subsets of the reverberant LibriCHiME-5 dataset to process.
    librimix_input_path : string
        Path to the reverberant LibriMix dataset.
    librimix_subsets : list of string
        Subsets of theLibriMix dataset to process.
    save_mix : boolean, optional
        Boolean indicating if the input noisy mixture signal should be saved. 
        The default is False.
    save_noise : boolean, optional
        Boolean indicating if the estimated noise signal should be saved. 
        The default is False.
    
    Returns
    -------
    None.
    """
            
    sr = 16000
    meter = pyln.Meter(sr)
    
    chime5_output_path = os.path.join(output_path, 'CHiME-5')
    reverberant_librichime5_output_path = os.path.join(output_path, 
                                                       'reverberant-LibriCHiME-5')
    librimix_output_path = os.path.join(output_path, 'LibriMix')    
    
    # model

    model = improved_sudormrf.SuDORMRF(
            out_channels=256,
            in_channels=512,
            num_blocks=8,
            upsampling_depth=7,
            enc_kernel_size=81,
            enc_num_basis=512,
            num_sources=2,
        )

    model.load_state_dict(torch.load(checkpoint))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # process datasets

    datasets = ['chime-5', 'reverberant-librichime-5', 'librimix']

    for dataset in datasets:
        
        if dataset == 'chime-5':
            print('Running the baseline on CHiME-5')
            subsets = chime5_subsets
        elif dataset == 'reverberant-librichime-5':
            print('Running the baseline on Reverberant LibriCHiME-5')
            subsets = reverberant_librichime5_subsets
        elif dataset == 'librimix':
            print('Running the baseline on LibriMix')
            subsets = librimix_subsets
            
        for subset in subsets:

            if dataset == 'chime-5':
                pattern = os.path.join(chime5_input_path, subset, '*.wav')
                curr_output_dir = os.path.join(chime5_output_path, subset)
            elif dataset == 'reverberant-librichime-5':
                pattern = os.path.join(reverberant_librichime5_input_path, subset,'*mix.wav')
                curr_output_dir = os.path.join(reverberant_librichime5_output_path, subset)
            elif dataset == 'librimix':
                pattern = os.path.join(librimix_input_path, subset,'*.wav')
                curr_output_dir = os.path.join(librimix_output_path, subset)
            
            if not os.path.isdir(curr_output_dir):
                os.makedirs(curr_output_dir)
            
            file_list = glob.glob(pattern)

            for ind, mix_file in tqdm(enumerate(file_list), total=len(file_list)):
                            
                file_name = os.path.basename(mix_file)
                file_name = os.path.splitext(file_name)[0]
                
                # Scale the input mixture
                input_mix, _ = torchaudio.load(mix_file) # audio file should be mono channel
                input_mix = input_mix.unsqueeze(1).cuda() 
                input_mix_std = input_mix.std(-1, keepdim=True)
                input_mix_mean = input_mix.mean(-1, keepdim=True)
                input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)
                
                # Perform inference and apply mixture consistency
                with torch.no_grad():
                    estimates = model(input_mix)
                    estimates = mixture_consistency.apply(estimates, input_mix)
            
                # Unscale the input mixture and estimates
                input_mix = input_mix*(input_mix_std + 1e-9) + input_mix_mean
                estimates = estimates*(input_mix_std + 1e-9) + input_mix_mean
            
                input_mix = input_mix.cpu().numpy().squeeze()
                estimates = estimates.cpu().numpy().squeeze()
                speech_est = estimates[0,:]
                noise_est = estimates[1,:]
            
                # normalize to -30 LUFS
                speech_est = normalize(speech_est, target_loudness=-30, meter=meter, sr=16000)
                noise_est = normalize(noise_est, target_loudness=-30, meter=meter, sr=16000)
                input_mix = normalize(input_mix, target_loudness=-30, meter=meter, sr=16000)
                
                # save
                if dataset == 'chime-5' or dataset == 'librimix':
                    mix_id = file_name
                elif dataset == 'reverberant-librichime-5':
                    mix_id = file_name[:-4]
                
                sf.write(os.path.join(curr_output_dir, mix_id + '_output.wav'), speech_est, sr)
                
                if save_noise:
                    sf.write(os.path.join(curr_output_dir, mix_id + '_noise_estimate.wav'), noise_est, sr)
                if save_mix:
                    sf.write(os.path.join(curr_output_dir, mix_id + '_mix.wav'), input_mix, sr)    


def compute_metrics(results_path,
                    chime5_input_path=None, 
                    chime5_subsets=None, 
                    reverberant_librichime5_input_path=None, 
                    reverberant_librichime5_subsets=None,
                    librimix_input_path=None,
                    librimix_subsets=None,
                    compute_input_scores=False):
    """
    Script to compute the objective performance metrics for the UDASE task of 
    the CHiME-7 challenge.
    
    The directory defined by the input variable 'results_path'
    should contain at least one of the three following folders: 
    CHiME-5, LibriMix, reverberant-LibriCHiME-5. Each folder should follow 
    the tree structure shown below.

    <results_path>
        ├── CHiME-5
        │   └── eval
        │       └── 1
        ├── LibriMix
        │   ├── Libri2Mix
        │   │   └── wav16k
        │   │       └── max
        │   │           └── test
        │   │               ├── mix_both
        │   │               └── mix_single
        │   └── Libri3Mix
        │       └── wav16k
        │           └── max
        │               └── test
        │                   └── mix_both
        └── reverberant-LibriCHiME-5
            └── eval
                ├── 1
                ├── 2
                └── 3
                
    At each leaf of this tree, we have a directory that contains the output
    wav files of the system. The naming convention is the following:
    - For CHiME-5, the output signal corresponding to the input signal
    <mix ID>.wav should be named <mix ID>_output.wav. For example, the 
    output signal <results_path>/CHiME-5/eval/1/S01_P01_0_output.wav 
    corresponds to the input signal <chime5_input_path>/eval/1/S01_P01_0.wav
    - For reverberant LibriCHiME-5, the output signal corresponding to the input signal
    <mix ID>_mix.wav should be named <mix ID>_output.wav. For example, the 
    output signal <results_path>/reverberant-LibriCHiME-5/eval/1/S01_P01_0a_output.wav
    corresponds to the input signal at <reverberant_librichime5_input_path>/eval/1/S01_P01_0a_mix.wav
    - For LibriMix, the output signal corresponding to the input signal
    <mix ID>.wav should be named <mix ID>_output.wav. For example, the 
    output signal <results_path>/LibriMix/Libri2Mix/wav16k/max/test/mix_single/61-70968-0000_8455-210777-0012_output.wav
    corresponds to the input signal at <librimix_input_path>/Libri2Mix/wav16k/max/test/mix_single/61-70968-0000_8455-210777-0012.wav
        
    To compute the results for a given dataset X with X in 
    (chime5, reverberant_librichime5, librimix), you should set the input
    variables 'X_input_path' and 'X_subsets' appropriately (see Parameters
    section below). If 'X_input_path' and 'X_subsets' are set to None 
    (default), results will not be computed for the dataset X.
    
    This function will save the objective performance results in a csv file 
    'results.csv' located in the folder CHiME-5, LibriMix or 
    reverberant-LibriCHiME-5 (in the above tree structure), depending on the 
    dataset.
    
    If the input variable 'compute_input_scores' is set to True, an
    additional csv file 'results_unprocessed.csv' will be saved at the same
    location as 'results.csv'. It will contain the performance scores for the
    unprocessed noisy speech signals.
    
    See Example Usage below.

    Parameters
    ----------
    
    results_path : string
        Path where the results (output signals) are saved.
    chime5_input_path : string, optional
        Path to the (preprocessed) CHiME-5 dataset (as provided for the UDASE task).
    chime5_subsets : list of string, optional
        Subsets of the CHiME-5 dataset to process.
    reverberant_librichime5_input_path : string, optional
        Path to the reverberant LibriCHiME-5 dataset.
    reverberant_librichime5_subsets : list of string, optional
        Subsets of the reverberant LibriCHiME-5 dataset to process.
    librimix_input_path : string, optional
        Path to the reverberant LibriMix dataset.
    librimix_subsets : list of string, optional
        Subsets of theLibriMix dataset to process.
    
    compute_input_scores : boolean, optional
        Boolean indicating if the metrics should also be computed on the 
        input unprocessed noisy speech signal. The default is False.

    Returns
    -------
    None.
    
    Example usage
    -------------
    
    # path to the baseline output signals
    results_path = '/data2/datasets/UDASE-CHiME2023/baseline_results/remixit-vad'

    ######## Config for the CHiME-5 dataset of the UDASE task ########

    # path to the input data
    chime5_input_path = '/data2/datasets/UDASE-CHiME2023/CHiME-5'
    # subsets to process
    chime5_subsets = ['eval/1']

    ######## Config for the Reverberant LibriCHiME-5 dataset of the UDASE task ########

    # path to the input data
    reverberant_librichime5_input_path = '/data2/datasets/UDASE-CHiME2023/reverberant-LibriCHiME-5'
    # subsets to process
    reverberant_librichime5_subsets = ['eval/1', 'eval/2', 'eval/3']

    ######## Config for the LibriMix dataset ########

    # path to the input data
    librimix_input_path = '/data/datasets/LibriMix'
    # subsets to process
    librimix_subsets = ['Libri2Mix/wav16k/max/test/mix_single', 
                        'Libri2Mix/wav16k/max/test/mix_both',
                        'Libri3Mix/wav16k/max/test/mix_both']
    
    compute_metrics(results_path=results_path,
                    chime5_input_path=chime5_input_path, 
                    chime5_subsets=chime5_subsets, 
                    reverberant_librichime5_input_path=reverberant_librichime5_input_path, 
                    reverberant_librichime5_subsets=reverberant_librichime5_subsets,
                    librimix_input_path=librimix_input_path,
                    librimix_subsets=librimix_subsets, 
                    compute_input_scores=True)

    """
        
    if compute_input_scores:    
        meter = pyln.Meter(16000)
    
    # CHiME-5
    
    if chime5_input_path is not None and chime5_subsets is not None:
    
        print('Compute results on CHiME-5')
        
        chime5_output_path = os.path.join(results_path, 'CHiME-5')
        chime5_df = pd.DataFrame(columns=['subset', 
                                          'input_file_name', 
                                          'output_file_name', 
                                          'SIG_MOS', 
                                          'BAK_MOS', 
                                          'OVR_MOS'])
        if compute_input_scores:
            unprocessed_chime5_df = pd.DataFrame(columns=['subset',
                                                          'input_file_name',
                                                          'output_file_name',
                                                          'SIG_MOS',
                                                          'BAK_MOS',
                                                          'OVR_MOS'])
        
        for subset in chime5_subsets:
            
            print(subset)
        
            input_file_list = glob.glob(os.path.join(chime5_input_path, subset,
                                                     '*.wav'))
                    
            for ind, input_file in tqdm(enumerate(input_file_list), 
                                        total=len(input_file_list)):
                
                mix_id = os.path.basename(input_file)
                mix_id = os.path.splitext(mix_id)[0]
                
                output_file = os.path.join(chime5_output_path, subset, 
                                           mix_id + '_output' + '.wav')
                assert os.path.isfile(output_file)
                
                speech_est, sr = sf.read(output_file)
                dnsmos_res = dnnmos_metric.compute_dnsmos(speech_est, fs=sr)
                    
                row = [subset, os.path.basename(input_file), 
                       os.path.basename(output_file), dnsmos_res['sig_mos'],
                       dnsmos_res['bak_mos'], dnsmos_res['ovr_mos']]
                
                chime5_df.loc[len(chime5_df)] = row
                
                if compute_input_scores:
                    
                    mix, sr = sf.read(input_file)
                    mix = normalize(mix, target_loudness=-30, meter=meter, sr=16000)
                    
                    dnsmos_res = dnnmos_metric.compute_dnsmos(mix, fs=sr)
                        
                    row = [subset, os.path.basename(input_file), 
                           os.path.basename(input_file), dnsmos_res['sig_mos'],
                           dnsmos_res['bak_mos'], dnsmos_res['ovr_mos']]
                    
                    unprocessed_chime5_df.loc[len(unprocessed_chime5_df)] = row
                    
        csv_file = os.path.join(chime5_output_path, 'results.csv')
        chime5_df.to_csv(csv_file)
        
        if compute_input_scores:
            csv_file = os.path.join(chime5_output_path, 'results_unprocessed.csv')
            unprocessed_chime5_df.to_csv(csv_file)    
        
    # Reverberant LibriCHiME-5
    
    if reverberant_librichime5_input_path is not None and reverberant_librichime5_subsets is not None:
        
        print('Compute results on Reverberant LibriCHiME-5')
        
        reverberant_librichime5_output_path = os.path.join(results_path, 
                                                           'reverberant-LibriCHiME-5')
        
        reverberant_librichime5_df = pd.DataFrame(columns=['subset', 
                                                           'input_file_name', 
                                                           'output_file_name', 
                                                           'SI-SDR'])
        if compute_input_scores:
            unprocessed_reverberant_librichime5_df = pd.DataFrame(columns=['subset',
                                                                           'input_file_name',
                                                                           'output_file_name',
                                                                           'SI-SDR'])
    
        for subset in reverberant_librichime5_subsets:
            
            print(subset)
            
            input_file_list = glob.glob(os.path.join(reverberant_librichime5_input_path, 
                                                     subset, '*mix.wav'))
            
            for ind, input_file in tqdm(enumerate(input_file_list), 
                                        total=len(input_file_list)):
            
                mix_id = os.path.basename(input_file)
                mix_id = os.path.splitext(mix_id)[0]
                mix_id = mix_id[:-4]
                
                speech_ref_file = os.path.join(os.path.dirname(input_file), 
                                               mix_id + '_speech.wav')
                assert os.path.isfile(speech_ref_file)
                
                output_file = os.path.join(reverberant_librichime5_output_path, subset, 
                                           mix_id + '_output' + '.wav')
                assert os.path.isfile(output_file)
            
                speech_est, sr = sf.read(output_file)
                speech_ref, sr = sf.read(speech_ref_file)
                
                si_sdr = sisdr_metric.compute_sisdr(speech_est, speech_ref)
                
                row = [subset, os.path.basename(input_file), 
                       os.path.basename(output_file), si_sdr]
                
                reverberant_librichime5_df.loc[len(reverberant_librichime5_df)] = row
                
                if compute_input_scores:
                    
                    mix, sr = sf.read(input_file)
                    
                    si_sdr = sisdr_metric.compute_sisdr(mix, speech_ref)
                    
                    row = [subset, os.path.basename(input_file), 
                           os.path.basename(input_file), si_sdr]
                    
                    unprocessed_reverberant_librichime5_df.loc[len(unprocessed_reverberant_librichime5_df)] = row
                    
        csv_file = os.path.join(reverberant_librichime5_output_path, 'results.csv')
        reverberant_librichime5_df.to_csv(csv_file)
    
        if compute_input_scores:     
            csv_file = os.path.join(reverberant_librichime5_output_path, 'results_unprocessed.csv')
            unprocessed_reverberant_librichime5_df.to_csv(csv_file)
        
    # LibriMix
    
    if librimix_input_path is not None and librimix_subsets is not None:
    
        print('Compute results on LibriMix')
        
        librimix_output_path = os.path.join(results_path, 'LibriMix')
        
        librimix_df = pd.DataFrame(columns=['subset', 
                                            'input_file_name', 
                                            'output_file_name', 
                                            'SI-SDR'])
        
        if compute_input_scores:
            unprocessed_librimix_df = pd.DataFrame(columns=['subset', 
                                                            'input_file_name', 
                                                            'output_file_name', 
                                                            'SI-SDR'])
        
        for subset in librimix_subsets:
            
            print(subset)
            
            input_file_list = glob.glob(os.path.join(librimix_input_path, 
                                                     subset, '*.wav'))
            
            for ind, input_file in tqdm(enumerate(input_file_list), 
                                        total=len(input_file_list)):
            
                mix_id = os.path.basename(input_file)
                mix_id = os.path.splitext(mix_id)[0]
                
                # read speech estimate
                output_file = os.path.join(librimix_output_path, subset, 
                                           mix_id + '_output' + '.wav')
                assert os.path.isfile(output_file)
            
                speech_est, sr = sf.read(output_file)
                
                # read speech reference
                if subset=='Libri2Mix/wav16k/max/test/mix_single':
                    sources = ['s1']
                elif subset=='Libri2Mix/wav16k/max/test/mix_both':
                    sources = ['s1', 's2']
                elif subset=='Libri3Mix/wav16k/max/test/mix_both':
                    sources = ['s1', 's2', 's3']
                
                speech_ref_list = []
                for source in sources:
                    
                    speech_ref_file = os.path.join(os.path.dirname(os.path.dirname(input_file)), 
                                                   source, mix_id + '.wav')
                    assert os.path.isfile(speech_ref_file)
                    
                    speech_ref_list.append(sf.read(speech_ref_file)[0])
                
                speech_ref = np.zeros_like(speech_ref_list[0])
                for s in speech_ref_list:
                    speech_ref += s    
                
                si_sdr = sisdr_metric.compute_sisdr(speech_est, speech_ref)
                
                row = [subset, os.path.basename(input_file), 
                       os.path.basename(output_file), si_sdr]
                
                librimix_df.loc[len(librimix_df)] = row
                
                if compute_input_scores:
                    
                    mix, sr = sf.read(input_file)
                    
                    si_sdr = sisdr_metric.compute_sisdr(mix, speech_ref)
                    
                    row = [subset, os.path.basename(input_file), 
                           os.path.basename(input_file), si_sdr]
                    
                    unprocessed_librimix_df.loc[len(unprocessed_librimix_df)] = row
                    
        csv_file = os.path.join(librimix_output_path, 'results.csv')
        librimix_df.to_csv(csv_file)
    
        if compute_input_scores:        
            csv_file = os.path.join(librimix_output_path, 'results_unprocessed.csv')
            unprocessed_librimix_df.to_csv(csv_file)
            
def summarize_results():
    
    models = ['sudo-rm-rf', 'remixit', 'remixit-vad']
    datasets = ['CHiME-5', 'reverberant-LibriCHiME-5', 'LibriMix']

    df_chime5_all = pd.DataFrame(columns=['model', 
                                          'SIG_MOS', 
                                          'BAK_MOS', 
                                          'OVR_MOS'])

    df_reverberant_librichime5_all = pd.DataFrame(columns=['model','SI-SDR'])
    df_reverberant_librimix_all = pd.DataFrame(columns=['model','SI-SDR'])

    df_all_list = [df_chime5_all, df_reverberant_librichime5_all, 
                   df_reverberant_librimix_all]

    for dataset, df_all in zip(datasets, df_all_list):
        
        for model in models:
            
            results_path = os.path.join(output_path, model, dataset)
            
            # unprocessed (input) results
            if os.path.isfile(os.path.join(results_path, 'results_unprocessed.csv')):
                
                df_unprocessed = pd.read_csv(os.path.join(results_path, 
                                                          'results_unprocessed.csv'))
                
                if dataset == 'CHiME-5':
                    row = ['unprocessed', np.mean(df_unprocessed['SIG_MOS']), 
                           np.mean(df_unprocessed['BAK_MOS']), 
                           np.mean(df_unprocessed['OVR_MOS'])]
                else:
                    row = ['unprocessed', np.mean(df_unprocessed['SI-SDR'])]
                    
                df_all.loc[len(df_all)] = row
            
            
            # model results
            df = pd.read_csv(os.path.join(results_path, 'results.csv'))
            if dataset == 'CHiME-5':
                row = [model, np.mean(df['SIG_MOS']), np.mean(df['BAK_MOS']), 
                       np.mean(df['OVR_MOS'])]
            else:
                row = [model, np.mean(df['SI-SDR'])]
            
            df_all.loc[len(df_all)] = row

        print(dataset)
        print(df_all)
        print('\n')
        
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-baseline", help="run the baseline",
                        action="store_true")
    
    parser.add_argument("--model", help="model to run/evaluate",
                        choices=['sudo-rm-rf', 'remixit', 'remixit-vad'])
    
    parser.add_argument("--eval-baseline", help="compute the baseline output scores",
                        action="store_true")
    
    parser.add_argument("--input-scores", help="additionally compute the performance metrics on the unprocessed noisy speech signals",
                        action="store_true")
        
    args = parser.parse_args()
    
    model = args.model
    
    output_path = os.path.join(output_path, model)
    
    if model == 'sudo-rm-rf':
        checkpoint = 'pretrained_checkpoints/libri1to3mix_supervised_teacher_w_mixconsist.pt'
    elif model == 'remixit':
        checkpoint = 'pretrained_checkpoints/remixit_chime_adapted_student.pt'
    elif model == 'remixit-vad':
        checkpoint = 'pretrained_checkpoints/remixit_chime_adapted_student_using_vad.pt'
    else:
        raise ValueError('Unknown model')
    
    # run baseline
    if args.run_baseline:
        run_baseline(checkpoint, output_path,
                     chime5_input_path, 
                     chime5_subsets, 
                     reverberant_librichime5_input_path, 
                     reverberant_librichime5_subsets,
                     librimix_input_path,
                     librimix_subsets)
    
    # compute scores
    if args.eval_baseline and args.input_scores:
        compute_metrics(results_path=output_path,
                        chime5_input_path=chime5_input_path, 
                        chime5_subsets=chime5_subsets, 
                        reverberant_librichime5_input_path=reverberant_librichime5_input_path, 
                        reverberant_librichime5_subsets=reverberant_librichime5_subsets,
                        librimix_input_path=librimix_input_path,
                        librimix_subsets=librimix_subsets, 
                        compute_input_scores=True)
    elif args.eval_baseline:
        compute_metrics(results_path=output_path,
                        chime5_input_path=chime5_input_path, 
                        chime5_subsets=chime5_subsets, 
                        reverberant_librichime5_input_path=reverberant_librichime5_input_path, 
                        reverberant_librichime5_subsets=reverberant_librichime5_subsets,
                        librimix_input_path=librimix_input_path,
                        librimix_subsets=librimix_subsets, 
                        compute_input_scores=False)
        
        
    