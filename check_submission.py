#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import glob
import os
import warnings
import numpy as np
import soundfile as sf
import pandas as pd
import pyloudnorm as pyln
import argparse

def check_wav_filename(filename):
    
    filename_split = filename.split('_')
    if len(filename_split) != 4:
        raise ValueError('''Incorrect name for the wav files, 
                         please check the namming conventions.''')
    if filename_split[0][0] != 'S' or filename_split[1][0] != 'P' or filename_split[3] != 'output.wav':
        raise ValueError('''Incorrect name for the wav files, 
                         please check the namming conventions.''')
                         
def check_wav_file(wav_file):
    
    sf_obj = sf.SoundFile(wav_file, 'r+')
    
    if sf_obj.samplerate != sr:
        raise ValueError('Sampling rate should be 16 kHz.')
        
    if sf_obj.subtype != 'PCM_16':
        raise ValueError('Signals should be submitted as 16-bit PCM WAV files.')

def check_loudness(wav_file):
    
    x, _ = sf.read(wav_file)
    loudness = meter.integrated_loudness(x)
    
    return np.isclose(loudness, -30.0, rtol=0.5)

if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Validation of the submission to the UDASE task of CHiME-7", 
                                     add_help=False)
    parser.add_argument("submission_dir", type=str, help="Path to the submission directory.")
    args = parser.parse_args()
    
    submission_dir = args.submission_dir

    # check folder structure
    
    folders_to_check = ['audio',
                        'csv',
                        'audio/reverberant-LibriCHiME-5',
                        'audio/CHiME-5',
                        'audio/reverberant-LibriCHiME-5/eval',
                        'audio/reverberant-LibriCHiME-5/eval/1',
                        'audio/reverberant-LibriCHiME-5/eval/2',
                        'audio/reverberant-LibriCHiME-5/eval/3',
                        'audio/CHiME-5/eval',
                        'audio/CHiME-5/eval/1',
                        'audio/CHiME-5/eval/listening_test',
                        'csv/reverberant-LibriCHiME-5',
                        'csv/CHiME-5']
    
    print("Verifying the structure of the submission folder...")
    
    for folder in folders_to_check:
        if not os.path.isdir(os.path.join(submission_dir, folder)):
            raise ValueError('Missing \'' + folder + 
                             '\' folder in the submission directory')
    
    print('...done')
    
    # check audio files
    
    sr = 16000
    meter = pyln.Meter(sr) # create loudness meter
    
    subsets = ['audio/reverberant-LibriCHiME-5/eval/1',
               'audio/reverberant-LibriCHiME-5/eval/2',
               'audio/reverberant-LibriCHiME-5/eval/3',
               'audio/CHiME-5/eval/1',
               'audio/CHiME-5/eval/listening_test']
    
    num_files_list = [1394, 494, 64, 3013, 241]
    
    print("Verifying the audio files...")
    
    for subset, num_files in zip(subsets, num_files_list):
        
        wav_file_list = glob.glob(os.path.join(submission_dir, subset, '*.wav'))
        
        if len(wav_file_list) != num_files:
            raise ValueError('Invalid number of wav files in \'' + subset + '\'')
        
        for wav_file in wav_file_list:
            
            check_wav_filename(os.path.basename(wav_file))
            
            check_wav_file(wav_file)
                    
            if (subset=='audio/CHiME-5/eval/1' or subset=='audio/CHiME-5/eval/listening_test'):
                            
                if not check_loudness(wav_file):
                    raise ValueError('Loudness is not -30 LUFS for file ' + wav_file)
    
    print('...done')
    
    # check csv files
    
    print("Verifying the csv files...")
    
    csv_file = os.path.join(submission_dir, 'csv', 'CHiME-5', 'results.csv')
    df_chime5 = pd.read_csv(csv_file)
    
    columns = ['subset', 'input_file_name', 'output_file_name', 'SIG_MOS', 'BAK_MOS', 'OVR_MOS']
    
    if len(df_chime5) != 3013:
        raise ValueError('Invalid number of rows in ' + csv_file)
    
    for column in columns:
        
        if not column in df_chime5.columns:
            raise ValueError(csv_file + ' should contain a column entitled ' + column)
    
    if not all(df_chime5['subset'] == 'eval/1'):
        raise ValueError('Column \'subset\' of ' + csv_file + ' should only contain \'eval/1\'')
        
    
    csv_file = os.path.join(submission_dir, 'csv', 'reverberant-LibriCHiME-5', 'results.csv')
    df_librichime5 = pd.read_csv(csv_file)
    
    columns = ['subset', 'input_file_name', 'output_file_name', 'SI-SDR']
    
    if len(df_librichime5) != 1952:
        raise ValueError('Invalid number of rows in ' + csv_file)
    
    for column in columns:
        
        if not column in df_librichime5.columns:
            raise ValueError(csv_file + ' should contain a column entitled ' + column)
    
    if not all(df_librichime5['subset'].unique() == ['eval/1', 'eval/2', 'eval/3']):
        raise ValueError('Column \'subset\' of ' + csv_file + ' should only contain \'eval/1\', \'eval/2\', or \'eval/3\'')
        
    print('...done')



    