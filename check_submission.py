#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import glob
import os
import warnings

submission_dir = '/data/datasets/UDASE-CHiME2023/remixit-vad-submission-example'

#%% check structure

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

for folder in folders_to_check:
    if not os.path.isdir(os.path.join(submission_dir, folder)):
        raise ValueError('Missing \'' + folder + 
                         '\' folder in the submission directory')


#%% check audio

def check_wav_filename(filename):
    
    filename_split = filename.split('_')
    if len(filename_split) != 4:
        raise ValueError('''Incorrect name for the wav files, 
                         please check the namming conventions.''')
    if filename_split[0][0] != 'S' or filename_split[1][0] != 'P' or filename_split[3] != 'output.wav':
        raise ValueError('''Incorrect name for the wav files, 
                         please check the namming conventions.''')
    

subsets = ['audio/reverberant-LibriCHiME-5/eval/1',
           'audio/reverberant-LibriCHiME-5/eval/2',
           'audio/reverberant-LibriCHiME-5/eval/3',
           'audio/CHiME-5/eval/1',
           'audio/CHiME-5/eval/listening_test']

num_files_list = [1394, 494, 64, 3013, 241]

for subset, num_files in zip(subsets, num_files_list):
    wav_file_list = glob.glob(os.path.join(submission_dir, subset, '*.wav'))
    if len(wav_file_list) != num_files:
        raise ValueError('Invalid number of wav files in \'' + subset + '\'')
    for wav_file in wav_file_list:
        check_wav_filename(os.path.basename(wav_file))


#%% check csv