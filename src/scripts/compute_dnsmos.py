#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
from pathlib import Path
import librosa
from dnsmos_metric import compute_dnsmos


def parse_args():
    parser=argparse.ArgumentParser(description=''' 
        DNS-MOS metric is computed for each audio segment in data_dir.
        ''')
    parser.add_argument("data_dir", help="Audio data directory.", type=Path)
    parser.add_argument("result_file", help="Result CSV file.", type=Path)
    
    args=parser.parse_args()
    return args

def main():
    # load arguments
    args = parse_args()
    print(args)
    data_dir = args.data_dir
    if not data_dir.exists():
        raise ValueError("data_dir doesn't exist.")
    if not os.path.exists(os.path.dirname(args.result_file)):
        os.makedirs(os.path.dirname(args.result_file))
    if os.path.exists(args.result_file):
        metrics = pd.read_csv(args.result_file, index_col=0)
    else:
        metrics = pd.DataFrame(columns=['sig_mos', 'bak_mos', 'ovr_mos'])
        

    # search audio files
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_list.append(os.path.join(root, file))
    
    # extract audio segments
    for count, file_path in enumerate(file_list):
        print(f'({count+1}/{len(file_list)}) {file_path}')
        if file_path in metrics.index:
            print('---- skipped')
        else:
            audio, _ = librosa.load(file_path)
            m = compute_dnsmos(audio)
            for col in metrics.columns:
                metrics.loc[file_path, col] = m[col]
            metrics.to_csv(args.result_file)
    print(f'\nMetrics saved in {args.result_file}')
                            
if __name__ == '__main__':
    main()    

