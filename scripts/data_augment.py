import os
import pandas as pd
import filter

origin_path = '../data/ctrl_test'
destination_path = '../data/processed_ctrl'

for fname in os.listdir(origin_path):
    if fname.endswith('.csv'):
        print(fname)
        data = filter.filter(f'{origin_path}/{fname}')
        data.sort_values(by=['time', 'part']).to_csv(f'{destination_path}/{fname}', sep=';', header=False, index=False, float_format='%.3f')
