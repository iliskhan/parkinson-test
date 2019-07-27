import os

import numpy as np 
import pandas as pd 

origin_path = '../../movement/паркинсон тест'
destination_path = '../data/prksn_test'

names = ['times', 'parts', 'x', 'y', 'z']


for name in os.listdir(origin_path):
	print(name)
	temp_path = os.path.join(origin_path, name)
	file = pd.read_csv(temp_path,';', names=names)

	start_new_frame = 0
	for i in range(1, len(file)):
		if file.iloc[i].times < file.iloc[i-1].times:
			temp_frame = file.iloc[start_new_frame:i]
			temp_frame.to_csv(f'{destination_path}/{name[:-4]}_{i}.csv', sep=';', header=False, index=False, float_format='%.3f')
			start_new_frame = i

	file.iloc[start_new_frame:].to_csv(f'{destination_path}/{name[:-4]}_{i}.csv', sep=';', header=False, index=False, float_format='%.3f')