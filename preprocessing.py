import sys
import pandas as pd
import numpy as np

leads = ['phys-raw-lead2-HRV']

W_PATH = 'datasets/corrected'

for name in leads:
	try:
		df = pd.read_feather(f'datasets/{name}')
	except:
		print(f'{name} was not found in datasets/')
		exit()

	df.loc[df.label == 'AF\n', 'label'] = 'AF'
	df.loc[df.label == 'I-AVB\n', 'label'] = 'I-AVB'
	df.loc[df.label == 'LBBB\n', 'label'] = 'LBBB'
	df.loc[df.label == 'Normal\n', 'label'] = 'Normal'
	df.loc[df.label == 'PAC\n', 'label'] = 'PAC'
	df.loc[df.label == 'PVC\n', 'label'] = 'PVC'
	df.loc[df.label == 'RBBB\n', 'label'] = 'RBBB'
	df.loc[df.label == 'STD\n', 'label'] = 'STD'
	df.loc[df.label == 'STE\n', 'label'] = 'STE'

	df.to_feather(f'{W_PATH}/{name}-corrected')