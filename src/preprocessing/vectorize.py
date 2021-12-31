import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def vectorize(data_3d):
	first24 = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
	'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM',
	'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
	'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE']
	header = []
	data_min = np.min(data_3d, axis=1)

	header.extend([x+'_min' for x in first24])
	data_max = np.max(data_3d, axis=1)
	
	header.extend([x+'_max' for x in first24])
	data_median = np.median(data_3d, axis=1)
	
	header.extend([x+'_median' for x in first24])
	data_sd = np.std(data_3d, axis=1)
	header.extend([x+'_sd' for x in first24])
	data_last = data_3d[:,-1,:]
	header.extend([x+'_last' for x in first24])
	data_skew = skew(data_3d, axis=1)
	header.extend([x+'_skew' for x in first24])
	data_kurtosis = kurtosis(data_3d, axis=1)
	header.extend([x+'_kurtosis' for x in first24])
	combined_data = np.hstack((data_min, data_max, data_median, data_sd, data_last, data_skew, data_kurtosis))
	combined_df = pd.DataFrame(combined_data, columns=header)
	return combined_df
