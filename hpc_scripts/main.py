import argparse
from pathlib import Path
from data import get_horse_data

from typing import Dict, Union

DATA_PATH = Path('data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('preprocessing')

    args = parser.parse_args()
    learning_method: str = args.method
    preprocessing_step: str = args.preprocessing

    experiment_id = f"{learning_method}_{preprocessing_step}"
    preprocessing_methods: Dict[str, Union[bool, int]] = dict()
    if preprocessing_step == 'standardized':
        preprocessing_methods['standardized'] = True
    elif preprocessing_step == 'normalized':
        preprocessing_methods['normalized'] = True
    elif preprocessing_step == 'peak_segmented':
        preprocessing_methods['peak_segmented'] = True
    elif preprocessing_step == 'smoothed':
        preprocessing_methods['smoothed'] = True
    elif preprocessing_step.startswith('resampled'):
        _, hz = preprocessing_step.split('_')
        hz = int(hz)
        preprocessing_methods['resampled'] = True
        preprocessing_methods['resampled_sr'] = hz
    elif preprocessing_step == 'pca':
        preprocessing_methods['pca'] = True
        preprocessing_step['pca_components'] = 25
    elif preprocessing_step == 'dft_filter':
        preprocessing_methods['dft_filtered'] = True
        preprocessing_methods['dft_filtered_hz'] = 10
    elif preprocessing_step == 'dft':
        preprocessing_methods['dft'] = True


    data_per_horse = get_horse_data(DATA_PATH, **preprocessing_methods)
    print(experiment_id)