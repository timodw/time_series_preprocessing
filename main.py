import pandas as pd
from pathlib import Path
from collections import defaultdict
from hashlib import sha1
import json
from base64 import b64encode

import numpy as np
from numpy.fft import fft, fftfreq, ifft

from typing import Optional, List, Tuple, Dict
from numpy.typing import NDArray


MOVEMENTS = ('standing', 'walking', 'trotting', 'galloping')
LABEL_MAPPING = {l: i for i, l in enumerate(MOVEMENTS)}
DATA_ROOT = Path('datasets/HorsingAround/csv')
SAMPLING_RATE = 200

WINDOW_SIZE = 2 * SAMPLING_RATE
STEP_SIZE = WINDOW_SIZE // 2
MA_WINDOW = None
STANDARDIZE = False
NORMALIZE = False
FFT_FILTER_CUTOFF = None
RESAMPLING = None
FFT_WINDOWS = False
WINDOW_NORMALIZATION = False


def moving_average_smoothing(X, window_size=20):
    X = X.copy()
    return np.convolve(X, np.ones(window_size), 'valid') / window_size


def standardize_time_series(X):
    x_mean, x_std = X.mean(), X.std()
    X = X.copy()
    X -= x_mean
    X /= x_std
    return X


def normalize_time_series(X, symmetric=False):
    x_min, x_max = X.min(), X.max()
    X = X.copy()
    X -= x_min
    X /= (x_max - x_min)
    if symmetric:
        X *= 2
        X -= 1
    return X

def normalize_windows(X, symmetric=False):
    min_val, max_val = X.min(), X.max()
    X -= min_val
    X /= (max_val - min_val)
    if symmetric:
        X *= 2
        X -= 1
    return X


def fft_based_filter(X, high_cutoff, sampling_rate):
    frequencies = fftfreq(len(X), 1 / sampling_rate)
    mask = np.abs(frequencies) > high_cutoff
    
    X_fft = fft(X)
    X_fft[mask] = 0.
    X = np.abs(ifft(X_fft))
    
    return X


def sliding_window(X, y, window_size, step_size):
    X_windows = []
    y_windows = []

    for start_i in range(0, len(X) - window_size, step_size):
        y_unique = np.unique(y[start_i:start_i + window_size])
        if len(y_unique) == 1:
            X_windows.append(X[start_i:start_i + window_size])
            y_windows.append(y_unique[0].split('-')[0])

    X_windows = np.stack(X_windows)
    y_windows = np.asarray(y_windows)
    
    return X_windows, y_windows


def resample_time_series(X, new_sampling_rate, original_sampling_rate):
    ratio = new_sampling_rate / original_sampling_rate
    original_size = len(X)
    new_size = int(original_size * ratio)
    x_coord_original = np.arange(0, original_size)
    x_coord_new = np.linspace(0, original_size, new_size)
    X_resampled = np.interp(x_coord_new, x_coord_original, X)
    return X_resampled


def fft_time_series(X):
    X_fft = fft(X)
    X_fft[0] = 0.
    return np.abs(X_fft)[:len(X) // 2]


def get_activity_distribution(data_root: Path, movements: List[str]) -> pd.DataFrame:
    activity_distribution = pd.read_csv(data_root / 'activity_distribution.csv')
    columns_of_interest = []
    for movement in movements:
        for column in activity_distribution.columns:
            if column.startswith(movement):
                columns_of_interest.append(column)
    activity_distribution = activity_distribution[['Row'] + columns_of_interest]
    return activity_distribution


def get_horses_of_interest(data_root: Path, movements=MOVEMENTS) -> List[str]:
    activity_distribution = get_activity_distribution(data_root, movements)
    horses_of_interest = []
    for i, row in activity_distribution.iterrows():
        if row['Row'] != 'total':
            movement_counts = defaultdict(float)
            for movement in activity_distribution.columns[1:]:
                movement_type = movement.split('_')[0]
                count = row[movement]
                if count == count:
                    movement_counts[movement_type] += row[movement]
            valid = True
            for movement in movements:
                if movement_counts[movement] < 1.:
                    valid = False
                    break
            if valid:
                horses_of_interest.append(row['Row'])
    return horses_of_interest


def get_horse_dataframes(data_root: Path, horse_name: str) -> List[pd.DataFrame]:
    dataframes = []
    for f in data_root.glob(f"*{horse_name}*"):
        df = pd.read_csv(f, low_memory=False)
        df = df[['label', 'segment', 'Ax', 'Ay', 'Az']].dropna()
        df['norm'] = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
        if len(df) > 0:
            dataframes.append(df)
    return dataframes


def process_horse_data(dataframes: List[pd.DataFrame], movements: List[str],
                       label_mapping: Dict[str, int],
                       ma_window: Optional[int], standardize: bool,
                       normalize: bool, fft_filter_cutoff: Optional[int],
                       window_size: int, step_size: int,
                       resampling: Optional[int], fft_windows: bool,
                       window_normalization: bool) -> Tuple[NDArray, NDArray]:
    X_windows, y_windows = [], []
    for df in dataframes:
        X, y = df[['Ax', 'Ay', 'Az']].values, df.label.values
        if ma_window:
            X = moving_average_smoothing(X, window_size=ma_window)
        if standardize:
            X = standardize_time_series(X)
        if normalize:
            X = normalize_time_series(X)
        if fft_filter_cutoff:
            X = fft_based_filter(X, fft_filter_cutoff, sampling_rate=SAMPLING_RATE)

        X, y = sliding_window(X, y, window_size, step_size)
        selected_windows = np.isin(y, movements)
        X = X[selected_windows]
        y = y[selected_windows]
        if len(X) == 0:
            continue
        y = np.vectorize(label_mapping.get)(y)
        if resampling:
            X_resampled = []
            for window in X:
                X_resampled.append(resample_time_series(window, resampling, SAMPLING_RATE))
            X = np.stack(X_resampled)
        if fft_windows:
            X_fft = []
            for window in X:
                X_fft.append(fft_time_series(window))
            X = np.stack(X_fft)
        if window_normalization:
            X = normalize_windows(X)

        X_windows.append(X)
        y_windows.append(y)
    return np.concatenate(X_windows), np.concatenate(y_windows)


if __name__ == '__main__':
    config_dict = {
        'ma_window': MA_WINDOW,
        'standardize': STANDARDIZE,
        'normalize': NORMALIZE,
        'fft_filter_cutoff': FFT_FILTER_CUTOFF,
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'resampling': RESAMPLING,
        'fft_windows': FFT_WINDOWS,
        'window_normalization': WINDOW_NORMALIZATION
    }
    config_json = json.dumps(config_dict, sort_keys=True)
    print(b64encode(sha1(repr(config_json).encode()).digest()).decode())
    # horses = get_horses_of_interest(DATA_ROOT, MOVEMENTS)
    # horses_dataframes = defaultdict(list)
    # for horse in horses:
    #     print(horse)
    #     dataframes = get_horse_dataframes(DATA_ROOT, horse)
    #     X, y = process_horse_data(dataframes, movements=MOVEMENTS,
    #                               label_mapping=LABEL_MAPPING, ma_window=None,
    #                               standardize=False, normalize=False, fft_filter_cutoff=None,
    #                               window_size=WINDOW_SIZE, step_size=WINDOW_SIZE // 2,
    #                               resampling=False, fft_windows=False, window_normalization=False)
    #     print(X.shape, y.shape)
    # print('DONE')
