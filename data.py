import pandas as pd
from pathlib import Path
from collections import defaultdict
from hashlib import sha1
import json
from base64 import b32encode
from time import time
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from itertools import combinations
import numpy as np
from numpy.fft import fft, fftfreq, ifft

from typing import Optional, List, Tuple, Dict
from numpy.typing import NDArray


MOVEMENTS = ('standing', 'walking', 'trotting', 'galloping')
LABEL_MAPPING = {l: i for i, l in enumerate(MOVEMENTS)}
DATA_ROOT = Path('datasets/HorsingAround/data/csv')
PROCESSED_DATA_ROOT = Path('processed_data')
SAMPLING_RATE = 100

USE_NORM = True
WINDOW_SIZE = 2 * SAMPLING_RATE
STEP_SIZE = WINDOW_SIZE // 2
MA_WINDOW = None
STANDARDIZE = False
NORMALIZE = False
FFT_FILTER_CUTOFF = None
RESAMPLING = None
FFT_WINDOWS = True
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
    X_fft = fft(X, axis=0)
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
                       label_mapping: Dict[str, int], use_norm: bool,
                       ma_window: Optional[int], standardize: bool,
                       normalize: bool, fft_filter_cutoff: Optional[int],
                       window_size: int, step_size: int,
                       resampling: Optional[int], fft_windows: bool,
                       window_normalization: bool) -> Tuple[NDArray, NDArray]:
    X_windows, y_windows = [], []
    for df in dataframes:
        if use_norm:
            X = df[['norm']].values
        else:
            X = df[['Ax', 'Ay', 'Az']].values
        y = df.label.values

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


def load_dataset(root_folder: Path, dataset_id: str) -> Tuple[List[NDArray], List[NDArray]]:
    X_list = []
    y_list = []

    folder_path = root_folder / dataset_id
    for X_path in folder_path.glob('X_*.npy'):
        horse_name = X_path.stem[2:]
        y_path = folder_path / f'y_{horse_name}.npy'
        X = np.load(X_path)
        y = np.load(y_path)
        X_list.append(X)
        y_list.append(y)

    return X_list, y_list


def get_stratified_split(y_list: List[NDArray]) -> Tuple[List[int], List[int]]:
    unique_movements = [i for i, _ in enumerate(MOVEMENTS)]
    min_diff = float('inf')
    best_split = None

    def calculate_ratio(data):
        ratio = {movement: (data == movement).sum() / len(data) for movement in unique_movements}
        return ratio

    for training_ids in combinations(range(len(y_list)), 5):
        y_train = np.concatenate([y_list[i] for i in training_ids])
        y_val = np.concatenate([y_list[i] for i in range(len(y_list)) if i not in training_ids])
        
        train_ratio = calculate_ratio(y_train)
        val_ratio = calculate_ratio(y_val)
        
        diff = sum(abs(train_ratio.get(activity, 0) - val_ratio.get(activity, 0)) for activity in unique_movements)
        
        if diff < min_diff:
            min_diff = diff
            best_split = (training_ids, tuple(i for i in range(len(y_list)) if i not in training_ids))
    return best_split


def get_training_and_validation_data(data_root: Path, dataset_id: str, balanced=False) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    X, y = load_dataset(data_root, dataset_id)
    train_indices, val_indices = get_stratified_split(y)
    X_train = np.concatenate([X[i] for i in train_indices])[:, :, 0]
    y_train = np.concatenate([y[i] for i in train_indices])
    if balanced:
        sampler = RandomUnderSampler()
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    X_val = np.concatenate([X[i] for i in val_indices])[:, :, 0]
    y_val = np.concatenate([y[i] for i in val_indices])

    return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    config_dict = {
        'use_norm': USE_NORM,
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
    config_json = json.dumps(config_dict, sort_keys=True, indent=4)
    config_hash = b32encode(sha1(repr(config_json).encode()).digest()).decode()[:10]
    output_folder = PROCESSED_DATA_ROOT / config_hash
    output_folder.mkdir(exist_ok=True, parents=True)
    (output_folder / 'config.json').write_text(config_json + '\n')

    print(config_json)

    horses = get_horses_of_interest(DATA_ROOT, MOVEMENTS)
    horses_dataframes = defaultdict(list)
    for horse in horses:
        print(horse, end='', flush=True)
        t0 = time()
        dataframes = get_horse_dataframes(DATA_ROOT, horse)
        X, y = process_horse_data(dataframes, movements=MOVEMENTS, label_mapping=LABEL_MAPPING,
                                  use_norm=USE_NORM, ma_window=MA_WINDOW,
                                  standardize=STANDARDIZE, normalize=NORMALIZE, fft_filter_cutoff=FFT_FILTER_CUTOFF,
                                  window_size=WINDOW_SIZE, step_size=WINDOW_SIZE // 2,
                                  resampling=RESAMPLING, fft_windows=FFT_WINDOWS, window_normalization=WINDOW_NORMALIZATION)
        np.save(output_folder / f"X_{horse.lower()}.npy", X)
        np.save(output_folder / f"y_{horse.lower()}.npy", y)
        print(f"; {y.shape[0]} windows; {time() - t0:.02f}s")
