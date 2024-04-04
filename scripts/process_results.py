import numpy as np
import pickle as pkl
from pathlib import Path
from collections import defaultdict

from typing import List, Dict
from numpy.typing import NDArray


DATA_ROOT: Path =  Path('/home/timodw/IDLab/time_series_preprocessing') 
RESULTS_PATH: Path = DATA_ROOT / 'results'

PREPROCESSING_METHODS: List[str] = ['raw', 'standardized', 'normalized', 'peak_segmented', 'smoothed', 'resampled_25', 'resampled_50', 'pca', 'dft_filter', 'dft']
LEARNING_METHODS: List[str] = ['cnn', 'kmeans', 'gmm', 'dec_km', 'dec_gmm']

PREPROCESSING_STRING_MAPPING: Dict[str, str] = {
   'raw': 'Raw',
   'standardized': 'Standardized',
   'normalized': 'Normalized',
   'peak_segmented': 'Peak segmented',
   'smoothed': 'Moving average smoothing',
   'resampled_25': 'Resampled (25 Hz)',
   'resampled_50': 'Resampled (50 Hz)',
   'pca': 'PCA',
   'dft_filter': 'DFT filter',
   'dft': 'DFT' 
}
LEARNING_STRING_MAPPING: Dict[str, str] = {
   'kmeans': 'K-Means',
   'gmm': 'GMM',
   'cnn': 'CNN',
   'dec_km': 'DEC (K-Means)',
   'dec_gmm': 'DEC (GMM)'
}

def process_report_dictionaries(reports: List[Dict[str, Dict[str, float]]]) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = defaultdict(list)
    for report in reports:
        precision, recall, f1_score = [], [], []
        for i in range(4):
            i = str(i)
            report_per_class = report[i]
            precision.append(report_per_class['precision'])
            recall.append(report_per_class['recall'])
            f1_score.append(report_per_class['f1-score'])
        results['precision'].append(np.mean(precision))
        results['recall'].append(np.mean(recall))
        results['f1-score'].append(np.mean(f1_score))
    return results

if __name__ == '__main__':
    print("\\multicolumn{2}{c}{} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\")
    print("\\midrule")

    for learning_method in LEARNING_METHODS:
        print(f"\\multirow{{10}}{{*}}{{\\textbf{{{LEARNING_STRING_MAPPING[learning_method]}}}}}", end=' ')
        for preprocessing_method in PREPROCESSING_METHODS:
            results_folder: Path = RESULTS_PATH / f"{learning_method}_{preprocessing_method}"
            train_acc: List[float] = pkl.load(open(results_folder / 'train_acc.pkl', 'rb'))
            val_acc: List[float] = pkl.load(open(results_folder / 'val_acc.pkl', 'rb'))
            train_conf_mat: List[NDArray] = pkl.load(open(results_folder / 'train_conf_mat.pkl', 'rb'))
            train_conf_mat: NDArray = np.stack(train_conf_mat, axis=0)
            val_conf_mat: List[NDArray] = pkl.load(open(results_folder / 'val_conf_mat.pkl', 'rb'))
            val_conf_mat: NDArray = np.stack(val_conf_mat, axis=0)
            train_class_report: List[Dict[str, Dict[str, float]]] = pkl.load(open(results_folder / 'train_class_report.pkl', 'rb'))
            train_class_report: Dict[str, List[float]] = process_report_dictionaries(train_class_report)
            val_class_report: List[Dict[str, Dict[str, float]]] = pkl.load(open(results_folder / 'val_class_report.pkl', 'rb'))
            val_class_report: Dict[str, List[float]] = process_report_dictionaries(val_class_report)

            print(f"& \\textbf{{{PREPROCESSING_STRING_MAPPING[preprocessing_method]}}}", end=' ')
            print(f"& ${100 * np.mean(val_acc):.2f}\\pm{100 * np.std(val_acc):.2f}\\%$", end=' ')
            print(f"& ${100 * np.mean(val_class_report['precision']):.2f}\\pm{100 * np.std(val_class_report['precision']):.2f}\\%$", end=' ')
            print(f"& ${100 * np.mean(val_class_report['recall']):.2f}\\pm{100 * np.std(val_class_report['recall']):.2f}\\%$", end=' ')
            print(f"& ${100 * np.mean(val_class_report['f1-score']):.2f}\\pm{100 * np.std(val_class_report['f1-score']):.2f}\\%$ \\\\")
        print('\\midrule')




