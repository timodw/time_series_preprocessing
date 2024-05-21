#!/bin/bash

methods=('kmeans' 'gmm' 'cnn' 'dec_km' 'dec_gmm')
preprocessing_methods=('raw' 'standardized' 'normalized' 'peak_segmented' 'smoothed' 'resampled_25' 'resampled_50' 'pca' 'dft_filter' 'dft')

for method in "${methods[@]}"; do
    for preprocessing in "${preprocessing_methods[@]}"; do
        echo "${method} ${preprocessing}"
        python /home/timodw/IDLab/time_series_preprocessing/hpc_scripts/main.py ${method} ${preprocessing}
    done
done