import argparse
from pathlib import Path
from metrics import acc
from cnn import CNN
from dec import DenseEncoder,  DenseDecoder, pretraining, train_dec
from time import time
import pickle as pkl

import numpy as np
from numpy.typing import NDArray

import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import skorch

import torch

from typing import Dict, Union

DATA_ROOT =  Path('/home/timodw/IDLab/time_series_preprocessing') 
DATA_PATH = DATA_ROOT / 'processed_data'
RESULTS_PATH = DATA_ROOT / 'results'
N_FOLDS = 4
N_ITER = 5

def evaluation(X_train, y_train, X_val, y_val, n_classes, method='kmeans'):
    supervised = False
    if method == 'kmeans':
        model = KMeans(n_clusters=n_classes, algorithm='elkan')
        model.fit(X_train)
        model = model.predict
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_classes)
        model.fit(X_train)
        model = model.predict
    elif method == 'cnn':
        X_train = np.expand_dims(X_train, 1).astype(np.float32)
        X_val = np.expand_dims(X_val, 1).astype(np.float32)
        net = skorch.classifier.NeuralNetClassifier(CNN, criterion=torch.nn.CrossEntropyLoss,
                                                    optimizer=torch.optim.Adam, optimizer__weight_decay=1E-2,
                                                    max_epochs=400, batch_size=256,
                                                    iterator_train__shuffle=True, module__input_size=X_train.shape[-1],
                                                    device='cuda', train_split=None, verbose=False)
        net.fit(X_train, y_train)
        model = net.predict
        supervised = True
    elif method.startswith('dec'):
        dense_encoder = DenseEncoder(X_train.shape[-1], 10, layer_sizes=[500, 500])
        dense_decoder = DenseDecoder(10, X_train.shape[-1], layer_sizes=[500, 500])
        pretraining(dense_encoder, dense_decoder,
                    X_train, X_val,
                    epochs=200, lr=1E-3, batch_size=512,
                    device='cuda', verbose=False)
        if method.endswith('km'):
            clustering_method = 'kmeans'
        else:
            clustering_method = 'gmm'
        model = train_dec(dense_encoder, X_train, X_val, n_clusters=len(np.unique(y_train)),
                          device='cuda', latent_dim=10, lr=1E-2, cluster_init=clustering_method, tol=1E-3, verbose=False)
        model.eval()
        
    
    if method.startswith('dec'):
        X_train = torch.from_numpy(X_train).to('cuda').to(torch.float32)
        X_val = torch.from_numpy(X_val).to('cuda').to(torch.float32)
        train_output = model(X_train)
        val_output = model(X_val)
        y_train_pred = np.argmax(train_output.detach().cpu().numpy(), axis=-1)
        y_val_pred = np.argmax(val_output.detach().cpu().numpy(), axis=-1)
    else:
        y_train_pred = model(X_train)
        y_val_pred = model(X_val)
    
    if not supervised:
        _, mapping = acc(y_train, y_train_pred, return_mapping=True)

        y_train_pred = [mapping[l] for l in y_train_pred]
        y_val_pred = [mapping[l] for l in y_val_pred]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_conf_mat = confusion_matrix(y_train, y_train_pred)
    train_class_report = classification_report(y_train, y_train_pred,
                                               output_dict=True, zero_division=0.)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_conf_mat = confusion_matrix(y_val, y_val_pred)
    val_class_report = classification_report(y_val, y_val_pred,
                                             output_dict=True, zero_division=0.)
    
    return train_acc, val_acc, train_conf_mat, val_conf_mat, train_class_report, val_class_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('preprocessing')

    args = parser.parse_args()
    learning_method: str = args.method
    preprocessing_step: str = args.preprocessing

    experiment_id: str = f"{learning_method}_{preprocessing_step}"
    results_folder = RESULTS_PATH / experiment_id
    results_folder.mkdir(exist_ok=True, parents=True)
    dataset_path: Path = DATA_PATH / preprocessing_step

    train_acc = []
    val_acc = []
    train_conf_mat = []
    val_conf_mat = []
    train_class_report = []
    val_class_report = []
    for i in range(N_FOLDS):
        fold_path: Path = dataset_path / f"fold_{i}"
        X_train: NDArray = np.load(fold_path / 'X_train.npy')
        y_train: NDArray = np.load(fold_path / 'y_train.npy')
        X_val: NDArray = np.load(fold_path / 'X_val.npy')
        y_val: NDArray = np.load(fold_path / 'y_val.npy')
        n_classes = len(np.unique(y_train))

        for j in range(N_ITER):
            print(f"Fold {i+1}/{N_FOLDS}; Iteration {j + 1}/{N_ITER}; Time: ", end='')
            t_start = time()
            train_acc_sub, val_acc_sub, train_conf_mat_sub, val_conf_mat_sub, train_class_rep, val_class_rep = \
                evaluation(X_train, y_train, X_val, y_val, n_classes=n_classes, method=learning_method)
            train_acc.append(train_acc_sub)
            val_acc.append(val_acc_sub)
            train_conf_mat.append(train_conf_mat_sub)
            val_conf_mat.append(val_conf_mat_sub)
            train_class_report.append(train_class_rep)
            val_class_report.append(val_class_rep)
            t_end = time()
            print(f"{t_end - t_start:.02f}s")

            pkl.dump(train_acc, open(results_folder / 'train_acc.pkl', 'wb'))
            pkl.dump(val_acc, open(results_folder / 'val_acc.pkl', 'wb'))
            pkl.dump(train_conf_mat, open(results_folder / 'train_conf_mat.pkl', 'wb'))
            pkl.dump(val_conf_mat, open(results_folder / 'val_conf_mat.pkl', 'wb'))
            pkl.dump(train_class_report, open(results_folder / 'train_class_report.pkl', 'wb'))
            pkl.dump(val_class_report, open(results_folder / 'val_class_report.pkl', 'wb'))

        print()

    
