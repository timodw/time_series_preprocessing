import numpy as np

def acc(y_true, y_pred, return_mapping=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    accuracy = sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size
    if return_mapping:
        mapping = {row: col for row, col in zip(ind[0], ind[1])}
        return accuracy, mapping
    else:
        return accuracy


def compute_purity(y_true, y_pred):
        """
        Calculate the purity, a measurement of quality for the clustering 
        results.
        
        Each cluster is assigned to the class which is most frequent in the 
        cluster.  Using these classes, the percent accuracy is then calculated.
        
        Returns:
          A number between 0 and 1.  Poor clusterings have a purity close to 0 
          while a perfect clustering has a purity of 1.
        """

        # get the set of unique cluster ids
        clusters = set(y_pred)

        # find out what class is most frequent in each cluster
        cluster_classes = {}
        correct = 0
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = np.where(y_pred == cluster)[0]

            cluster_labels = y_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)
        
        return float(correct) / len(y_pred)