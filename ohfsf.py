import numpy as np
from minepy import MINE

class OHSFS:
    """
    Online Heterogeneous Streaming Feature Selection (OHSFS)
    implements MICCor and MICGain to select streaming features.
    """
    def __init__(self):
        self.selected_features = []  # indices of selected features
        self.mine = MINE(alpha=0.6, c=15)  # default parameters
        self.mean_corr = 0.0

    def mic(self, x, y):
        """
        Compute Maximal Information Coefficient (MIC) between two arrays.
        """
        self.mine.compute_score(x, y)
        return self.mine.mic()

    def update_mean_corr(self, X, y):
        """
        Recalculate mean correlation over selected features.
        X: 2D array, shape (n_samples, n_features)
        y: 1D array, shape (n_samples,)
        """
        if not self.selected_features:
            self.mean_corr = 0.0
            return
        corrs = []
        for idx in self.selected_features:
            corr = self.mic(X[:, idx], y)
            corrs.append(corr)
        self.mean_corr = np.mean(corrs)

    def mic_gain(self, X, y, feat_idx):
        """
        Compute MICGain for a new feature at feat_idx.
        """
        mic_ft = self.mic(X[:, feat_idx], y)
        if not self.selected_features:
            return mic_ft
        corrs = [self.mic(X[:, j], X[:, feat_idx]) for j in self.selected_features]
        return mic_ft - np.mean(corrs)

    def partial_fit(self, X, y, stream_order):
        """
        Process features in streaming order.
        X: full data array (n_samples, n_total_features)
        y: labels array (n_samples,)
        stream_order: iterable of feature indices in arrival order
        """
        for idx in stream_order:
            mic_ft = self.mic(X[:, idx], y)
            # STEP 1: discard low-correlation features
            if mic_ft <= self.mean_corr:
                continue
            # STEP 2: test MICGain
            gain = self.mic_gain(X, y, idx)
            if gain > 0:
                self.selected_features.append(idx)
                self.update_mean_corr(X, y)

    def transform(self, X):
        """
        Reduce X to selected features.
        """
        return X[:, self.selected_features]

if __name__ == '__main__':
    # Example usage
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    data = load_breast_cancer()
    X, y = data.data, data.target
    # Simulate streaming: random feature order
    n_feats = X.shape[1]
    stream_order = np.random.permutation(n_feats)

    selector = OHSFS()
    selector.partial_fit(X, y, stream_order)
    X_sel = selector.transform(X)

    # Evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.3, random_state=42
    )
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    print("Test accuracy on selected features:", clf.score(X_test, y_test))
