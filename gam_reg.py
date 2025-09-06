from sklearn.base import BaseEstimator, RegressorMixin
from pygam import LinearGAM

class GAMRegressor(BaseEstimator, RegressorMixin):
    gam_ : LinearGAM = None
    """
    A scikit‐learn compatible wrapper for pygam.LinearGAM.
    
    __init__ parameters:
      - terms: list of pygam terms (e.g. [s(0), s(1), f(2)]) or a single 
               object sum(s(i) for i in ...) 
      - lam_grid: 1D array or list of lambda‐penalty values to try in `gridsearch`.
      - max_iter: how many boosting/iterations (if you want to specify num_splines, etc.). 
      - verbose: whether to print pygam's gridsearch progress.
    """
    def __init__(self, terms, lam=None):
        self.terms = terms
        # The fitted LinearGAM will be stored here:
        self.gam_ = None
        self.lam = lam if lam is not None else [0.001,0.01,0.1, 1, 10, 100]

    def fit(self, X, y):
        """
        X: 2D numpy array (n_samples × n_features) after any preprocessing
        y: 1D numpy array of targets
        """
        # Create a LinearGAM with the specified terms
        # (terms must be something like sum(s(i) for i in range(P)), or f(j) for categorical)
        self.gam_ = LinearGAM(terms=self.terms).gridsearch(X,y, lam=self.lam, objective="GCV")
        return self

    def predict(self, X):
        """
        Simply delegates to the internal pygam.LinearGAM.predict
        """
        return self.gam_.predict(X)

    def score(self, X, y):
        """ Returns R² (identity link) on the given data """
        return self.gam_.score(X, y)