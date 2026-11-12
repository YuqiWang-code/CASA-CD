"""Exact pixel counts at a fixed validation threshold grid."""
import numpy as np


THRESHOLDS = np.linspace(.10, .70, 121, dtype=np.float32)
DEFAULT_INDEX = int(np.searchsorted(THRESHOLDS, .5))


class ThresholdCounts:
    def __init__(self, thresholds=THRESHOLDS):
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.histogram = np.zeros((2, len(self.thresholds) + 1), dtype=np.int64)

    def update(self, probabilities, targets):
        bins = np.searchsorted(self.thresholds, probabilities, side='left')
        joint = bins.ravel() + self.histogram.shape[1] * targets.ravel().astype(np.int64)
        self.histogram += np.bincount(joint, minlength=self.histogram.size).reshape(self.histogram.shape)

    def curve(self):
        fp, tp = np.cumsum(self.histogram[:, :0:-1], axis=1)[:, ::-1]
        fn = self.histogram[1].sum() - tp
        tn = self.histogram[0].sum() - fp
        return [dict(threshold=round(float(threshold), 3), tp=int(t), fp=int(f), fn=int(n), tn=int(k),
                     F1=float(2*t / max(2*t+f+n, 1)), recall=float(t / max(t+n, 1)),
                     precision=float(t / max(t+f, 1)))
                for threshold, t, f, n, k in zip(self.thresholds, tp, fp, fn, tn)]

    def summary(self):
        curve = self.curve()
        best = max(curve, key=lambda point: (point['F1'], -abs(point['threshold'] - .5)))
        return dict(default=curve[DEFAULT_INDEX], best=best, curve=curve)
