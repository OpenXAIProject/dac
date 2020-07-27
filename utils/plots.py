import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import torch
from utils.tensor import to_numpy

def scatter(X, **kwargs):
    if type(X) == torch.Tensor:
        X = to_numpy(X)
    labels = kwargs.pop('labels', None)
    if labels is not None and type(labels) == torch.Tensor:
        labels = to_numpy(labels)
    ax = kwargs.pop('ax', plt.gca())
    if kwargs.pop('no_ticks', False):
        ax.set_xticks([])
        ax.set_yticks([])
    if labels is None:
        c = kwargs.pop('color', 'k')
        ec = kwargs.pop('edgecolor', [0.2,0.2,0.2])
        ax.scatter(X[:,0], X[:,1], color=c, edgecolor=ec, **kwargs)
    else:
        ulabels = np.sort(np.unique(labels))
        colors = kwargs.pop('colors', cm.rainbow(np.linspace(0, 1, len(ulabels))))
        edgecolors = kwargs.pop('edgecolors', 0.6*colors)
        for (l, c, ec) in zip(ulabels, colors, edgecolors):
            ax.scatter(X[labels==l,0], X[labels==l,1],
                    color=c, edgecolor=ec, **kwargs)
        return ulabels, colors

def draw_ellipse(pos, cov, ax=None, **kwargs):
    if type(pos) != np.ndarray:
        pos = to_numpy(pos)
    if type(cov) != np.ndarray:
        cov = to_numpy(cov)
    ax = ax or plt.gca()
    U, s, Vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
    width, height = 2 * np.sqrt(s)
    for nsig in range(1, 6):
        ax.add_patch(Ellipse(pos, nsig*width, nsig*height, angle,
            alpha=0.5/nsig, **kwargs))

def scatter_mog(X, labels, mu, cov, ax=None):
    ax = ax or plt.gca()
    ulabels, colors = scatter(X, labels=labels, ax=ax, zorder=10)

    for i, l in enumerate(ulabels):
        draw_ellipse(mu[l], cov[l], ax=ax, fc=colors[i])
