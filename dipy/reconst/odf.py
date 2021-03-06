from __future__ import division, print_function, absolute_import

from warnings import warn, catch_warnings, simplefilter

from ..utils.six.moves import xrange

import numpy as np
import scipy.optimize as opt
from .recspeed import local_maxima, remove_similar_vertices, search_descending
from ..core.onetime import auto_attr
from dipy.core.sphere import HemiSphere, Sphere
from dipy.data import get_sphere


# Classes OdfModel and OdfFit are using API ReconstModel and ReconstFit from
# .base

default_sphere = HemiSphere.from_sphere(get_sphere('symmetric724'))


def peak_directions_nl(sphere_eval, relative_peak_threshold=.25,
                       min_separation_angle=45, sphere=default_sphere,
                       xtol=1e-7):
    """Non Linear Direction Finder

    Parameters
    ----------
    sphere_eval : callable
        A function which can be evaluated on a sphere.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close only
        the larger of the two is returned.
    sphere : Sphere
        A discrete Sphere. The points on the sphere will be used for initial
        estimate of maximums.
    xtol : float
        Relative tolerance for optimization.

    Returns
    -------
    directions : array (N, 3)
        Points on the sphere corresponding to N local maxima on the sphere.
    values : array (N,)
        Value of sphere_eval at each point on directions.

    """
    # Find discrete peaks for use as seeds in non-linear search
    discrete_values = sphere_eval(sphere)
    values, indices = local_maxima(discrete_values, sphere.edges)

    seeds = np.column_stack([sphere.theta[indices], sphere.phi[indices]])

    # Helper function
    def _helper(x):
        sphere = Sphere(theta=x[0], phi=x[1])
        return -sphere_eval(sphere)

    # Non-linear search
    num_seeds = len(seeds)
    theta = np.empty(num_seeds)
    phi = np.empty(num_seeds)
    for i in xrange(num_seeds):
        peak = opt.fmin(_helper, seeds[i], xtol=xtol, disp=False)
        theta[i], phi[i] = peak

    # Evaluate on new-found peaks
    small_sphere = Sphere(theta=theta, phi=phi)
    values = sphere_eval(small_sphere)

    # Sort in descending order
    order = values.argsort()[::-1]
    values = values[order]
    directions = small_sphere.vertices[order]

    # Remove directions that are too small
    n = search_descending(values, relative_peak_threshold)
    directions = directions[:n]

    # Remove peaks too close to each-other
    directions, idx = remove_similar_vertices(directions, min_separation_angle,
                                              return_index=True)
    values = values[idx]
    return directions, values


class OdfModel(object):
    """An abstract class to be sub-classed by specific odf models

    All odf models should provide a fit method which may take data as it's
    first and only argument.
    """
    def fit(self, data):
        """To be implemented by specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


class OdfFit(object):
    def odf(self, sphere):
        """To be implemented but specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


def peak_directions(odf, sphere, relative_peak_threshold=.25,
                    min_separation_angle=45):
    """Get the directions of odf peaks

    Parameters
    ----------
    odf : 1d ndarray
        The odf function evaluated on the vertices of `sphere`
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90] The minimum distance between
        directions. If two peaks are too close only the larger of the two is
        returned.

    Returns
    -------
    directions : (N, 3) ndarray
        N vertices for sphere, one for each peak
    values : (N,) ndarray
        peak values
    indices : (N,) ndarray
        peak indices of the directions on the sphere

    """
    values, indices = local_maxima(odf, sphere.edges)
    # If there is only one peak return
    if len(indices) == 1:
        return sphere.vertices[indices], values, indices

    n = search_descending(values, relative_peak_threshold)
    indices = indices[:n]
    directions = sphere.vertices[indices]
    directions, uniq = remove_similar_vertices(directions,
                                               min_separation_angle,
                                               return_index=True)
    values = values[uniq]
    indices = indices[uniq]
    return directions, values, indices


class PeaksAndMetrics(object):
    pass


def peaks_from_model(model, data, sphere, relative_peak_threshold,
                     min_separation_angle, mask=None, return_odf=False,
                     return_sh=True, gfa_thr=0, normalize_peaks=False,
                     sh_order=8, sh_basis_type=None):
    """Fits the model to data and computes peaks and metrics

    Parameters
    ----------
    model : a model instance
        `model` will be used to fit the data.
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90] The minimum distance between
        directions. If two peaks are too close only the larger of the two is
        returned.
    mask : array, optional
        If `mask` is provided, voxels that are False in `mask` are skipped and
        no peaks are returned.
    return_odf : bool
        If True, the odfs are returned.
    return_sh : bool
        If True, the odf as spherical harmonics coefficients is returned
    gfa_thr : float
        Voxels with gfa less than `gfa_thr` are skipped, no peaks are returned.
    normalize_peaks : bool
        If true, all peak values are calculated relative to `max(odf)`.
    sh_order : int, optional
        Maximum SH order in the SH fit.  For `sh_order`, there will be
        ``(sh_order + 1) * (sh_order + 2) / 2`` SH coefficients (default 8).
    sh_basis_type : {None, 'mrtrix', 'fibernav'}
        ``None`` for the default dipy basis which is the fibernav basis,
        ``mrtrix`` for the MRtrix basis, and
        ``fibernav`` for the FiberNavigator basis
    
    Returns
    -------
    pam : PeaksAndMetrics
        an object with ``gfa``, ``peak_values``, ``peak_indices``, ``odf``,
        ``shm_coeffs`` as attributes

    """

    data_flat = data.reshape((-1, data.shape[-1]))
    size = len(data_flat)
    if mask is None:
        mask = np.ones(size, dtype='bool')
    else:
        mask = mask.ravel()
        if len(mask) != size:
            raise ValueError("mask is not the same size as data")

    npeaks = 5
    sh_smooth=0
    gfa_array = np.zeros(size)
    qa_array = np.zeros((size, npeaks))
    peak_values = np.zeros((size, npeaks))
    peak_indices = np.zeros((size, npeaks), dtype='int')
    peak_indices.fill(-1)

    if return_sh:

        #import here to avoid circular imports
        from dipy.reconst.shm import sph_harm_lookup, smooth_pinv

        sph_harm_basis = sph_harm_lookup.get(sh_basis_type)
        if sph_harm_basis is None:
            raise ValueError("Invalid basis name.")
        B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
        L = -n * (n + 1)
        invB = smooth_pinv(B, np.sqrt(sh_smooth) * L)
        n_shm_coeff = (sh_order + 2) * (sh_order + 1) / 2
        shm_coeff = np.zeros((size, n_shm_coeff))
        invB = invB.T
        #sh = np.dot(sf, invB.T)

    if return_odf:
        odf_array = np.zeros((size, len(sphere.vertices)))

    global_max = -np.inf
    for i, sig in enumerate(data_flat):
        if not mask[i]:
            continue
        odf = model.fit(sig).odf(sphere)

        if return_sh:
            shm_coeff[i] = np.dot(odf, invB)

        if return_odf:
            odf_array[i] = odf

        gfa_array[i] = gfa(odf)
        if gfa_array[i] < gfa_thr:
            global_max = max(global_max, odf.max())
            continue

        # Get peaks of odf
        _, pk, ind = peak_directions(odf, sphere, relative_peak_threshold,
                                     min_separation_angle)

        # Calculate peak metrics
        global_max = max(global_max, pk[0])
        n = min(npeaks, len(pk))
        qa_array[i, :n] = pk[:n] - odf.min()
        if normalize_peaks:
            peak_values[i, :n] = pk[:n] / pk[0]
        else:
            peak_values[i, :n] = pk[:n]
        peak_indices[i, :n] = ind[:n]

    shape = data.shape[:-1]
    gfa_array = gfa_array.reshape(shape)
    qa_array = qa_array.reshape(shape + (npeaks,)) / global_max
    peak_values = peak_values.reshape(shape + (npeaks,))
    peak_indices = peak_indices.reshape(shape + (npeaks,))

    pam = PeaksAndMetrics()
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.gfa = gfa_array
    pam.qa = qa_array

    if return_sh:
        pam.shm_coeff = shm_coeff.reshape(shape + (n_shm_coeff,))
        pam.invB = invB
    else:
        pam.shm_coeff = None
        pam.invB = None

    if return_odf:
        pam.odf = odf_array.reshape(shape + odf_array.shape[-1:])
    else:
        pam.odf = None

    return pam


def gfa(samples):
    """The general fractional anisotropy of a function evaluated on the unit sphere"""
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n*(diff*diff).sum(-1)
    denom = (n-1)*(samples*samples).sum(-1)
    return np.sqrt(numer/denom)


def minmax_normalize(samples, out=None):
    """Min-max normalization of a function evaluated on the unit sphere

    Normalizes samples to ``(samples - min(samples)) / (max(samples) -
    min(samples))`` for each unit sphere.

    Parameters
    ----------
    samples : ndarray (..., N)
        N samples on a unit sphere for each point, stored along the last axis
        of the array.
    out : ndrray (..., N), optional
        An array to store the normalized samples.

    Returns
    -------
    out : ndarray, (..., N)
        Normalized samples.

    """
    if out is None:
        dtype = np.common_type(np.empty(0, 'float32'), samples)
        out = np.array(samples, dtype=dtype, copy=True)
    else:
        out[:] = samples

    sample_mins = np.min(samples, -1)[..., None]
    sample_maxes = np.max(samples, -1)[..., None]
    out -= sample_mins
    out /= (sample_maxes - sample_mins)
    return out
