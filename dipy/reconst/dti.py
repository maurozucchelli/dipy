#!/usr/bin/python
""" Classes and functions for fitting tensors """
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

import scipy.optimize as opt

from dipy.utils.six.moves import range
from dipy.data import get_sphere
from ..core.geometry import vector_norm
from .vec_val_sum import vec_val_vect
from ..core.onetime import auto_attr


def _roll_evals(evals, axis=-1):
    """
    Helper function to check that the evals provided to functions calculating
    tensor statistics have the right shape

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor. shape should be (...,3).

    axis : int
        The axis of the array which contains the 3 eigenvals. Default: -1

    Returns
    -------
    evals : array-like
        Eigenvalues of a diffusion tensor, rolled so that the 3 eigenvals are
        the last axis.
    """
    if evals.shape[-1] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[-1])
        raise ValueError(msg)

    evals = np.rollaxis(evals, axis)

    return evals


def fractional_anisotropy(evals, axis=-1):
    r"""
    Fractional anisotropy (FA) of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    fa : array
        Calculated FA. Range is 0 <= FA <= 1.

    Notes
    --------
    FA is calculated using the following equation:

    .. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2}}

    """
    evals = _roll_evals(evals, axis)
    # Make sure not to get nans
    all_zero = (evals == 0).all(axis=0)
    ev1, ev2, ev3 = evals
    fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2)
                  / ((evals * evals).sum(0) + all_zero))

    return fa


def mean_diffusivity(evals, axis=-1):
    r"""
    Mean Diffusivity (MD) of a diffusion tensor. Also, called
    Apparent diffusion coefficient (ADC)

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    md : array
        Calculated MD.

    Notes
    --------
    MD is calculated with the following equation:

    .. math::

        MD = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}

    """
    evals = _roll_evals(evals, axis)
    return evals.mean(0)


def axial_diffusivity(evals, axis=-1):
    r"""
    Axial Diffusivity (AD) of a diffusion tensor.
    Also called parallel diffusivity.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor, must be sorted in descending order
        along `axis`.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    ad : array
        Calculated AD.

    Notes
    --------
    AD is calculated with the following equation:

    .. math::

        AD = \lambda_1

    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return ev1


def radial_diffusivity(evals, axis=-1):
    r"""
    Radial Diffusivity (RD) of a diffusion tensor.
    Also called perpendicular diffusivity.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor, must be sorted in descending order
        along `axis`.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

        Returns
    -------
    rd : array
        Calculated RD.

    Notes
    --------
    RD is calculated with the following equation:

    .. math::

        RD = \frac{\lambda_2 + \lambda_3}{2}

    """
    evals = _roll_evals(evals, axis)
    return evals[1:].mean(0)


def trace(evals, axis=-1):
    r"""
    Trace of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    trace : array
        Calculated trace of the diffusion tensor.

    Notes
    --------
    Trace is calculated with the following equation:

    .. math::

        Trace = \lambda_1 + \lambda_2 + \lambda_3

    """
    evals = _roll_evals(evals, axis)
    return evals.sum(0)


def color_fa(fa, evecs):
    r""" Color fractional anisotropy of diffusion tensor

    Parameters
    ----------
    fa : array-like
        Array of the fractional anisotropy (can be 1D, 2D or 3D)

    evecs : array-like
        eigen vectors from the tensor model

    Returns
    -------
    rgb : Array with 3 channels for each color as the last dimension.
        Colormap of the FA with red for the x value, y for the green
        value and z for the blue value.

    Note
    -----

    It is computed from the clipped FA between 0 and 1 using the following
    formula

    .. math::

        rgb = abs(max(\vec{e})) \times fa
    """

    if (fa.shape != evecs[..., 0, 0].shape) or ((3, 3) != evecs.shape[-2:]):
        raise ValueError("Wrong number of dimensions for evecs")

    return np.abs(evecs[..., 0]) * np.clip(fa, 0, 1)[..., None]


# The following are used to calculate the tensor mode:
def determinant(q_form):
    """
    The determinant of a tensor, given in quadratic form

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x, y, z, 3, 3) or (n, 3, 3) or (3, 3).

    Returns
    -------
    det : array
        The determinant of the tensor in each spatial coordinate
    """

    # Following the conventions used here:
    # http://en.wikipedia.org/wiki/Determinant
    aei = q_form[..., 0, 0] * q_form[..., 1, 1] * q_form[..., 2, 2]
    bfg = q_form[..., 0, 1] * q_form[..., 1, 2] * q_form[..., 2, 0]
    cdh = q_form[..., 0, 2] * q_form[..., 1, 0] * q_form[..., 2, 1]
    ceg = q_form[..., 0, 2] * q_form[..., 1, 1] * q_form[..., 2, 0]
    bdi = q_form[..., 0, 1] * q_form[..., 1, 0] * q_form[..., 2, 2]
    afh = q_form[..., 0, 0] * q_form[..., 1, 2] * q_form[..., 2, 1]
    return aei + bfg + cdh - ceg - bdi - afh


def isotropic(q_form):
    r"""
    Calculate the isotropic part of the tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    A_hat: ndarray
        The isotropic part of the tensor in each spatial coordinate

    Notes
    -----
    The isotropic part of a tensor is defined as (equations 3-5 of [1]_):

    .. math ::
        \bar{A} = \frac{1}{2} tr(A) I

    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """
    tr_A = q_form[..., 0, 0] + q_form[..., 1, 1] + q_form[..., 2, 2]
    n_dims = len(q_form.shape)
    add_dims = n_dims - 2  # These are the last two (the 3,3):
    my_I = np.eye(3)
    tr_AI = (tr_A.reshape(tr_A.shape + (1, 1)) * my_I)
    return (1 / 3.0) * tr_AI


def deviatoric(q_form):
    r"""
    Calculate the deviatoric (anisotropic) part of the tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    A_squiggle : ndarray
        The deviatoric part of the tensor in each spatial coordinate.

    Notes
    -----
    The deviatoric part of the tensor is defined as (equations 3-5 in [1]_):

    .. math ::
         \widetilde{A} = A - \bar{A}

    Where $A$ is the tensor quadratic form and $\bar{A}$ is the anisotropic
    part of the tensor.

    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """
    A_squiggle = q_form - isotropic(q_form)
    return A_squiggle


def norm(q_form):
    r"""
    Calculate the Frobenius norm of a tensor quadratic form

    Parameters
    ----------
    q_form: ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    norm : ndarray
        The Frobenius norm of the 3,3 tensor q_form in each spatial
        coordinate.

    Notes
    -----
    The Frobenius norm is defined as:

    :math:
        ||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}

    See also
    --------
    np.linalg.norm
    """
    return np.sqrt(np.sum(np.sum(np.abs(q_form ** 2), -1), -1))


def mode(q_form):
    r"""
    Mode (MO) of a diffusion tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x, y, z, 3, 3) or (n, 3, 3) or (3, 3).

    Returns
    -------
    mode : array
        Calculated tensor mode in each spatial coordinate.

    Notes
    -----
    Mode ranges between -1 (linear anisotropy) and +1 (planar anisotropy)
    with 0 representing orthotropy. Mode is calculated with the
    following equation (equation 9 in [1]_):

    .. math::

        Mode = 3*\sqrt{6}*det(\widetilde{A}/norm(\widetilde{A}))

    Where $\widetilde{A}$ is the deviatoric part of the tensor quadratic form.

    References
    ----------

    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """

    A_squiggle = deviatoric(q_form)
    A_s_norm = norm(A_squiggle)
    # Add two dims for the (3,3), so that it can broadcast on A_squiggle:
    A_s_norm = A_s_norm.reshape(A_s_norm.shape + (1, 1))
    return 3 * np.sqrt(6) * determinant((A_squiggle / A_s_norm))


def linearity(evals, axis=-1):
    r"""
    The linearity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    linearity : array
        Calculated linearity of the diffusion tensor.

    Notes
    --------
    Linearity is calculated with the following equation:

    .. math::

        Linearity = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\lambda_3}

    Notes
    -----
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (ev1 - ev2) / evals.sum(0)


def planarity(evals, axis=-1):
    r"""
    The planarity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    linearity : array
        Calculated linearity of the diffusion tensor.

    Notes
    --------
    Linearity is calculated with the following equation:

    .. math::

        Planarity = \frac{2 (\lambda_2-\lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

    Notes
    -----
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (2 * (ev2 - ev3) / evals.sum(0))


def sphericity(evals, axis=-1):
    r"""
    The sphericity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    sphericity : array
        Calculated sphericity of the diffusion tensor.

    Notes
    --------
    Linearity is calculated with the following equation:

    .. math::

        Sphericity = \frac{3 \lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

    Notes
    -----
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (3 * ev3) / evals.sum(0)


class TensorModel(object):
    """ Diffusion Tensor
    """
    def __init__(self, gtab, fit_method="WLS", *args, **kwargs):
        """ A Diffusion Tensor Model [1]_, [2]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'WLS' for weighted least squares
                dti.wls_fit_tensor
            'LS' for ordinary least squares
                dti.ols_fit_tensor

            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dti.wls_fit_tensor, dti.ols_fit_tensor for details

        References
        ----------
        .. [1] Basser, P.J., Mattiello, J., LeBihan, D., 1994. Estimation of
           the effective self-diffusion tensor from the NMR spin echo. J Magn
           Reson B 103, 247-254.
        .. [2] Basser, P., Pierpaoli, C., 1996. Microstructural and
           physiological features of tissues elucidated by quantitative
           diffusion-tensor MRI.  Journal of Magnetic Resonance 111, 209-219.

        """
        if not callable(fit_method):
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"' + str(fit_method) + '" is not a known fit '
                                 'method, the fit method should either be a '
                                 'function or one of the common fit methods')
        self.bvec = gtab.bvecs
        self.bval = gtab.bvals
        self.design_matrix = design_matrix(self.bvec.T, self.bval)
        self.args = args
        self.kwargs = kwargs


    def fit(self, data, mask=None):
        """ Fit method of the DTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]
        """
        # If a mask is provided, we will use it to access the data
        if mask is not None:
            # Make sure it's boolean, so that it can be used to mask
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = data[mask]
        else:
            data_in_mask = data

        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        dti_params = np.zeros(data.shape[:-1] + (12,))

        dti_params[mask, :] = params_in_mask

        return TensorFit(self, dti_params)


class TensorFit(object):
    def __init__(self, model, model_params):
        """ Initialize a TensorFit class instance.
        """
        self.model = model
        self.model_params = model_params

    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])

    @property
    def shape(self):
        return self.model_params.shape[:-1]

    @property
    def directions(self):
        """
        For tracking - return the primary direction in each voxel
        """
        return self.evecs[..., None, :, 0]

    @property
    def evals(self):
        """
        Returns the eigenvalues of the tensor as an array
        """
        return self.model_params[..., :3]

    @property
    def evecs(self):
        """
        Returns the eigenvectors of the tensor as an array
        """
        evecs = self.model_params[..., 3:]
        return evecs.reshape(self.shape + (3, 3))

    @property
    def quadratic_form(self):
        """Calculates the 3x3 diffusion tensor for each voxel"""
        # do `evecs * evals * evecs.T` where * is matrix multiply
        # einsum does this with:
        # np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)
        return vec_val_vect(self.evecs, self.evals)

    def lower_triangular(self, b0=None):
        return lower_triangular(self.quadratic_form, b0)

    @auto_attr
    def fa(self):
        """Fractional anisotropy (FA) calculated from cached eigenvalues."""
        return fractional_anisotropy(self.evals)

    @auto_attr
    def mode(self):
        """
        Tensor mode calculated from cached eigenvalues.
        """
        return mode(self.quadratic_form)

    @auto_attr
    def md(self):
        r"""
        Mean diffusitivity (MD) calculated from cached eigenvalues.

        Returns
        ---------
        md : array (V, 1)
            Calculated MD.

        Notes
        --------
        MD is calculated with the following equation:

        .. math::

            MD = \frac{\lambda_1+\lambda_2+\lambda_3}{3}

        """
        return self.trace / 3.0

    @auto_attr
    def rd(self):
        r"""
        Radial diffusitivity (RD) calculated from cached eigenvalues.

        Returns
        ---------
        rd : array (V, 1)
            Calculated RD.

        Notes
        --------
        RD is calculated with the following equation:

        .. math::

          RD = \frac{\lambda_2 + \lambda_3}{2}


        """
        return radial_diffusivity(self.evals)

    @auto_attr
    def ad(self):
        r"""
        Axial diffusivity (AD) calculated from cached eigenvalues.

        Returns
        ---------
        ad : array (V, 1)
            Calculated AD.

        Notes
        --------
        RD is calculated with the following equation:

        .. math::

          AD = \lambda_1


        """
        return axial_diffusivity(self.evals)

    @auto_attr
    def trace(self):
        r"""
        Trace of the tensor calculated from cached eigenvalues.

        Returns
        ---------
        trace : array (V, 1)
            Calculated trace.

        Notes
        --------
        The trace is calculated with the following equation:

        .. math::

          trace = \lambda_1 + \lambda_2 + \lambda_3
        """
        return trace(self.evals)

    @auto_attr
    def planarity(self):
        r"""
        Returns
        -------
        sphericity : array
            Calculated sphericity of the diffusion tensor [1]_.

        Notes
        --------
        Sphericity is calculated with the following equation:

        .. math::

            Sphericity = \frac{2 (\lambda2 - \lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

        Notes
        -----
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return planarity(self.evals)

    @auto_attr
    def linearity(self):
        r"""
        Returns
        -------
        linearity : array
            Calculated linearity of the diffusion tensor [1]_.

        Notes
        --------
        Linearity is calculated with the following equation:

        .. math::

            Linearity = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\lambda_3}

        Notes
        -----
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return linearity(self.evals)

    @auto_attr
    def sphericity(self):
        r"""
        Returns
        -------
        sphericity : array
            Calculated sphericity of the diffusion tensor [1]_.

        Notes
        --------
        Sphericity is calculated with the following equation:

        .. math::

            Sphericity = \frac{3 \lambda_3}{\lambda_1+\lambda_2+\lambda_3}

        Notes
        -----
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return sphericity(self.evals)

    def odf(self, sphere):
        lower = 4 * np.pi * np.sqrt(np.prod(self.evals, -1))
        projection = np.dot(sphere.vertices, self.evecs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            projection /= np.sqrt(self.evals)
            odf = (vector_norm(projection) ** -3) / lower
        # Zero evals are non-physical, we replace nans with zeros
        any_zero = (self.evals == 0).any(-1)
        odf = np.where(any_zero, 0, odf)
        # Move odf to be on the last dimension
        odf = np.rollaxis(odf, 0, odf.ndim)
        return odf


def wls_fit_tensor(design_matrix, data, min_signal=1):
    r"""
    Computes weighted least squares (WLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])


    See Also
    --------
    decompose_tensor

    Notes
    -----
    In Chung, et al. 2006, the regression of the WLS fit needed an unbiased
    preliminary estimate of the weights and therefore the ordinary least
    squares (OLS) estimates were used. A "two pass" method was implemented:

        1. calculate OLS estimates of the data
        2. apply the OLS estimates as weights to the WLS fit of the data

    This ensured heteroscadasticity could be properly modeled for various
    types of bootstrap resampling (namely residual bootstrap).

    .. math::

        y = \mathrm{data} \\
        X = \mathrm{design matrix} \\
        \hat{\beta}_\mathrm{WLS} = \mathrm{desired regression coefficients (e.g. tensor)}\\
        \\
        \hat{\beta}_\mathrm{WLS} = (X^T W X)^{-1} X^T W y \\
        \\
        W = \mathrm{diag}((X \hat{\beta}_\mathrm{OLS})^2),
        \mathrm{where} \hat{\beta}_\mathrm{OLS} = (X^T X)^{-1} X^T y

    References
    ----------
    .. [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
       approaches for estimation of uncertainties of DTI parameters.
       NeuroImage 33, 531-541.
    """
    tol = 1e-6
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dti_params = np.empty((len(data_flat), 4, 3))

    #obtain OLS fitting matrix
    #U,S,V = np.linalg.svd(design_matrix, False)
    #math: beta_ols = inv(X.T*X)*X.T*y
    #math: ols_fit = X*beta_ols*inv(y)
    #ols_fit = np.dot(U, U.T)
    ols_fit = _ols_fit_matrix(design_matrix)
    min_diffusivity = tol / -design_matrix.min()

    for param, sig in zip(dti_params, data_flat):
        param[0], param[1:] = _wls_iter(ols_fit, design_matrix, sig,
                                        min_signal, min_diffusivity)
    dti_params.shape = data.shape[:-1] + (12,)
    dti_params = dti_params
    return dti_params


def _wls_iter(ols_fit, design_matrix, sig, min_signal, min_diffusivity):
    ''' Helper function used by wls_fit_tensor.
    '''
    sig = np.maximum(sig, min_signal)  # throw out zero signals
    log_s = np.log(sig)
    w = np.exp(np.dot(ols_fit, log_s))
    D = np.dot(np.linalg.pinv(design_matrix * w[:, None]), w * log_s)
    # D, _, _, _ = np.linalg.lstsq(design_matrix * w[:, None], log_s)
    tensor = from_lower_triangular(D)
    return decompose_tensor(tensor, min_diffusivity=min_diffusivity)


def _ols_iter(inv_design, sig, min_signal, min_diffusivity):
    ''' Helper function used by ols_fit_tensor.
    '''
    sig = np.maximum(sig, min_signal)  # throw out zero signals
    log_s = np.log(sig)
    D = np.dot(inv_design, log_s)
    tensor = from_lower_triangular(D)
    return decompose_tensor(tensor, min_diffusivity=min_diffusivity)


def ols_fit_tensor(design_matrix, data, min_signal=1):
    r"""
    Computes ordinary least squares (OLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])


    See Also
    --------
    WLS_fit_tensor, decompose_tensor, design_matrix

    Notes
    -----
    .. math::

        y = \mathrm{data} \\
        X = \mathrm{design matrix} \\

        \hat{\beta}_\mathrm{OLS} = (X^T X)^{-1} X^T y

    References
    ----------
    ..  [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
        approaches for estimation of uncertainties of DTI parameters.
        NeuroImage 33, 531-541.
    """
    tol = 1e-6

    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    evals = np.empty((len(data_flat), 3))
    evecs = np.empty((len(data_flat), 3, 3))
    dti_params = np.empty((len(data_flat), 4, 3))

    #obtain OLS fitting matrix
    #U,S,V = np.linalg.svd(design_matrix, False)
    #math: beta_ols = inv(X.T*X)*X.T*y
    #math: ols_fit = X*beta_ols*inv(y)
    #ols_fit =  np.dot(U, U.T)

    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    for param, sig in zip(dti_params, data_flat):
        param[0], param[1:] = _ols_iter(inv_design, sig,
            min_signal, min_diffusivity)

    dti_params.shape = data.shape[:-1] + (12,)
    dti_params = dti_params
    return dti_params


def _ols_fit_matrix(design_matrix):
    """
    Helper function to calculate the ordinary least squares (OLS)
    fit as a matrix multiplication. Mainly used to calculate WLS weights. Can
    be used to calculate regression coefficients in OLS but not recommended.

    See Also:
    ---------
    wls_fit_tensor, ols_fit_tensor

    Example:
    --------
    ols_fit = _ols_fit_matrix(design_mat)
    ols_data = np.dot(ols_fit, data)
    """

    U, S, V = np.linalg.svd(design_matrix, False)
    return np.dot(U, U.T)


def _nlls_err_func(tensor, design_matrix, data, weighting=None,
                   sigma=None):
    """
    Error function for the non-linear least-squares fit of the tensor.

    Parameters
    ----------
    tensor : array (3,3)
        The 3-by-3 tensor matrix

    design_matrix : array
        The design matrix

    data : array
        The voxel signal in all gradient directions

    weighting : str (optional).
         Whether to use the Geman McClure weighting criterion (see [1]_
         for details)

    sigma : float or float array (optional)
        If 'sigma' weighting is used, we will weight the error function
        according to the background noise estimated either in aggregate over
        all directions (when a float is provided), or to an estimate of the
        noise in each diffusion-weighting direction (if an array is
        provided). If 'gmm', the Geman-Mclure M-estimator is used for
        weighting (see Notes.

    Notes
    -----
    The GemanMcClure M-estimator is described as follows [1]_ (page 1089): "The
    scale factor C affects the shape of the GMM [Geman-McClure M-estimator]
    weighting function and represents the expected spread of the residuals
    (i.e., the SD of the residuals) due to Gaussian distributed noise. The
    scale factor C can be estimated by many robust scale estimators. We used
    the median absolute deviation (MAD) estimator because it is very robust to
    outliers having a 50% breakdown point (6,7). The explicit formula for C
    using the MAD estimator is:

    .. math :: 

            C = 1.4826 x MAD = 1.4826 x median{|r1-\hat{r}|,... |r_n-\hat{r}|}

    where $\hat{r} = median{r_1, r_2, ..., r_3}$ and n is the number of data
    points. The multiplicative constant 1.4826 makes this an approximately
    unbiased estimate of scale when the error model is Gaussian." 


    References
    ----------
    [1] Chang, L-C, Jones, DK and Pierpaoli, C (2005). RESTORE: robust estimation
    of tensors by outlier rejection. MRM, 53: 1088-95.
    """
    # This is the predicted signal given the params:
    y = np.exp(np.dot(design_matrix, tensor))

    # Compute the residuals
    residuals = data - y

    # If we don't want to weight the residuals, we are basically done:
    if weighting is None:
       # And we return the SSE:
       return residuals

    se = residuals ** 2
    # If the user provided a sigma (e.g 1.5267 * std(background_noise), as
    # suggested by Chang et al.) we will use it:
    if weighting == 'sigma':
        if sigma is None:
             e_s = "Must provide sigma value as input to use this weighting"
             e_s += " method"
             raise ValueError(e_s)
        w = 1/(sigma**2)

    elif weighting == 'gmm':
        # We use the Geman McClure M-estimator to compute the weights on the
        # residuals:
        C = 1.4826 * np.median(residuals - np.median(residuals))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = 1/(se + C**2)

    # Return the weighted residuals:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return np.sqrt(w * se)


def _nlls_jacobian_func(tensor, design_matrix, data, *arg, **kwargs):
    """The Jacobian is the first derivative of the error function [1]_.

    Notes
    -----
    This is an implementation of equation 14 in [1]_.

    References
    ----------
    [1] Koay, CG, Chang, L-C, Carew, JD, Pierpaoli, C, Basser PJ (2006).
        A unifying theoretical and algorithmic framework for least squares
        methods of estimation in diffusion tensor imaging. MRM 182, 115-25.

    """
    pred = np.exp(np.dot(design_matrix, tensor))
    return -pred[:, None] * design_matrix


def nlls_fit_tensor(design_matrix, data, min_signal=1, weighting=None,
                    sigma=None, jac=True):
    """
    Fit the tensor params using non-linear least-squares.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.

    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    min_signal : float, optional
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting. Default = 1

    weighting: str
           the weighting scheme to use in considering the
           squared-error. Default behavior is to use uniform weighting. Other
           options: 'sigma' 'gmm'

    sigma: float
        If the 'sigma' weighting scheme is used, a value of sigma needs to be
        provided here. According to [Chang2005]_, a good value to use is
        1.5267 * std(background_noise), where background_noise is estimated
        from some part of the image known to contain no signal (only noise).

    jac : bool
        Use the Jacobian? Default: True

    Returns
    -------
    nlls_params: the eigen-values and eigen-vectors of the tensor in each voxel.

    """
    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))
    # Use the OLS method parameters as the starting point for the optimization:
    inv_design = np.linalg.pinv(design_matrix)
    sig = np.maximum(flat_data, min_signal)
    log_s = np.log(sig)
    D = np.dot(inv_design, log_s.T).T

    # Flatten for the iteration over voxels:
    ols_params = np.reshape(D, (-1, D.shape[-1]))
    # 12 parameters per voxel (evals + evecs):
    dti_params = np.empty((flat_data.shape[0], 12))
    for vox in range(flat_data.shape[0]):
        start_params = ols_params[vox]
        # Do the optimization in this voxel:
        if jac:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    weighting,
                                                    sigma),
                                              Dfun=_nlls_jacobian_func)
        else:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    weighting,
                                                    sigma))

        # The parameters are the evals and the evecs:
        try:
            evals,evecs=decompose_tensor(from_lower_triangular(this_tensor[:6]))
            dti_params[vox, :3] = evals
            dti_params[vox, 3:] = evecs.ravel()

        # If leastsq failed to converge and produced nans, we'll resort to the
        # OLS solution in this voxel:
        except np.linalg.LinAlgError:
            print(vox)
            dti_params[vox, :] = start_params

    dti_params.shape = data.shape[:-1] + (12,)
    return dti_params


def restore_fit_tensor(design_matrix, data, min_signal=1.0, sigma=None,
                       jac=True):
    """
    Use the RESTORE algorithm [Chang2005]_ to calculate a robust tensor fit

    Parameters
    ----------

    design_matrix : array of shape (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.

    data : array of shape ([X, Y, Z, n_directions], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    min_signal : float, optional
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting. Default = 1

    sigma : float
        An estimate of the variance. [Chang2005]_ recommend to use
        1.5267 * std(background_noise), where background_noise is estimated
        from some part of the image known to contain no signal (only noise).

    jac : bool, optional
        Whether to use the Jacobian of the tensor to speed the non-linear
        optimization procedure used to fit the tensor paramters (see also
        :func:`nlls_fit_tensor`). Default: True


    Returns
    -------
    restore_params : an estimate of the tensor parameters in each voxel.

    Note
    ----
    Chang, L-C, Jones, DK and Pierpaoli, C (2005). RESTORE: robust estimation
    of tensors by outlier rejection. MRM, 53: 1088-95.

    """

    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))
    # Use the OLS method parameters as the starting point for the optimization:
    inv_design = np.linalg.pinv(design_matrix)
    sig = np.maximum(flat_data, min_signal)
    log_s = np.log(sig)
    D = np.dot(inv_design, log_s.T).T
    ols_params = np.reshape(D, (-1, D.shape[-1]))
    # 12 parameters per voxel (evals + evecs):
    dti_params = np.empty((flat_data.shape[0], 12))
    for vox in range(flat_data.shape[0]):
        start_params = ols_params[vox]
        # Do nlls using sigma weighting in this voxel:
        if jac:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    'sigma',
                                                    sigma),
                                              Dfun=_nlls_jacobian_func)
        else:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                             args=(design_matrix,
                                                   flat_data[vox],
                                                   'sigma',
                                                   sigma))

        # Get the residuals:
        pred_sig = np.exp(np.dot(design_matrix, this_tensor))
        residuals = flat_data[vox] - pred_sig
        # If any of the residuals are outliers (using 3 sigma as a criterion
        # following Chang et al., e.g page 1089):
        if np.any(residuals > 3 * sigma):
            # Do nlls with GMM-weighting:
            if jac:
                this_tensor, status= opt.leastsq(_nlls_err_func,
                                                 start_params,
                                                 args=(design_matrix,
                                                       flat_data[vox],
                                                       'gmm'),
                                                 Dfun=_nlls_jacobian_func)
            else:
                this_tensor, status= opt.leastsq(_nlls_err_func,
                                                 start_params,
                                                 args=(design_matrix,
                                                       flat_data[vox],
                                                       'gmm'))

            # How are you doin' on those residuals?
            pred_sig = np.exp(np.dot(design_matrix, this_tensor))
            residuals = flat_data[vox] - pred_sig
            if np.any(residuals > 3 * sigma):
                # If you still have outliers, refit without those outliers:
                non_outlier_idx = np.where(residuals <= 3 * sigma)
                clean_design = design_matrix[non_outlier_idx]
                clean_sig = flat_data[vox][non_outlier_idx]
                if len(sigma>1):
                    this_sigma = sigma[non_outlier_idx]
                else:
                    this_sigma = sigma

                if jac:
                    this_tensor, status= opt.leastsq(_nlls_err_func,
                                                     start_params,
                                                     args=(clean_design,
                                                           clean_sig,
                                                           'sigma',
                                                           this_sigma),
                                                     Dfun=_nlls_jacobian_func)
                else:
                    this_tensor, status= opt.leastsq(_nlls_err_func,
                                                     start_params,
                                                     args=(clean_design,
                                                           clean_sig,
                                                           'sigma',
                                                           this_sigma))

        # The parameters are the evals and the evecs:
        try:
            evals,evecs=decompose_tensor(from_lower_triangular(this_tensor[:6]))
            dti_params[vox, :3] = evals
            dti_params[vox, 3:] = evecs.ravel()

        # If leastsq failed to converge and produced nans, we'll resort to the
        # OLS solution in this voxel:
        except np.linalg.LinAlgError:
            print(vox)
            dti_params[vox, :] = start_params


    dti_params.shape = data.shape[:-1] + (12,)
    restore_params = dti_params
    return restore_params





_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])


def from_lower_triangular(D):
    """ Returns a tensor given the six unique tensor elements

    Given the six unique tensor elments (in the order: Dxx, Dxy, Dyy, Dxz, Dyz,
    Dzz) returns a 3 by 3 tensor. All elements after the sixth are ignored.

    Parameters
    -----------
    D : array_like, (..., >6)
        Unique elements of the tensors

    Returns
    --------
    tensor : ndarray (..., 3, 3)
        3 by 3 tensors

    """
    return D[..., _lt_indices]


_lt_rows = np.array([0, 1, 1, 2, 2, 2])
_lt_cols = np.array([0, 0, 1, 0, 1, 2])


def lower_triangular(tensor, b0=None):
    """
    Returns the six lower triangular values of the tensor and a dummy variable
    if b0 is not None

    Parameters
    ----------
    tensor : array_like (..., 3, 3)
        a collection of 3, 3 diffusion tensors
    b0 : float
        if b0 is not none log(b0) is returned as the dummy variable

    Returns
    -------
    D : ndarray
        If b0 is none, then the shape will be (..., 6) otherwise (..., 7)

    """
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("Diffusion tensors should be (..., 3, 3)")
    if b0 is None:
        return tensor[..., _lt_rows, _lt_cols]
    else:
        D = np.empty(tensor.shape[:-2] + (7,), dtype=tensor.dtype)
        D[..., 6] = -np.log(b0)
        D[..., :6] = tensor[..., _lt_rows, _lt_cols]
        return D


def eig_from_lo_tri(data):
    """Calculates parameters for creating a Tensor instance

    Calculates tensor parameters from the six unique tensor elements. This
    function can be passed to the Tensor class as a fit_method for creating a
    Tensor instance from tensors stored in a nifti file.

    Parameters
    ----------
    data : array_like (..., 6)
        diffusion tensors elements stored in lower triangular order

    Returns
    -------
    dti_params
        Eigen values and vectors, used by the Tensor class to create an
        instance
    """
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dti_params = np.empty((len(data_flat), 4, 3))

    for ii in range(len(data_flat)):
        tensor = from_lower_triangular(data_flat[ii])
        eigvals, eigvecs = decompose_tensor(tensor)
        dti_params[ii, 0] = eigvals
        dti_params[ii, 1:] = eigvecs

    dti_params.shape = data.shape[:-1] + (12,)
    return dti_params


def decompose_tensor(tensor, min_diffusivity=0):
    """ Returns eigenvalues and eigenvectors given a diffusion tensor

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors (Basser et al., 1994a).

    Parameters
    ----------
    tensor : array (3, 3)
        Hermitian matrix representing a diffusion tensor.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    eigvals : array (3,)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])

    """
    #outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = np.linalg.eigh(tensor)

    #need to sort the eigenvalues and associated eigenvectors
    order = eigenvals.argsort()[::-1]
    eigenvecs = eigenvecs[:, order]
    eigenvals = eigenvals[order]

    eigenvals = eigenvals.clip(min=min_diffusivity)
    # eigenvecs: each vector is columnar

    return eigenvals, eigenvecs


def design_matrix(gtab, bval, dtype=None):
    """  Constructs design matrix for DTI weighted least squares or
    least squares fitting. (Basser et al., 1994a)

    Parameters
    ----------
    gtab : array with shape (3,g)
        Diffusion gradient table found in DICOM header as a numpy array.
    bval : array with shape (g,)
        Diffusion weighting factor b for each vector in gtab.
    dtype : string
        Parameter to control the dtype of returned designed matrix

    Returns
    -------
    design_matrix : array (g,7)
        Design matrix or B matrix assuming Gaussian distributed tensor model
        design_matrix[j, :] = (Bxx, Byy, Bzz, Bxy, Bxz, Byz, dummy)
    """
    G = gtab
    B = np.zeros((bval.size, 7), dtype=G.dtype)
    if gtab.shape[1] != bval.shape[0]:
        raise ValueError('The number of b values and gradient directions must'
                          + ' be the same')
    B[:, 0] = G[0, :] * G[0, :] * 1. * bval   # Bxx
    B[:, 1] = G[0, :] * G[1, :] * 2. * bval   # Bxy
    B[:, 2] = G[1, :] * G[1, :] * 1. * bval   # Byy
    B[:, 3] = G[0, :] * G[2, :] * 2. * bval   # Bxz
    B[:, 4] = G[1, :] * G[2, :] * 2. * bval   # Byz
    B[:, 5] = G[2, :] * G[2, :] * 1. * bval   # Bzz
    B[:, 6] = np.ones(bval.size)
    return -B


def quantize_evecs(evecs, odf_vertices=None):
    """ Find the closest orientation of an evenly distributed sphere

    Parameters
    ----------
    evecs : ndarray
    odf_vertices : None or ndarray
        If None, then set vertices from symmetric362 sphere.  Otherwise use
        passed ndarray as vertices

    Returns
    -------
    IN : ndarray
    """
    max_evecs = evecs[..., :, 0]
    if odf_vertices == None:
        odf_vertices = get_sphere('symmetric362').vertices
    tup = max_evecs.shape[:-1]
    mec = max_evecs.reshape(np.prod(np.array(tup)), 3)
    IN = np.array([np.argmin(np.dot(odf_vertices, m)) for m in mec])
    IN = IN.reshape(tup)
    return IN

common_fit_methods = {'WLS': wls_fit_tensor,
                      'LS': ols_fit_tensor,
                      'OLS': ols_fit_tensor,
                      'NLLS': nlls_fit_tensor,
                      'RT': restore_fit_tensor,
                      'restore':restore_fit_tensor,
                     }
