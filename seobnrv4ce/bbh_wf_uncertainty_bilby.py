import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from bilby.gw.source import _base_lal_cbc_fd_waveform
from bilby.core import utils
from .uncertainty_model import WaveformUncertaintyInterpolation

# Originally implemented in https://git.ligo.org/michael.puerrer/bilby/-/blob/SEOBNRv4_uncertainty/bilby/gw/source.py#L103


_wferr = None

def setup_waveform_uncertainty_model():
    """
    Setup function which should be called once before using lal_binary_black_hole_with_waveform_uncertainty()
    """
    global _wferr
    if _wferr is None:
        print('Instantiating WaveformUncertaintyInterpolation() and loading uncertainty model from disk.')
        _wferr = WaveformUncertaintyInterpolation()
        _wferr.load_interpolation()


def lal_binary_black_hole_with_waveform_uncertainty(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
        P1, P2, P3, P4, P5, P6, P7, P8, P9, P10,
        **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation
    including a draw from a model for the internal uncertainty of the
    waveform. This model is currently only available for SEOBNRv4.

    Parameters
    ----------
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at coalescence
    A1, ..., A10: amplitude uncertainty at frequency nodes
    P1, ..., P10: phase uncertainty at frequency nodes
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='SEOBNRv4_ROM', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    if waveform_kwargs['waveform_approximant'] != 'SEOBNRv4_ROM':
        raise ValueError('Waveform uncertainty model is only available for SEOBNRv4_ROM.')

    # Compute waveform polarizations
    polarizations = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)

    Mtot_SI = (mass_1 + mass_2) * utils.solar_mass
    q = mass_2 / mass_1
    assert q <= 1.0
    spin_1z = a_1 * np.cos(tilt_1)
    spin_2z = a_2 * np.cos(tilt_2)
    # Apply amplitude and phase corrections from SEOBNRv4 uncertainty model
    uncertainty_parameters = np.array([
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
        P1, P2, P3, P4, P5, P6, P7, P8, P9, P10])
    damp, dphi, f_model = _wferr.draw_sample(Mtot_SI, q, spin_1z, spin_2z,
        eps=uncertainty_parameters)
    # For the current model degree = 10, and eps has size 2*degree.

    # We could use ext=0 (spline extrapolation) or ext=3 (use boundary value)
    # Below I use the safer ext=3 which uses constant extrapolation.
    # A cubic spline should be OK which is the default.
    delta_ampI = InterpolatedUnivariateSpline(f_model, damp.T, ext=3)
    delta_phiI = InterpolatedUnivariateSpline(f_model, dphi.T, ext=3)
    f = frequency_array
    idx = np.where((f >= waveform_kwargs['minimum_frequency']) &
                   (f <= waveform_kwargs['maximum_frequency']))
    wf_fac = (1 + delta_ampI(f[idx])) \
        * (2 + 1j * delta_phiI(f[idx])) / (2 - 1j * delta_phiI(f[idx]))

    # Apply amplitude and phase error model
    for key, pol in polarizations.items():
        pol[idx] *= wf_fac

    return polarizations
