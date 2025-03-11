from seobnrv4ce.bbh_wf_uncertainty_bilby import \
setup_waveform_uncertainty_model, lal_binary_black_hole_with_waveform_uncertainty
import bilby
import numpy as np
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal


def test_generate_seobnrv4ce0():
    setup_waveform_uncertainty_model()

    # Arrange
    eps = np.zeros(20)
    param_names = [f"A{i}" for i in range(1, 11)] + [f"P{i}" for i in range(1, 11)]
    eps_params = dict(zip(param_names, eps))

    p = dict(
            mass_1=40.840315, mass_2=33.257586, a_1=0.32, a_2=-0.57, tilt_1=0.0, tilt_2=0.0,
            phi_12=0.0, phi_jl=0.0, luminosity_distance=410, theta_jn=2.83701491, psi=1.42892206,
            phase=1.3, geocent_time=1126259462.0, ra=-1.26157296, dec=1.94972503
        )
    p['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(p['mass_1'], p['mass_2'])
    p['mass_ratio'] = p['mass_2'] / p['mass_1']

    waveform_arguments = dict(waveform_approximant='SEOBNRv4_ROM', reference_frequency=20.0, minimum_frequency=20.0)
    frequency_array = np.linspace(0, 512.0, 10)

    # Act
    pol = lal_binary_black_hole_with_waveform_uncertainty(
            frequency_array,
            p['mass_1'],
            p['mass_2'],
            p['luminosity_distance'],
            p['a_1'],
            p['tilt_1'],
            p['phi_12'],
            p['a_2'],
            p['tilt_2'],
            p['phi_jl'],
            p['theta_jn'],
            p['phase'],
            **eps_params,
            **waveform_arguments)

    # Assert
    h_p_expected = np.array([0.00000000e+00+0.00000000e+00j, -1.11594310e-23-1.89957497e-23j,
        -3.00000400e-24-1.05235172e-23j,  4.92857890e-24+6.63461846e-24j,
        -2.84144750e-24+5.54284127e-24j, -6.58079748e-25+6.56762576e-25j,
        -8.45882527e-26-3.80904756e-26j,  5.44181196e-27-1.21965065e-26j,
        1.68445195e-27+1.68587853e-27j,  0.00000000e+00+0.00000000e+00j])
    assert np.allclose(pol['plus'], h_p_expected), "Unexpected result for h_plus in test_generate_seobnrv4ce0()."
