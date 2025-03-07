"""
Perform a parameter estimation analysis with bilby on a synthetic signal with support for waveform uncertainty model SEOBNRv4CE and SEOBNRv4CE0.
"""
import numpy as np
import bilby
import sys
import argparse
import lalsimulation as LS


def setup_waveform_generator_standard(waveform_approximant: str, reference_frequency, minimum_frequency, duration, sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, waveform_arguments=None,  fundamental_mode=False
):
    # Check for waveform approcimant string
    # FIXME: Not always raising error
    try:
        approximant = LS.GetApproximantFromString(waveform_approximant)
    except:
        raise ValueError(f'Unknown waveform approximant {waveform_approximant}')

    # Fixed arguments passed into the source model
    if waveform_arguments is None:
        waveform_arguments = dict(waveform_approximant=waveform_approximant, reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency, catch_waveform_errors=True)
        # Only for NRHybSur3dq8, if not all_mode
        if waveform_approximant=='NRHybSur3dq8' and fundamental_mode:
            waveform_arguments['mode_array']=[(2,2),(2,-2)]
    return bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=frequency_domain_source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments)


def setup_waveform_generator_with_uncertainty(reference_frequency, minimum_frequency, duration, sampling_frequency):
    waveform_approximant = 'SEOBNRv4_ROM'
    waveform_arguments = dict(waveform_approximant=waveform_approximant, 
        reference_frequency=reference_frequency, minimum_frequency=minimum_frequency)
    # Load SEOBNRv4 waveform uncertainty model
    sys.path.append('/work/pi_mpuerrer_uri_edu/ritesh/wf-uncertainty-marg')
    import eob_gpr_model
    gpr_model_data_path = '/work/pi_mpuerrer_uri_edu/projects/wf_uncertainty_marg/data/uncertainty_interpolation_Mtot-50_fmin-20.0.hdf5'
    wferr = eob_gpr_model.WaveformUncertaintyInterpolation()
    wferr.load_interpolation(gpr_model_data_path)
    waveform_arguments['waveform_error_model'] = wferr
    frequency_domain_source_model = eob_gpr_model.lal_binary_black_hole_with_waveform_uncertainty
    return setup_waveform_generator_standard(waveform_approximant, reference_frequency, minimum_frequency,
        duration, sampling_frequency, frequency_domain_source_model=frequency_domain_source_model,
        waveform_arguments=waveform_arguments)


def setup_ifos_and_injection(injection_parameters, sampling_frequency, duration, minimum_frequency,
    use_zero_noise: bool, ifo_list=['H1', 'L1', 'V1']
):
    start_time = injection_parameters['geocent_time'] - 3
    ifos = bilby.gw.detector.InterferometerList(ifo_list)
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency

    if use_zero_noise:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)
    else:
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)

    ifos.inject_signal(waveform_generator=waveform_generator_signal, parameters=injection_parameters)
    return ifos


def setup_standard_prior(mc_min=25.0, mc_max=50.0, q_min=0.125, chi_max=0.8):
    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    priors.pop('mass_1')
    priors.pop('mass_2')
    priors['chirp_mass'] = bilby.prior.Uniform(
        name='chirp_mass', latex_label='$M$', minimum=mc_min, maximum=mc_max, unit='$M_{\\odot}$')
    priors['mass_ratio'] = bilby.prior.Uniform(
        name='mass_ratio', latex_label='$q$', minimum=q_min, maximum=1.0)
    priors['geocent_time'] = bilby.core.prior.Uniform(
        minimum=injection_parameters['geocent_time'] - 0.1,
        maximum=injection_parameters['geocent_time'] + 0.1,
        name='geocent_time', latex_label='$t_c$', unit='$s$')
    priors['chi_1'] = bilby.core.prior.Uniform(name='chi_1', minimum=-chi_max, maximum=chi_max, latex_label='$\chi_1$', unit=None)
    priors['chi_2'] = bilby.core.prior.Uniform(name='chi_2', minimum=-chi_max, maximum=chi_max, latex_label='$\chi_2$', unit=None)

    # These parameters will not be sampled
    for key in ['tilt_1', 'tilt_2', 'phi_12', 'phi_jl']:
        if key in injection_parameters:
            priors[key] = injection_parameters[key]

    return priors


def setup_uncertainty_model_priors(priors, uncertainty_parameters, sigma):
    # Add Gaussian priors for uncertainty parameters in amplitude and phase
    for p in uncertainty_parameters:
        priors[p] = bilby.core.prior.Gaussian(mu=0.0, sigma=sigma, name=p, unit=None)


def pin_uncertainty_model_parameters(priors, uncertainty_parameters):
    # We don't sample in \epsilon here, instead we just set \epsilon = 0.
    for p in uncertainty_parameters:
        priors[p] = 0.0


def pin_standard_parameters(priors, injection_parameters):
    # Pin all standard parameters except geocent_time and phase
    for key in ['mass_ratio', 'chi_1', 'chi_2', 'luminosity_distance', 'ra', 'dec', 'theta_jn', 'psi']:
        priors[key] = injection_parameters[key]


def pin_extrinsic_parameters(priors, injection_parameters):
    # Pin all extrinsic parameters except geocent_time and phase
    for key in ['luminosity_distance', 'ra', 'dec', 'theta_jn', 'psi']:
        priors[key] = injection_parameters[key]


def pin_spin_parameters(priors, injection_parameters):
    # Pin chi_1 and chi_2
    for key in ['chi_1', 'chi_2']:
        priors[key] = injection_parameters[key]


def setup_uncertainty_parameters(injection_parameters, uncertainty_parameters):
    # Only needed for SEOBNRv4CE0 signal
    for p in uncertainty_parameters:
        injection_parameters[p] = 0.0


def setup_likelihood_and_sampler(ifos, waveform_generator_template, priors, label,
    sampler='dynesty', distance_marginalization=True, phase_marginalization=True, time_marginalization=True,
    sample = "acceptance-walk", naccept=100, npoints=1024, maxmcmc=5000, npool=64
):
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator_template, priors=priors,
        distance_marginalization=distance_marginalization, phase_marginalization=phase_marginalization,
        time_marginalization=time_marginalization)
    np.seterr(divide='ignore')

    if sampler == 'dynesty':
        # See https://lscsoft.docs.ligo.org/bilby/api/bilby.core.sampler.dynesty.Dynesty.html
        # and https://lscsoft.docs.ligo.org/bilby/dynesty-guide.html
        # Note: Should really first run a single analysis with sample="act-walk" to estimate a viable
        # number of accepted steps before switching to acceptance-walk.
        result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler=sampler,
            npoints=npoints, naccept=naccept, sample=sample,
            injection_parameters=injection_parameters, outdir=outdir,
            label=label, maxmcmc=maxmcmc, npool=npool,
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            result_class=bilby.gw.result.CBCResult
        )
    elif sampler == 'mcmc':
        # See https://lscsoft.docs.ligo.org/bilby/bilby-mcmc-guide.html
        sampler = "bilby_mcmc"
        nsamples = 5000             # This is the number of raw samples
        thin_by_nact = 0.2          # This sets the thinning factor
        ntemps = 8                  # The number of parallel-tempered chains -- use npool=8 or 4
        L1steps = 100               # The number of internal steps to take for each iteration
        proposal_cycle = 'gwA'      # GW-specific proposal cycle described in Table 1 of 2106.08730
        printdt = 60                # Print a progress update every 60s
        check_point_delta_t = 1800  # Checkpoint and create progress plots every 30m
        result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler=sampler,
            nsamples=nsamples, thin_by_nact=thin_by_nact, ntemps=ntemps,
            L1steps=L1steps, proposal_cycle=proposal_cycle, printdt=printdt,
            check_point_delta_t=check_point_delta_t,
            injection_parameters=injection_parameters, outdir=outdir,
            label=label, npool=npool,
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            result_class=bilby.gw.result.CBCResult
        )
    else:
        raise ValueError(f'Sampler {sampler} not supported')
    return result


def make_reduced_corner_plots(result):
    # Make corner plots for intrinsic parameters
    pars_to_plot = {}
    for p in ['chirp_mass', 'mass_ratio', 'theta_jn', 'luminosity_distance']:
        pars_to_plot[p] = result.injection_parameters[p]
    pars_to_plot['chi_1'] = result.injection_parameters['spin_1z'][()]
    pars_to_plot['chi_2'] = result.injection_parameters['spin_2z'][()]
    result.plot_corner(parameters=pars_to_plot, filename=f"{outdir}/corner_intrinsic1.png")

    pars_to_plot_eff = {}
    for p in ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']:
        pars_to_plot_eff[p] = result.injection_parameters[p]
    result.plot_corner(parameters=pars_to_plot_eff, filename=f"{outdir}/corner_intrinsic2.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze an aligned-spin injection with bilby.")
    parser.add_argument("--signal_approximant", type=str, required=True, help="Waveform model to use as signal.")
    parser.add_argument("--template_approximant", type=str, required=True, help="Waveform model to use as template.")
    parser.add_argument("--distance", type=float, required=True, help="Luminosity distance of the source binary in Mpc.")
    parser.add_argument("--mass-ratio", type=float, required=False, default=0.8143322596801715, help="Mass-ratio (q<=1) of the source binary.")
    parser.add_argument("--chi_1", type=float, required=False, default=0.32, help="Dimensionless spin chi of BH 1.")
    parser.add_argument("--chi_2", type=float, required=False, default=-0.57, help="Dimensionless spin chi of BH 2.")
    parser.add_argument("--time_marginalization", action="store_true", help="Use time marginalization.")
    parser.add_argument("--phase_marginalization", action="store_true", help="Use phase marginalization.")
    parser.add_argument("--distance_marginalization", action="store_true", help="Use distance marginalization.")
    parser.add_argument("--zero-noise", action="store_true", help="Use an average 'zero' noise realization.")
    parser.add_argument("--pin-parameters", action="store_true", help="Pin most standard parameters, except chirp mass.")
    parser.add_argument("--pin-extrinsic-parameters", action="store_true", help="Pin all extrinsic parameters.")
    parser.add_argument("--pin-spin-parameters", action="store_true", help="Pin spin parameters chi_1 and chi_2.")
    parser.add_argument("--fundamental_mode_signal", action="store_true", help="Set only (2,2) and (2,-2) mode for signal waveform.")
    parser.add_argument("--fundamental_mode_template", action="store_true", help="Set only (2,2) and (2,-2) mode for template waveform.")
    parser.add_argument("--sigma", type=float, required=False, default=1, help="Standard deviation for epsilon priors.")
    parser.add_argument("--npool", type=int, required=True, help="Number of parallel chains.")
    parser.add_argument("--naccept", type=int, default=120, help="Average number of accepted jumps during each chain.")
    parser.add_argument("--npoints", type=int, default=1024, help="Number live points for dynesty.")
    parser.add_argument("--maxmcmc", type=int, default=5000, help="maximum number of walks to use.")
    parser.add_argument("--label", type=str, default='', help="Label to be appended to 'signal_template'.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Old way
    # signal_str = args.signal_approximant + '_22' if args.fundamental_mode_signal else args.signal_approximant
    # template_str = args.template_approximant + '_22' if args.fundamental_mode_template else args.template_approximant
    # Updated way
    if "_22" in args.signal_approximant:
        signal_str = args.signal_approximant
        args.signal_approximant = args.signal_approximant.replace("_22", "")
        args.fundamental_mode_signal = True
    else:
        signal_str = args.signal_approximant
        args.signal_approximant = args.signal_approximant
        args.fundamental_mode_signal = False
        
    if "_22" in args.template_approximant:
        template_str = args.template_approximant
        args.template_approximant = args.template_approximant.replace("_22", "")
        args.fundamental_mode_template = True
    else:
        template_str = args.template_approximant
        args.template_approximant = args.template_approximant
        args.fundamental_mode_template = False
        
    label = signal_str + '_' + template_str
    if args.label != '':
        label += '_' + args.label
    outdir = 'outdir_' + label
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    # Define signal parameters
    injection_parameters = dict(
        chirp_mass=32.04990174188462, mass_ratio=args.mass_ratio, chi_1=args.chi_1, chi_2=args.chi_2,
        luminosity_distance=args.distance, theta_jn=2.83701491, psi=1.42892206,
        phase=1.3, geocent_time=1126259462.0, ra=-1.26157296, dec=1.94972503
    )

    duration = 4.0
    sampling_frequency = 2048.0
    minimum_frequency = 20.0
    reference_frequency = minimum_frequency

    np.random.seed(88170235) # For reproducibility with noise realizations

    waveform_generator_signal = setup_waveform_generator_standard(args.signal_approximant, reference_frequency,
                                                                  minimum_frequency, duration, sampling_frequency, fundamental_mode=args.fundamental_mode_signal)
    if args.template_approximant == 'SEOBNRv4CE' or args.template_approximant == 'SEOBNRv4CE0':
        waveform_generator_template = setup_waveform_generator_with_uncertainty(
            reference_frequency, minimum_frequency, duration, sampling_frequency)
        uncertainty_parameters = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                                  'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    else:
        waveform_generator_template = setup_waveform_generator_standard(
            args.template_approximant, reference_frequency, minimum_frequency, duration, sampling_frequency, fundamental_mode=args.fundamental_mode_template)

    ifos = setup_ifos_and_injection(injection_parameters, sampling_frequency, duration, minimum_frequency, use_zero_noise=args.zero_noise)
    if injection_parameters['mass_ratio'] < 0.3 and args.signal_approximant!="NRHybSur3dq8":
        q_min = 1.0/16.0
    elif injection_parameters['mass_ratio'] < 0.3 and args.signal_approximant=="NRHybSur3dq8":
        # Hard boundary for NRHybSur3dq8 model
        q_min = 1.0/10.0
    else:
        q_min = 1.0/4.0
    
    # Prior setup
    priors = setup_standard_prior(mc_min=25.0, mc_max=50.0, q_min=q_min, chi_max=0.8)
    if args.template_approximant == 'SEOBNRv4CE':
        setup_uncertainty_model_priors(priors, uncertainty_parameters, args.sigma)
    elif args.template_approximant == 'SEOBNRv4CE0':
        pin_uncertainty_model_parameters(priors, uncertainty_parameters)

    # Pinning all standard parameters except geocentric_time and phase
    if args.pin_parameters:
        pin_standard_parameters(priors, injection_parameters)
    
    # Pinning all extrinsic parameters expect geocentric_time and phase
    if args.pin_extrinsic_parameters:
        pin_extrinsic_parameters(priors, injection_parameters)

    # Pinning chi_1 and chi_2
    if args.pin_spin_parameters:
        pin_spin_parameters(priors, injection_parameters)

    result = setup_likelihood_and_sampler(ifos, waveform_generator_template, priors, label,
        sampler='dynesty', distance_marginalization=args.distance_marginalization,
        phase_marginalization=args.phase_marginalization, time_marginalization=args.time_marginalization,
        sample = "acceptance-walk", naccept=args.naccept, npoints=args.npoints, maxmcmc=args.maxmcmc, npool=args.npool)
    make_reduced_corner_plots(result)
