# The main components of this module are two classes:
#
# 1. WaveformUncertainty. Represents the uncertainty associated to a
# SEOBNRv4 waveform at a given calibration point, ie. the uncertainty
# at a single point in physical parameter space.
#
# 2. WaveformUncertaintyInterpolation. Represents the uncertainty
# associated to the entire SEOBNRv4 model, ie. the uncertainty
# interpolated throughout physical parameter space.
#
#
# This module does not require Roberto's modified lalsuite, however it
# does require a database of waveforms *or* a previously saved
# uncertainty model.

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import stats
import scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor

data_dir = r'../data/'

#
# Uncertainty at a given point in physical parameter space.
#

class WaveformUncertainty():
    """Represents the waveform uncertainty at one point in physical
    parameter space.

    Requires previously-generated database of sampled waveforms.
    """

    EPSILON = 1e-9

    def __init__(self, filename, cfg, f_nodes):
        """Initializes object to contain waveform samples drawn from the
        calibration posterior at a particular configuration. This data
        should have been previously saved in a database.

        After extracting the data, the reduced representation of the
        waveform distribution is built and stored.

        Args:
        -----
        filename:  string, filename of a database of waveforms

        cfg:  string, name of particular physical configuration at
            which to build the object

        f_nodes: frequency nodes at which to represent the
            distribution over waveforms

        """
        fs = h5py.File(filename, 'r')

        self.cfg = cfg
        self.q = fs[cfg].attrs['q']
        self.chi1 = fs[cfg].attrs['chi1']
        self.chi2 = fs[cfg].attrs['chi2']
        self.Mtot = fs.attrs['Mtot']
        self.f_grid = np.array(fs[cfg+'/f_grid'][:])
        self.f_min = self.f_grid[0]
        self.f_max = self.f_grid[-1]
        amp = np.array(fs[cfg+'/amp'][:])
        phi = np.array(fs[cfg+'/phi'][:])
        amp_eob = np.array(fs[cfg+'/amp_EOB'][:])
        phi_eob = np.array(fs[cfg+'/phi_EOB'][:])

        self.nsamples = len(amp)

        self.generate_diffs(amp, phi, amp_eob, phi_eob)
        self.generate_reduced_representation(f_nodes)
        self.generate_mean_and_cov()

        fs.close()

    def generate_diffs(self, amp_draw_array, phi_draw_array, amp_eob, phi_eob):
        """Computes the deviations between sampled waveforms and EOB waveform.
        Stores the amplitude deviation as the fractional deviation
        from the EOB waveform.

        Args:
        -----
        amp_draw_array:  array of amplitudes of sample waveforms
        phi_draw_array:  array of phases of sample waveforms

        amp_eob:  amplitude of EOB waveform
        phi_eob:  phase of EOB waveform

        """
        # a_eob, b_eob = self.linear_fit(phi_eob)
        amp_diff_list = []
        phi_diff_list = []
        for i in range(self.nsamples):
            phi_draw = phi_draw_array[i]
            amp_draw = amp_draw_array[i]
            # a_draw, b_draw = self.linear_fit(phi_draw)
            #phi_diff = (phi_draw - a_draw*self.f_grid - b_draw
            #            - (phi_eob - a_eob*self.f_grid - b_eob))
            phi_diff = phi_draw - phi_eob
            #amp_diff = amp_draw - amp_eob
            amp_diff = np.where(np.all((amp_eob > 0.0, amp_draw > 0.0), axis=0),
                                amp_draw / amp_eob - 1.0,
                                0.0)
            phi_diff_list.append(phi_diff)
            amp_diff_list.append(amp_diff)

        self.amp_array = amp_draw_array
        self.amp_eob = amp_eob
        self.phi_array = phi_draw_array
        self.phi_eob = phi_eob

        self.amp_diff_array = np.array(amp_diff_list)
        self.phi_diff_array = np.array(phi_diff_list)

    def linear_fit(self, phi):
        """Perform a linear fit to function phi."""
        a, b, r_value, p_value, std_err = stats.linregress(self.f_grid, phi)
        return a, b

    def resample_grid(self, x, y, new_x):
        """Resample the array y (defined at x) at new points new_x.  y can be
        a NxM array; function iterates through rows N.

        """
        assert y.ndim == 1 or y.ndim == 2

        if y.ndim == 1:
            yI = spline(x, y)
            return yI(new_x)

        elif y.ndim == 2:
            new_y_list = []
            for i in range(len(y)):
                yI = spline(x, y[i])
                new_y_list.append(yI(new_x))
            return np.array(new_y_list)

    def generate_reduced_representation(self, f_nodes):
        """Generates a representation of the waveform differences (amplitude
        and phase) at a smaller set of frequency nodes.

        Args:
        -----
        f_nodes:  array of frequencies at which to define the waveform
            differences

        """
        if (f_nodes[0] < self.f_min - self.EPSILON or
            f_nodes[-1] > self.f_max + self.EPSILON):
            raise Exception("""Error: new frequency nodes {0}...{1} out
                                of bounds [{2},{3}].""".format(f_nodes[0], f_nodes[-1],
                                                               self.f_min, self.f_max))

        self.nodes = f_nodes
        self.degree = len(f_nodes)
        self.amp_diff_reduced = self.resample_grid(self.f_grid, self.amp_diff_array,
                                                   self.nodes)
        self.phi_diff_reduced = self.resample_grid(self.f_grid, self.phi_diff_array,
                                                   self.nodes)

    def generate_mean_and_cov(self):
        """Combines amplitude and phase of reduced representation into one
        array and compute mean and covariance matrices. Store the
        covariance also as its Cholesky decomposition.

        """
        stacked_diffs = np.hstack((self.amp_diff_reduced,
                                   self.phi_diff_reduced))
        self.mean_diffs = np.mean(stacked_diffs, axis=0)
        self.cov_diffs = np.cov(stacked_diffs, rowvar=False)

        # Express the covariance in terms of its Cholesky
        # decomposition. This quantity is more suitable for
        # interpolating across paramater space because it is better
        # able to enforce positivity.

        self.chol = scipy.linalg.cholesky(self.cov_diffs, lower=True)

    def draw_reduced_sample(self):
        """Sample from the approximate distribution.

        Returns:
            Fractional amplitude and phase deviations, evaluated
            at the frequency nodes.

        """
        reduced_sample = (self.mean_diffs
                          + np.matmul(self.chol,
                                      np.random.normal(size=2*self.degree)))

        amp_sample_reduced = reduced_sample[:self.degree]
        phi_sample_reduced = reduced_sample[self.degree:]

        return amp_sample_reduced, phi_sample_reduced

    def draw_sample(self):
        """Draws a sample waveform difference from the gaussian
        distribution.

        Returns
        -------
        Three lists: frequency nodes, fractional amplitude difference,
        phase difference.
        """
        amp_sample_nodes, phi_sample_nodes = self.draw_reduced_sample()
        return self.nodes, amp_sample_nodes, phi_sample_nodes

    #
    # Plotting tools
    #

    def plot_diffs(self, nsamples=50, save=False, fn="figure"):
        """Plot some waveform diffs"""

        # Randomly select some samples
        n = min(nsamples, len(self.amp_diff_array))
        selection = np.random.choice(list(range(len(self.amp_diff_array))),
                                     size=n, replace=False)
        fig = plt.figure(figsize=(16,6))
        fig.suptitle((r'{0}: $q = {1}$, $\chi_1 = {2}$, $\chi_2 = {3}$; \
                      $M_{{\mathrm{{tot}}}} = {4}$').format(self.cfg, self.q,
                                                            self.chi1, self.chi2,
                                                            self.Mtot))

        mask = np.all((self.f_grid >= self.nodes[0],
                       self.f_grid <= self.nodes[-1]),
                      axis=0)

        ax1 = plt.subplot(121)
        for i in selection:
            plt.plot(self.f_grid[mask], self.phi_diff_array[i,mask], linewidth=0.5)
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta\phi$')
        plt.xscale('log')
        plt.xlim(self.nodes[0], self.nodes[-1])

        ax2 = plt.subplot(122)
        for i in selection:
            plt.plot(self.f_grid[mask], self.amp_diff_array[i,mask], linewidth=0.5)
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta A/A_{\mathrm{EOB}}$')
        plt.xscale('log')
        plt.xlim(self.nodes[0], self.nodes[-1])

        if save:
            plt.savefig(fn)
        plt.show()

    def plot_generated_diffs(self, nsamples=50, save=False, fn='figure'):
        generated_diffs = []
        for i in range(nsamples):
            generated_diffs.append(self.draw_reduced_sample())

        fig = plt.figure(figsize=(16,6))
        fig.suptitle((r'{0}: $q = {1}$, $\chi_1 = {2}$, $\chi_2 = {3}$;\
                      $M_{{\mathrm{{tot}}}} = {4}$').format(self.cfg, self.q,
                                                            self.chi1, self.chi2,
                                                            self.Mtot))

        ax1 = plt.subplot(121)
        for diff in generated_diffs:
            plt.plot(self.nodes, diff[1], linewidth=0.5)
        plt.scatter(self.nodes, self.mean_diffs[self.degree:])
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta\phi$')
        plt.xscale('log')
        plt.xlim(self.nodes[0], self.nodes[-1])

        ax2 = plt.subplot(122)
        for diff in generated_diffs:
            plt.plot(self.nodes, diff[0], linewidth=0.5)
        plt.scatter(self.nodes, self.mean_diffs[:self.degree])
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta A / A_{\textrm{EOB}}$')
        plt.xscale('log')
        plt.xlim(self.nodes[0], self.nodes[-1])

        if save:
            plt.savefig(fn)
        plt.show()

class WaveformUncertaintyInterpolation():
    """Object that interpolates waveform uncertainty across parameter space."""

    SOLAR_MASS_KG = 1.98847e30
    SOLAR_MASS_S = 4.92703806e-6
    MF_MAX_CUTOFF = 0.1

    def __init__(self, *args, **kwargs):
        self.waveforms_present = False
        self.interpolation_present = False

    def generate_interpolation(self, interpolation_degree):
        """Builds the interpolation throughout physical parameter space,
        either polynomial or gpr.

        Args:
            interpolation_degree: The degree of the interpolation.

        """
        wf_uncertainties_train = [self.wf_uncertainties[i]
                                  for i in self.train_selection]

        # Training set parameters
        params_train = np.array([[wfu.q, wfu.chi1, wfu.chi2]
                                 for wfu in wf_uncertainties_train])

        # Training set targets: mean and covariance of distribution, repackaged
        means_train = np.array([wfu.mean_diffs for wfu in wf_uncertainties_train])
        chols_train = np.array([wfu.chol[np.tril_indices(2*self.degree)]
                                for wfu in wf_uncertainties_train])
        y_train = np.hstack((means_train, chols_train))

        if self.method == 'polynomial':
            # Polynomial interpolation
            self.poly = PolynomialFeatures(degree=interpolation_degree)
            params_train_poly = self.poly.fit_transform(params_train)
            self.reg = LinearRegression().fit(params_train_poly, y_train)

        elif self.method == 'gpr':
            # Gaussian process regression
            # Define separate GP for each target in y_train
            self.gp_list = []
            for i, target in enumerate(y_train.T):
                kernel_type = 'squaredexponential'
                hp0 = np.array([0.3, 1, 1, 1, 0.03])
                limits = np.array([[1e-5, 1e1], [1e-2, 20], [1e-2, 2], [1e-2, 2],
                                   [1e-5, 1]])
                gp = generate_gp(params_train, target, hp0,
                                 kernel_type=kernel_type, fixed=False,
                                 hyper_limits=limits)
                print('Hyperparameters ({0}/{1}): {2}'.format(i, len(y_train.T),
                                                              gp.kernel_))
                self.gp_list.append(gp)

    def initialize_reduced_grid(self, degree):
        """Constructs the frequency-grid for the uncertainty model."""
        self.degree = degree

        # Impose a cutoff at a maximum Mf value of MF_MAX_CUTOFF. This
        # is to remove noisy features at high frequencies that we do
        # not want to model.

        Mf_max = self.f_max * self.Mtot * self.SOLAR_MASS_S
        if Mf_max > self.MF_MAX_CUTOFF:
            self.f_max = self.MF_MAX_CUTOFF / (self.Mtot * self.SOLAR_MASS_S)

        self.nodes = np.exp(np.linspace(np.log(self.f_min),
                                        np.log(self.f_max), self.degree))

    def build_interpolation_from_waveform_database(self, filename, degree=10,
                                                   training_frac=0.8,
                                                   interpolation_degree=3,
                                                   method='gpr'):
        """Loads sampled waveforms from database, splits into train and test
        sets, and builds the interpolation throughout parameter space.

        Args:
        -----
        filename: string, hdf5 filename

        degree: integer > 0; Degree of the interpolation. Corresponds
            to number of frequency nodes.

        training_frac: real number between 0.0 and 1.0; Fraction of
            calibration points in physical parameter space with which
            to train the interpolation. Remainder are test points.

        interpolation_degree: integer; For polynomial interpolation,
            the degree of the polynomial.

        method: 'gpr' or 'polynomial'

        """
        if method not in ['gpr', 'polynomial']:
            raise Exception("Method must be 'gpr' or 'polynomial'.")
        self.method = method

        self.db_filename = filename

        f = h5py.File(filename, 'r')
        cfgs = list(f.keys())
        self.Mtot = f.attrs['Mtot']
        self.f_min = f.attrs['f_min']
        self.f_max = f.attrs['f_max']
        f.close()

        self.initialize_reduced_grid(degree)

        # Define WaveformUncertainty instances for each cfg.
        self.wf_uncertainties = []
        for cfg in cfgs:
            self.wf_uncertainties.append(WaveformUncertainty(filename, cfg, self.nodes))

        # Split into training and test sets.
        ntrain = int(round(training_frac*len(cfgs)))
        self.train_selection = np.random.choice(list(range(len(cfgs))),
                                                size=ntrain, replace=False)
        self.test_selection = [i for i in range(len(cfgs))
                               if i not in self.train_selection]

        # Build the interpolation.

        self.generate_interpolation(interpolation_degree)

        self.waveforms_present = True
        self.interpolation_present = True

    def save_interpolation(self):
        """Save interpolation data to file."""

        if self.interpolation_present == False:
            raise Exception('Interpolation must be present to save it.')

        fn = data_dir + r'uncertainty_interpolation_Mtot-{0}_fmin-{1}.hdf5'\
            .format(self.Mtot, self.f_min)
        f = h5py.File(fn, 'w')

        f.attrs['Mtot'] = self.Mtot
        f.attrs['f_min'] = self.f_min
        f.attrs['f_max'] = self.f_max
        f.attrs['degree'] = self.degree
        f.attrs['method'] = self.method
        f.attrs['db_filename'] = self.db_filename

        if self.method == 'gpr':
            gpr_group = f.create_group('gpr')
            # Save the internal data from each gaussian process
            for i, gp in enumerate(self.gp_list):
                group = gpr_group.create_group('gp_' + str(i))
                group.attrs['kernel_type'] = 'squaredexponential'
                group['hyperparameters'] = get_hyperparameters(gp)
                group['points'] = gp.X_train_
                group['data'] = gp.y_train_

        # Save information about training vs test sets.
        f['train_selection'] = self.train_selection
        f['test_selection'] = self.test_selection

        f.close()

    def load_interpolation(self, fn=None, load_wf_db=False):
        """Load interpolation data from a file.

        Parameters
        ----------
        fn : string
            hdf5 filename
        """
        if fn is None:
            # Use data file in this package
            fname = "uncertainty_interpolation_Mtot-50_fmin-20.0.hdf5"
            fn = os.path.join(os.path.dirname(__file__), f"data/{fname}")

        print(f"Attempting to load datafile from {fn}")
        f = h5py.File(fn, 'r')

        self.Mtot = f.attrs['Mtot']
        self.f_min = f.attrs['f_min']
        self.f_max = f.attrs['f_max']
        self.method = f.attrs['method']
        self.db_filename = f.attrs['db_filename']
        degree = f.attrs['degree']

        self.initialize_reduced_grid(degree)
        if self.method == 'gpr':
            gpr_group = f['gpr']
            # Load each gaussian process
            groups = list(gpr_group.keys())
            self.gp_list = []
            for i in range(len(groups)):
                group = gpr_group['gp_' + str(i)]
                kernel_type = group.attrs['kernel_type']
                hp0 = group['hyperparameters']
                points = group['points']
                data = group['data']
                gp = generate_gp(points, data, hp0,
                                 fixed=True, kernel_type=kernel_type)
                self.gp_list.append(gp)

        self.train_selection = f['train_selection'][:]
        self.test_selection = f['test_selection'][:]

        self.interpolation_present = True

        if load_wf_db:
            self.reload_waveform_database()

        f.close()

    def reload_waveform_database(self):
        """Reloads the waveforms from the database file.

        This method is to be used when a WaveformUncertainty object
        has been loaded from a file, and additionally the training and
        test data is desired as well. Generally it would be called by
        the load_interpolation() method.

        """

        if self.interpolation_present == False:
            raise Exception('Requies interpolation to be present.')

        f = h5py.File(self.db_filename, 'r')
        cfgs = list(f.keys())
        f.close()

        # Define WaveformUncertainty instances for each cfg.
        self.wf_uncertainties = []
        for cfg in cfgs:
            self.wf_uncertainties.append(
                WaveformUncertainty(self.db_filename, cfg, self.nodes))

        self.waveforms_present = True

    def predict_mean_chol(self, q, chi1, chi2):
        """Use model to predict mean and cholesky decomposition of covariance
           at a new point in physical parameter space.

        """
        if self.method == 'polynomial':
            params_poly = self.poly.fit_transform([[q, chi1, chi2]])
            y_prediction = self.reg.predict(params_poly)[0]

        elif self.method == 'gpr':
            y_prediction = np.zeros(len(self.gp_list))
            for i, gp in enumerate(self.gp_list):
                y_prediction[i] = gp.predict([[q, chi1, chi2]])[0]

        # Unpack prediction into mean and cholesky decomposition matrix
        mean_prediction = y_prediction[:2*self.degree]
        chol_prediction = np.zeros((2*self.degree, 2*self.degree))
        chol_prediction[np.tril_indices(2*self.degree)] = y_prediction[2*self.degree:]

        return mean_prediction, chol_prediction

    def draw_samples(self, Mtot, q, chi1, chi2, nsamples=1, eps=None):
        """Samples from the predicted distribution for waveform deviation.

        Args:
        -----
        Mtot:  total mass in kg
        q:  float < 1, mass ratio M1/M2 (or M2/M1?)
        chi1:  dimensionless spin of first black hole
        chi2:  dimensionless spin of second black hole

        nsamples:  integer number of samples desired

        eps: (optional) array of sampling parameters. This should have
                shape (nsamples, 2*self.degree). If this is left
                off, draw sampling parameters from normal
                distribution.

        Returns:
        --------
        Three arrays:
        (1) fractional amplitude deviation for each sample
                dimension (nsamples, degree)
        (2) phase deviation for each sample
                dimension (nsamples, degree)
        (3) freqencies at which deviations are evaluated
                dimension (degree)

        """

        if eps is not None:
            if eps.shape != (nsamples, 2*self.degree):
                raise Exception('eps must have shape (nsamples, 2*self.degree)')

        Mtot_solar = Mtot / self.SOLAR_MASS_KG

        # Get predicted mean and Cholesky decomposition (covariance)
        # of uncertainty. This depends on the point in physical
        # parameter space.

        mean, chol = self.predict_mean_chol(q, chi1, chi2)

        amp_sample_list = []
        phi_sample_list = []
        for i in range(nsamples):
            # Sample from distribution defined by mean and Cholesky matrix.
            if eps is None:
                sample_eps = np.random.normal(size=2*self.degree)
            else:
                sample_eps = eps[i]

            sample = mean + np.matmul(chol, sample_eps)

            amp_sample = sample[:self.degree]
            phi_sample = sample[self.degree:]

            amp_sample_list.append(amp_sample)
            phi_sample_list.append(phi_sample)

        # Compute the shifted frequency nodes. This depends on the
        # ratio of the total mass of the desired waveform to the total
        # mass of the modeled waveform.

        shifted_frequency_nodes = self.nodes * self.Mtot / Mtot_solar

        return (np.array(amp_sample_list), np.array(phi_sample_list),
                shifted_frequency_nodes)

    def draw_sample(self, Mtot, q, chi1, chi2, eps=None):
        """Draws a single sample from the predicted distribution for waveform
        deviation.

        Calls draw_samples with n=1 and repackages result in simpler form.

        Args:
        -----
        Mtot:  total mass in kg
        q:  float < 1, mass ratio M1/M2 (or M2/M1?)
        chi1:  dimensionless spin of first black hole
        chi2:  dimensionless spin of second black hole

        eps: (optional) array of sampling parameters. This should have
                dimension 2*self.degree. If this is left
                off, draw sampling parameters from normal
                distribution.

        Returns:
        --------
        Three arrays:
        (1) fractional amplitude deviation
        (2) phase deviation
        (3) freqencies at which deviations are evaluated

        """
        if eps is not None:
            amp, phi, freq = self.draw_samples(Mtot, q, chi1, chi2, nsamples=1,
                                               eps=np.expand_dims(eps, axis=0))
        else:
            amp, phi, freq = self.draw_samples(Mtot, q, chi1, chi2, nsamples=1)

        return amp[0], phi[0], freq

    def plot_comparison(self, n=1, train=False):
        """Plot generated and true waveform differences at n points."""

        # Compare at either test or train points
        if not train:
            selection = np.random.choice(self.test_selection,
                                         min(len(self.test_selection), n),
                                         replace=False)
        else:
            selection = np.random.choice(self.train_selection,
                                         min(len(self.train_selection), n),
                                         replace=False)
        wf_uncertainties_selection = [self.wf_uncertainties[i] for i in selection]

        for wfu in wf_uncertainties_selection:
            wfu.plot_diffs()
            self.plot_sampled_diffs(wfu.Mtot * self.SOLAR_MASS_KG,
                                    wfu.q, wfu.chi1, wfu.chi2)
            self.plot_aux_comparison(wfu)

    def plot_sampled_diffs(self, Mtot, q, chi1, chi2, nsamples=50,
                           save=False, fn='figure'):
        """Draw samples from predicted distribution and plot them."""

        # Draw samples
        amp_samples, phi_samples, f_grid = self.draw_samples(Mtot, q,
                                                             chi1, chi2, nsamples)

        # Also get the mean prediction (for plotting)
        mean, chol = self.predict_mean_chol(q, chi1, chi2)

        fig = plt.figure(figsize=(16,6))
        fig.suptitle((r'Predicted waveform differences: $q = {0}$, $\chi_1 = {1}$, $\chi_2 = {2}$; $M_{{\mathrm{{tot}}}} = {3}$').format(q, chi1, chi2, Mtot))

        ax1 = plt.subplot(121)
        for i in range(len(phi_samples)):
            plt.plot(f_grid, phi_samples[i], linewidth=0.5)
        plt.scatter(f_grid, mean[self.degree:])
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta\phi$')
        plt.xscale('log')
        plt.xlim(f_grid[0], f_grid[-1])

        ax2 = plt.subplot(122)
        for i in range(len(amp_samples)):
            plt.plot(f_grid, amp_samples[i], linewidth=0.5)
        plt.scatter(f_grid, mean[:self.degree])
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta A / A_\mathrm{EOB}$')
        plt.xscale('log')
        plt.xlim(f_grid[0], f_grid[-1])

        if save:
            plt.savefig(fn)
        plt.show()

    def plot_aux_comparison(self, wfu):
        """At a calibration point, compare the predicted and actual mean
        values of the distribution.

        Args:
        -----
        wfu:  WaveformUncertainty object for which to perform comparison.

        """

        mean_predicted, chol_predicted = self.predict_mean_chol(wfu.q, wfu.chi1, wfu.chi2)
        mean_actual = wfu.mean_diffs

        fig = plt.figure(figsize=(16,6))
        fig.suptitle(('Means comparison (predicted vs actual)'))

        ax1 = plt.subplot(121)
        plt.scatter(self.nodes, mean_predicted[self.degree:], label='predicted')
        plt.scatter(self.nodes, mean_actual[self.degree:], label='actual')
        plt.legend()
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta\phi$')
        plt.xscale('log')
        plt.xlim(self.nodes[0], self.nodes[-1])

        ax2 = plt.subplot(122)
        plt.scatter(self.nodes, mean_predicted[:self.degree], label='predicted')
        plt.scatter(self.nodes, mean_actual[:self.degree], label='actual')
        plt.legend()
        plt.xlabel(r'$f$')
        plt.ylabel(r'$\Delta A$')
        plt.xscale('log')
        plt.xlim(self.nodes[0], self.nodes[-1])

        plt.show()

def get_hyperparameters(gp):
    """Get the hyperparameters of the GaussianProcessRegressor gp.
    The order is (sigma_f, ls_0, ls_1, ..., sigma_n).
    """

    # kernel is what is used to initialize GaussianProcessRegressor
    # kernel_ is the kernel after applying gp.fit(points, data)
    # The gp stores the hyperparameters as theta = ln(hyper_pramas)
    hp = np.exp(gp.kernel_.theta)
    # The scale and noise terms in gp are sigma_f^2 and sigma_n^2.
    # You want sigma_f and sigma_n.
    hp[0] = hp[0]**0.5
    hp[-1] = hp[-1]**0.5
    return hp

def generate_gp(points, data, hp0, kernel_type='squaredexponential',
                fixed=False, hyper_limits=None, n_restarts_optimizer=9):
    """Gaussian Process for ndim dimensional parameter space.

    Parameters
    ----------
    points : array of shape (npoints, ndim).
        Coordinates in paramete space of sampled data.
    data : array of shape (npoints,).
        Data at each of the sampled points.
    hp0 : array of shape (ndim+2,)
        Initial hyperparameter guess for optimizer.
        Order is (sigma_f, ls_0, ls_1, ..., sigma_n).
    kernel_type : 'squaredexponential', 'matern32', 'matern52'
    limits : array of shape (ndim+2, 2)
        Lower and upper bounds on the value of each hyperparameter.
    n_restarts_optimizer : int
        Number of random points in the hyperparameter space to restart optimization
        routine for searching for the maximum log-likelihood.
        Total number of optimizations will be n_restarts_optimizer+1.

    Returns
    -------
    gp : GaussianProcessRegressor
    """

    # ******* Generate kernel *******

    # ConstantKernel = c multiplies *all* elements of kernel matrix by c
    # If you want to specify sigma_f (where c=sigma_f^2) then use
    # sigma_f^2 and bounds (sigma_flow^2, sigma_fhigh^2)

    # WhiteKernel = c \delta_{ij} multiplies *diagonal* elements by c
    # If you want to specify sigma_n (where c=sigma_n^2) then use
    # sigma_n^2 and bounds (sigma_nlow^2, sigma_nhigh^2)

    # radial part uses the length scales [l_0, l_1, ...] not [l_0^2, l_1^2, ...]

    # Constant and noise term
    if fixed==True:
        const = ConstantKernel(hp0[0]**2)
        noise = WhiteKernel(hp0[-1]**2)
    elif fixed==False:
        const = ConstantKernel(hp0[0]**2, hyper_limits[0]**2)
        noise = WhiteKernel(hp0[-1]**2, hyper_limits[-1]**2)
    else:
        raise Exception("'fixed' must be True or False.")

    # Radial term
    if fixed==True:
        if kernel_type=='squaredexponential':
            radial = RBF(hp0[1:-1])
        elif kernel_type=='matern32':
            radial = Matern(hp0[1:-1], nu=1.5)
        elif kernel_type=='matern52':
            radial = Matern(hp0[1:-1], nu=2.5)
        else: raise Exception("Options for kernel_type are: 'squaredexponential', 'matern32', 'matern52'.")
    elif fixed==False:
        if kernel_type=='squaredexponential':
            radial = RBF(hp0[1:-1], hyper_limits[1:-1])
        elif kernel_type=='matern32':
            radial = Matern(hp0[1:-1], hyper_limits[1:-1], nu=1.5)
        elif kernel_type=='matern52':
            radial = Matern(hp0[1:-1], hyper_limits[1:-1], nu=2.5)
        else: raise Exception("Options for kernel_type are: 'squaredexponential', 'matern32', 'matern52'.")
    else:
        raise Exception("'fixed' must be True or False.")

    kernel = const * radial + noise

    # ******* Initialize GaussianProcessRegressor and optimize hyperparameters if not fixed *******

    if fixed==True:
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                      normalize_y=True,
                                      copy_X_train=True)
        # Supply the points and data, but don't optimize the hyperparameters
        gp.fit(points, data)
        return gp
    elif fixed==False:
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=n_restarts_optimizer,
                                      normalize_y=True,
                                      copy_X_train=True)
        # Optimize the hyperparameters by maximizing the log-likelihood
        gp.fit(points, data)
        return gp
    else:
        raise Exception("'fixed' must be True or False.")
