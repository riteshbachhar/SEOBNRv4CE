# SEOBNRv4CE

Implementation of the **SEOBNRv4** model incorporating **waveform uncertainty** for gravitational wave analysis. This repository provides tools for waveform generation with wavefrom uncertainty, and an implementation of the model for data analysis with the [Bilby](https://lscsoft.docs.ligo.org/bilby/) Bayesian inference code.

## Installation

Either run

```sh
pip install git+https://github.com/riteshbachhar/SEOBNRv4CE.git@main
```

or, to install from a git checkout in directory "SEOBNRv4CE", create a [venv](https://docs.python.org/3/library/venv.html) and install from your checkout with pip:

```sh
python -m venv seobnrv4ce-venv
source seobnrv4ce-venv/bin/activate
cd SEOBNRv4CE
pip install .
```

## SEOBNRv4_ROM base model in LALSuite

SEOBNRv4_ROM acts as a base model for the uncertainty model and is implemented in LALSuite, specifically in [LALSimulation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/).
For this model to work the environment variable `LAL_DATA_PATH` needs to point to a directory which contains the ROM data file:
  * For LALSuite versions < 7.25 the file is `SEOBNRv4ROM_v2.0.hdf5` and can be found
in https://git.ligo.org/lscsoft/lalsuite-extra.
  * For LALSuite versions >= 7.25 the file is `SEOBNRv4ROM_v3.0.hdf5` and can be
found in https://git.ligo.org/waveforms/software/lalsuite-waveform-data.

## Usage

For an example of how to use the uncertainty model please see the [tutorial
notebook](https://github.com/riteshbachhar/SEOBNRv4CE/blob/main/tutorial/SEOBNRv4CE.ipynb).
To run parameter estimation simulations, please use
[bilby_analyze_injection.py](https://github.com/riteshbachhar/SEOBNRv4CE/blob/main/seobnrv4ce/scripts/bilby_analyze_injection.py)
and see the example Slurm script
[run_example_slurm.sh](https://github.com/riteshbachhar/SEOBNRv4CE/blob/main/seobnrv4ce/scripts/run_example_slurm.sh).

## References

SEOBNRv4CE is based on the paper https://arxiv.org/abs/2410.17168.
