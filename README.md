# SEOBNRv4CE

Implementation of the **SEOBNRv4** model incorporating **waveform uncertainty** for gravitational wave analysis. This repository provides tools for waveform generation with wavefrom uncertainty, implimentation of the model in data analysis framework.

## SEOBNRv4_ROM base model in LALSuite

SEOBNRv4_ROM acts as a base model for the uncertainty model and is implemented in LALSuite.
For this model to work the environment variable `LAL_DATA_PATH` needs to point to a directory which contains the ROM data file:
  * For LALSuite versions < 7.25 the file is SEOBNRv4ROM_v2.0.hdf5 and can be found
in https://git.ligo.org/lscsoft/lalsuite-extra.
  * For LALSuite versions >= 7.25 the file is `SEOBNRv4ROM_v3.0.hdf5` and can be
found in https://git.ligo.org/waveforms/software/lalsuite-waveform-data.
