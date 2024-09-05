<a name="top"></a>

[![Python package](https://github.com/remsens-lim/pyMakeRetrieval/actions/workflows/python-package.yml/badge.svg)](https://github.com/remsens-lim/pyMakeRetrieval/actions/workflows/python-package.yml)
[![Pylint](https://github.com/remsens-lim/pyMakeRetrieval/actions/workflows/pylint.yml/badge.svg)](https://github.com/remsens-lim/pyMakeRetrieval/actions/workflows/pylint.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10014291.svg)](https://doi.org/10.5281/zenodo.10014291)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Github all releases](https://img.shields.io/github/downloads/Naereen/StrapDown.js/total.svg)](https://github.com/remsens-lim/pyMakeRetrieval/releases/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/remsens-lim/pyMakeRetrieval/graphs/commit-activity)
[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/RSAtmos_LIM.svg?style=social&label=Follow%20%40RSAtmos_LIM)](https://twitter.com/RSAtmos_LIM)
![Mastodon Follow](https://img.shields.io/mastodon/follow/109461236453474330?domain=https%3A%2F%2Fmeteo.social&logoColor=%230066cc&style=social)

<!-- [![Release][release-shield]][release-url] -->
<!-- [![PyPi version](https://badgen.net/pypi/v/pip/)](https://pypi.com/project/pip) -->

<!-- [![Twitter](https://img.shields.io/twitter/follow/RSAtmos_LIM?style=for-the-badge)](https://twitter.com/RSAtmos_LIM) -->

# pyMakeRetrieval
Generates microwave radiometer retrievals for LWP, IWV, TPT, HPT, TPB, TBX based on radiative transfer calculation data  


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Introduction">Introduction</a></li>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <!-- <li><a href="#contributing">Contributing</a></li> -->
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- Introduction -->
## Introduction

This repository was created for generating microwave radiometer retrievals for LWP, IWV, TPT, HPT, TPB, TBX based on radiative transfer calculation data. The default settings in the config files are set for the microwave radiometer HATPRO from the german manufacturer Radiometer Physics GmbH. 

<!-- Installation -->
## Installation

Below is an example of how run the script, which reads in the radiative transfer data and makes the retrievals. This method relies on external dependencies such as xarray, numpy and others (see `setup.py`). The output are nc files which contains the coefficients and plots about the retrieval performance based on test data.

1. install from github
   ```sh
   git clone https://github.com/remsens-lim/pyMakeRetrieval.git
   cd pyMakeRetrieval
   python3 -m venv venv
   source venv/bin/activate
   pip3 install --upgrade pip
   pip3 install .
   ```

<p text-align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. set your specifications in the general_config file (e.g. paths and site specifics) and set your retrieval specifications in the retrieval config files. Use config_tpb.yaml and the ret_specs.yaml for tpb retrievals.
2. run the routine
   ```sh
   py_make_retrieval/cli.py --ret RETRIEVAL
   ```
`RETRIEVAL`
- `iwv`: integrated water vapour
- `lwp`: liquid water path
- `tbx`: spectrum retrievals for spectral consistency checks
- `hpt`: absolute humidity profiles from single angle observations
- `tpt`: temperature profiles from single angle observations
- `tpb`: temperature profiles from multi angle observations (elevation scans)
- `all`: makes all of the retrievals above


[//]: # (<img src="eval_ac/results_ln2_cal.png" width="70%">)

<p text-align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] add meaningful docstrings
- [ ] make documentation --> readthedocs
- [ ] enable pip install ...
- [ ] Add Tests

See the [open issues](https://github.com/remsens-lim/pyMakeRetrieval/issues) for a full list of proposed features (and known issues).

<p text-align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p text-align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

[Andreas Foth](https://www.uni-leipzig.de/personenprofil/mitarbeiter/dr-andreas-foth)


<p text-align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Special thanks for templates and help during implementation.

* [Readme Template](https://github.com/othneildrew/Best-README-Template)
* [cloudnetpy GitHub](https://github.com/actris-cloudnet/cloudnetpy.git)
* [mwrpy GitHub](https://github.com/actris-cloudnet/mwrpy.git)
* [Jonas Witthuhn](https://github.com/jonas-witthuhn)

<p text-align="right">(<a href="#top">back to top</a>)</p>
