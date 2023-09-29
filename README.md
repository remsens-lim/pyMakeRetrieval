<a name="top"></a>

[//]: # ([![Python package]&#40;https://github.com/remsens-lim/pyMakeRetrieval/actions/workflows/python-package.yml/badge.svg&#41;]&#40;https://github.com/WillyWallace/pyMakeRetrieval/actions/workflows/python-package.yml&#41;)
[//]: # ([![Pylint]&#40;https://github.com/remsens-lim/pyMakeRetrieval/actions/workflows/pylint.yml/badge.svg&#41;]&#40;https://github.com/remsens-lim/pyMakeRetrieval/actions/workflows/pylint.yml&#41;)
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
    <li><a href="#getting-started">Getting Started</a></li>
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

<!-- GETTING STARTED -->
## Getting Started

<!-- Installation -->
## Installation

Below is an example of how run the script, which reads in the data and plots the results. This method relies on external dependencies such as xarray, numpy and others (see `setup.py`).

1. Clone the repo
   ```sh
   git clone https://github.com/WillyWallace/eval_ac.git
   ```

2. Install the package
   ```sh
   python setup.py install
   ```

<p text-align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. set your specifications (e.g. paths of the ABSCAL.his file) in the user specifications section at the top within the evaluate_absolute_calibration.py routine 
2. run the evaluate_absolute_calibration.py routine
3. see the results in the results_ln2_cal.png image

[//]: # (<img src="eval_ac/results_ln2_cal.png" width="70%">)

<p text-align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] add meaningful docstrings
- [ ] make documentation --> readthedocs
- [ ] enable pip install ...
- [ ] Released version 1
- [ ] Add Tests

See the [open issues](https://github.com/WillyWallace/eval_ac/issues) for a full list of proposed features (and known issues).

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

<p text-align="right">(<a href="#top">back to top</a>)</p>
