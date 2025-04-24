This repository contains the code used to generate results shown in [Russeil *et al.*, 2024,  Multi-View Symbolic Regression](https://arxiv.org/abs/2402.04298). **A modern and user friendly implementation of MvSR is now available here : [https://github.com/folivetti/pyeggp]. See pyeggp/test/mvsr_example.ipynb**

## Content

Files:  

- `mvsr.py`: MvSR basic implementation.
- `generate_data.py`: generates the artificial benchmark dataset
- `analysis.py`: Run SR/MvSR on artificial benchmark. Allow to refit and evaluate.
- `run_all.sh`: Run the main analysis for every setup presented in the paper.
- `results.py`: Read results and aggregate them into a table.
- `plots.py`: Generate plots from the aggregated table.


Folders:  

- `real_data` : The real datasets from chemistry, finance and astrophysics
- For each data set, we provide a notebook with the specific setups used to generate the parametric functions presented in the paper.

To run the code, in addition to the dependencies listed in the  `requirement.txt` file, it requires two additional setups.  


## Installation instructions

This step-by-step will ensure the installation of the correct version of the adapted Operon:

- Install PyEnv (https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)
- Run these commands:

```bash
pyenv install 3.11
pyenv shell 3.11
python3.11 -m venv mvsrenv
source mvsrenv/bin/activate
pip3.11 install wheels/pyoperon-0.3.6-cp311-cp311-linux_x86_64.whl
pip3.11 install -r requirement.txt
```
- Edit the file `mvsrenv/lib/python3.11/site-packages/iminuit/cost.py` and replace
   * line 1827 with `self._ndim = x.shape[0]`
   * line 1836 with `x = self._masked.T[: self._ndim]`
