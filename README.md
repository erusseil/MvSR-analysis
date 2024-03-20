This repository contains the code used to generate results shown in [Russeil *et al.*, 2024,  Multi-View Symbolic Regression](https://arxiv.org/abs/2402.04298).  

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

## First requirement

The iminuit version we use has been modify to fix an occuring error from version 2.24. The `cost.py` file was modified so that:

Line 1827 becomes 
```bash
self._ndim = x.shape[0]
```

And line 1836 becomes
```bash
 x = self._masked.T[: self._ndim]
```

## Second requirement  

The pyoperon version used includes some adaptations to make MvSR possible. Therefore the specific branch: https://github.com/heal-research/pyoperon/tree/cpp17 should be installed.
Therefore the following wheel should be used : https://github.com/heal-research/pyoperon/releases
After that you can just 
```bash
pip install <wheel-filename>
```
