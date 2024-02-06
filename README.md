This Git-Hub presents the code used to generate results from the "Multi-View Symbolic Regression" paper.
The artificial benchmark dataset can be generated using the generate_data.py file. The real datasets from chemistry, finance and astrophysics are available in the real_data folder.  
For each one we provide a notebook with the specific setups used to generate the parametric functions presented in the paper.


In addition to the requirement.txt file it requires two additional setup.

## First requirement
The iminuit version we use has been slightly modify to fix an occuring error from version 2.24. The cost.py file was modified so that:

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
