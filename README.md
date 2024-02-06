This Git-Hub presents the code used to generate results from the "Multi-View Symbolic Regression" paper.
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
