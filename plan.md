# Plan

## Likelihood
- use xnli dataset (15 languages, including english)
- compute likelihood using mbert
- compute fourier spectra for all languages per sample
- calculate overlap for all pairs per sample (maybe use KL div)
- average overlap for all samples per pair
- compare to FSI categories
