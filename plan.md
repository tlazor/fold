# Plan

## Likelihood
- use xnli dataset (15 languages, including english)
- compute likelihood using mbert
- z score normalize likelihood
- compute fourier spectra for all languages per sample (maybe performe test to see if likelihood is stationary)
- calculate overlap for all pairs per sample (maybe use KL div)
- average overlap for all samples per pair
- compare to FSI categories
