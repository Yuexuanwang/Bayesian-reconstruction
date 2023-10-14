# Bayesian-reconstruction
This repository consists of the algorithm that reconstructs the design and parameter matrices jointly by MCMC sampling. 

In this repository, the code of the MCMC algorithm and all the simulated data is contained. For the real dataset, only the third patient with vpu region and the region 14924777-15216613 bp in Caenorhabditis elegans are available, the whole data are available in the original paper ([1], [2]).

-The 'reconstruct.py' is the python code. In this file, 
-'Gibbs' is the function of the MCMC algorithm, which require the observed matrix, inner dimension parameter, the path to save the output, number of burn-in steps, and number of iterations after burn-in.
-The 'ToyExample' folder contains all the simulated datasets. 
  -'m3', 'm4' and 'm5' are the experiences with inner dimension index 3,4 and 5 respectively. 
  -And the folder inside, for example, 'p5_m3_T10_N100_sigma0.05' the data is simulated with p=0.5 for the Bernoulli distribution for S, the observed matrix has the row number N=100, column number T=10 and under the noise level sigma=0.05. 
  -'toytest_cor' is the dataset simulated by the Drosophila data, in folder 'p0.5_T16_s0.05_m2_m04_N100', we replaced 25%, 50%, and 75% of SNPs of the data with 0.05 noise level to investigate the performance of our estimator under the range of correlation.
-The 'RealData' contains the region of the dataset we tested in our paper. 

###Example of how to run the code:
In order to employ our code, please execute it through the command line, with the following code:

python3 gau_error.py --Y [path_to_Y] --m [positive integer] --N_samples [positive integer] --N_burnin [positive integer]

--Y [path_to_Y]: Specifies the path of txt file contains Y, representing the observed allele frequencies matrix.
--m [positive integer]: Denotes the haplotype_number, need to be smaller than column number of Y and 2^m should be smaller than row number of Y.
--N_samples [positive integer]: Indicates the desired number of MCMC samples in the output sequence.
--N_burnin [positive integer]: Designates the Burn-in number.

[1]. F. Zanini, J. Brodin, L. Thebo, C. Lanz, G. Bratt, J. Albert, R. A. Neher, Population genomics of intrapatient hiv-1 evolution, Elife 4 (2015)
e11282.
[2]. L. M. Noble, M. V. Rockman, H. Teot ́onio, Gene-level quantitative trait mapping in caenorhabditis elegans, G3 11(2)(2021)jkaa061.
