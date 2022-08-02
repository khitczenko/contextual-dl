This directory contains three subdirectories:

1. scripts: 

- calculateEMS.py: runs all of the analyses included in the paper (except it does not make the graphs) and outputs the results to csv files. These csv files are included in the results subdirectory.
- makeBoxPlots.py: makes the boxplot graphs in the paper (i.e., Figure 2, Figure 4, Figure 5, Figure 6, Figure S2). This script takes as input the result csv files that calculateEMS.py generates (included in results subdirectory)
- makeLinePlots.py: makes the lineplot graphs in the paper (i.e., Figure 3, Figure S1). This script takes as input the result csv files that calculateEMS.py generates (included in results subdirectory). Note that the measure used (mean/maximum) can be changed on l.26.

2. results: This subdirectory contains the outputs of calculateEMS from the paper (each file is generated using different settings, e.g. how context is defined, how many iterations to run, etc.). These files are used in scripts/makeBoxPlots.py and scripts/makeLinePlots.py. The file naming convention is the following:

ContextType_DurationRepresentation_nIter=X_rank=X_Measure_ResultPresentationType_DataSet.csv

Each of these is explained in more detail below:

ContextType:
- combo: P+NS+VQ
- wf: Word Frame

DurationRepresentation:
- Duration: absolute duration (i.e., as opposed to z-scored duration)

nIter=X:
- nIter=1: This is used to generate boxplots (i.e., we just run it once)
- nIter=50: This is used to generate lineplots (i.e., we average across 50 runs)

rank=X: 
- This tells us which contexts were included in the analysis. For example, if rank=200 this means that the top 200 most frequent contexts were included.

Measure:
- em: Earthmover's distance
- kl: KL Divergence

ResultPresentationType:
- contrast: Each row corresponds to the Earthmover's distance calculated between one pair of contexts in the contrastive case. This is output when nIter = 1.
- nocontrast: Each row corresponds to the Earthmover's distance calculated between one pair of contexts in the non-contrastive case. This is output when nIter = 1.
- nolongvowels: Each row corresponds to the Earthmover's distance calculated between one pair of contexts in the contrastive case (but where long vowels are removed). This is output when nIter = 1.
- summary: Calculates summary statistics that compare the contrastive and non-contrastive findings in XXX_contrast.csv and XXX_nocontrast.csv. Each row corresponds to one iteration of one corpus size (either contrastive or non-contrastive). It reports summary statistics of that run, which are used for making the lineplots. Note that this file is not output when long vowels are removed.

3. data/: This subdirectory contains an example of the datafile headers used as input to calculateEMs. The data we use for the paper come from previously-collected corpora. They are available via request from the researchers who control their distribution. Feel free to get in touch about this!