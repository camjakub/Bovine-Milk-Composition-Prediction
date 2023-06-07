# Prediction and Heritability Estimation of Bovine Milk Composition Using Mid-Infrared Spectroscopy
This is a project I worked on in summer 2019 under the supervision of Dr. Ayesha Ali (Professor of Statistics, University of Guelph). We extended the work of a previous PhD student who had used mid-infrared spectroscopy data on bovine milk samples to predict the composition of the milk (particularly fatty acid contents and milk fat globule size). Fatty acid content and fat globule size are factors which determine the overall quality of the milk, but traditional methods for measuring these quantities are slow and expensive. Spectroscopy data is relatively quick and cheap to obtain, so using spectroscopy to accurately predict the milk composition could save time and money.

The original model developed for predicting milk composition was done using partial least squares regression, but my advisor believed we could achieve better predictive power using more complicated regression methods. My job was to implement various regression methods using R and MATLAB, and compare and contrast them to partial least squares regression models which were originally created. (Note: this repo only contains the MATLAB files, since that is the language we ultimately decided to use. We originally ran these regressions in R, but switched to MATLAB halfway through). We also created univariate animal models to predict the heritability of the milk composition. The final report for this project is included in the repo as a pdf.

## Code Description

### PLS.m, LASSO.m, CARS_PLS.m
These 3 files perform Partial Least Squares, LASSO, and Competitive Adaptive Reweighted Sampling-Partial Least Squares regression, respectively. These files were used to run the regressions on the Fat, Milk, and Ln datasets. Data used in this code was saved as a .csv and had the response variable (fatty acid/MFG size) as the first column followed by 862 columns of PINS 240-410, 440-790, and 960-1299. The first row of the data files used were column labels and are therefore skipped during the dlmread command when loading the data. Outlier screening involving fitting the data to a PLS model with 30 components and removing samples with RMSE greater than 3 standard deviations away from the mean is implemented before models are fit to the data.

Results are printed to the screen and also saved to a .txt file with name specified by the "str" variable.

### PLS_subset.m, LASSO_subset.m, CARS_PLS_subset.m 
These 3 files also perform the 3 regression methods, but they implement a subset selection method to the data to create a more uniform distribution of the response variable prior to performing the regression. 10 different subsets are created and all results are averaged.

Results are printed to the screen and also saved to a .txt file with name specified by the "str" variable.



### rr_ai.DIR 
This is the driver file used to run univariate animal models. The term “FAdata” on line 6 specifies the filename of the data file being used to run the model. The term “pedigree” on line 22 specifies the filename of the pedigree file used to run the model. The columns of “FAdata” were organized in this order: animid, DIM, htest, hcalving, fa/mfg

where:
* animid - animal id number (integer)
* DIM - days in milk class (integer)
* test - herd-test day class (integer)
* hcalving - herd-calving age class (integer)
* fa/mfg - predicted fatty acid/mfg content (real number)
