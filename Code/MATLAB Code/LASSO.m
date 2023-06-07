% Code used to run LASSO regression on fatty acid/MFG data
% Cameron Jakub
%
% NOTE:
% This code uses cvglmnet function found in the glmnet package for MATLAB
%
% J. Qian, T. Hastie, J. Friedman, R. Tibshirani and N. Simon. Glmnet for
% matlab, 2013.
% URL: https://web.stanford.edu/~hastie/glmnet_matlab/

% Add glmnet package to MATLAB search path
addpath('/~path~/glmnet_matlab');

% Load data
data = dlmread('/~path~/data.csv', ',', 1, 0);

% Set string for saving results
str = 'FA';

% Divide data into predictor matrix X and response vector y
X = data(:,2:end);
y = data(:,1);

% Standardize spectroscopy data to have columnwise mean of 0 and standard
% deviation of 1
X = zscore(X);

% Find size of input matrix
n = size(X,1);

% Set seed for reproducibilty
rng(100);

% Allison first performed PLSR using 30 components and used the results to
% determine outliers in the data. Observations with RMSE > 3 standard 
% deviations away from the mean were removed.

% First perform PLSR using 30 components
[XL, YL, XS, YS, betaCoef, pctVar, plsMSE, stats] = plsregress(X, y, 30);

% Create vector of fitted values
yFit = [ones(n,1) X]*betaCoef;

% Create vector of root squared residuals
Res = (yFit - y).^2;
Res = sqrt(Res);

% Find mean and standard deviation of squared residuals
MeanRes = mean(Res);
StdRes = std(Res);

% Create threshold value to determine which entries are outliers
% (3 standard deviations above the mean)
Thrsh1 = MeanRes + 3*StdRes;
Thrsh2 = MeanRes - 3*StdRes;

% Find indices at which samples are deemed outliers
index1 = find(Res > Thrsh1);
index2 = find(Res < Thrsh2);

% Remove these samples from our training data
if isempty(index1) ~= 1
    X(index1,:) = [];
    y(index1) = [];
end

if isempty(index2) ~= 1
    X(index2,:) = [];
    y(index2) = [];
end

% Find new size of our data
n = size(X,1);

% Fit a LASSO model to our data using 10-fold cross validation
fit = cvglmnet(X, y, [], [], 'mse', 10);

% Find index of our optimal model
idx = find(fit.lambda == fit.lambda_min);

% Create vector of beta coefficients corresponding to our best model
betaCoef = [fit.glmnet_fit.a0(idx); fit.glmnet_fit.beta(:,idx)];

% Find the MSEP corresponding to the optimal lambda value
idx = find(fit.lambda == fit.lambda_min);
MSEP = fit.cvm(idx);

% Calculate R2CV value for our model
SStot = sum((y - mean(y)).^2);
SSEP = n*MSEP;
R2CV = 1 - SSEP/SStot;

% Find number of nonzero coefficients in our model
nCoef = fit.nzero(idx);

% Print results and save to a .txt file
sprintf('LASSO\n%s\nR2cv: %g\n# nonzero coef: %g\nLambda: %g', str, R2CV, nCoef, fit.lambda_min)

txtStr = strcat(str, ".txt");
fid = fopen(txtStr, 'wt');
fprintf(fid, 'LASSO\n%s\nR2cv: %g\n# nonzero coef: %g\nLambda: %g', str, R2CV, nCoef, fit.lambda_min);
fclose(fid);







