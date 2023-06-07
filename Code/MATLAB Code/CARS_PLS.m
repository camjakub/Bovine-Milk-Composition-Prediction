% Code used to run CARS-PLS regression on fatty acid/MFG data
% Cameron Jakub
%
% NOTE:
% This code uses carpls function found in the libPLS 1.98 package for MATLAB
%
% H.-D. Li, Q.-S. Xu, and Y.-Z. Liang. libpls: an integrated library for
% partial least squares regression and discriminant analysis. Chemom. 
% Intell. Lab. Syst., 176:34-43, 2018.
%
% https://www.libpls.net/download.php

% Add libPLS package to MATLAB search path
addpath('/~path~/libPLS_1.98');

% Set string for saving results
str = 'FA';

% Load data
data = dlmread('/~path~/data.csv', ',', 1, 0);

% Divide data into predictor matrix X and response vector y
X = data(:,2:end);
y = data(:,1);

R2CV = [];
nComp = [];
nVar = [];

% Standardize spectroscopy data to have columnwise mean of 0 and standard
% deviation of 1
X = zscore(X);

n = size(X,1);

% Allison first performed PLSR using 30 components and used the results to
% determine outliers in the data. Observations with RMSE > 3 standard 
% deviations away from the mean were removed.

% First perform PLSR using 30 components
[XL, YL, XS, YS, betaCoef, pctVar, plsMSE, stats] = plsregress(X, y, 30);

% First, create vector of fitted values
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


for j = 1:25
    
    disp(j)

    % Set seed for reproducibility, increase value by 1 for each CARS 
    % sampling run
    seed = 99+j;
    rng(seed);
    
    % Fit a CARS-PLS model to the data using 50 Monte Carlo sampling runs
    % and 10-fold cross validation
    CARS=carspls(X,y,50,10,'center',50,0,1,2);
    
    % Store results in vector
    R2CV = [R2CV CARS.Q2_max];
    nComp = [nComp CARS.optLV];
    nVar = [nVar size(CARS.vsel,2)];

end

% Take average of results
R2CV = mean(R2CV);
nComp = mean(nComp);
nVar = mean(nVar);

% Print results and save to a .txt file
sprintf('CARS-PLS\n%s\nN: %g\nR2CV: %g\nnComp: %g\nnVar: %g', str, n, R2CV, nComp, nVar)

txtStr = strcat(str, ".txt");
fid = fopen(txtStr, 'wt');
fprintf(fid, 'CARS-PLS\n%s\nN: %g\nR2CV: %g\nnComp: %g\nnVar: %g', str, n, R2CV, nComp, nVar);
fclose(fid);






