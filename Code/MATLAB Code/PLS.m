% Code used to run partial least squares regression on fatty acid/MFG data
% Cameron Jakub

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
% (3 standard deviations above/below the mean)
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

% Now that we have removed outliers, we will perform a PLS regression with
% 100 components and leave-one-out cross validation
[XL, YL, XS, YS, betaCoef, pctVar, plsMSEP, stats] = plsregress(X, y, 100, 'CV', n);

% Find minimum MSEP & its corresponding index
minMSEP = min(plsMSEP(2,:));
idxMin = find(plsMSEP(2,:) == minMSEP) -  1;

% Calculate fitted values of our model
Fit = [ones(n,1) X]*betaCoef;

% Calculate R2CV for our model
SStot = sum((y - mean(y)).^2);
SSEP = n*minMSEP;
R2CV = 1 - SSEP/SStot;

% Print results and save to a .txt file
sprintf("PLSR\n%s\nR2: %g\nnComp: %g\nN: %g", str, R2CV, idxMin, n)

txtStr = strcat(str, ".txt");

fid = fopen(txtStr, 'wt');
fprintf(fid, "PLSR\n%s\nR2: %g\nnComp: %g\nN: %g", str, R2CV, idxMin, n);
fclose(fid);





