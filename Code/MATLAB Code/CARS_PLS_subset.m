% Code used to run CARS-PLS regression on fatty acid/MFG data
% Cameron Jakub
%
% NOTE:
% This code uses carpls function found in the libPLS 1.98 package for MATLAB
%
% H.-D. Li, Q.-S. Xu, and Y.-Z. Liang. libpls: an integrated library for
% partial least squares regression and discriminant analysis. Chemom. 
% Intell. Lab. Syst., 176:34-43, 2018.
% URL: https://www.libpls.net/download.php

% Add libPLS package to MATLAB search path
addpath('/~path~/libPLS_1.98');

% Set string for saving results
str = 'FA';

for j = 1:10

    % Load data
    data = dlmread('/~path~/data.csv', ',', 1, 0);

    % Set seed for reproducibility, increase value by 1 for each CARS 
    % sampling run
    seed = 99+j;
    rng(seed);

    % Divide data into predictor matrix X and response vector y
    X = data(:,2:end);
    y = data(:,1);

    R2CV = [];
    nComp = [];
    nVar = [];

    % Standardize spectroscopy data to have columnwise mean of 0 and standard
    % deviation of 1
    X = zscore(X);

    % Find size of data
    n = size(X,1);

    % Sort data into 100 evenly spaced bins
    [Y,E] = discretize(y, 100);

    % Create a cell array of bins which stores the data in each bin
    for i = 1:100
        index = find(Y == i);
        binsY{i} = y(index);
        binsX{i} = X(index, :);
    end

    % Generate a maximum of 18 random samples for each bin and add to data
    % matrices
    Ybin = [];
    Xbin = [];

    for i = 1:100
        Binsize = size(binsY{i});

        clear addY;
        clear addX;
        clear sampIdx;

        % Set limit to 18 samples per bin unless Binsize < 18, in which 
        % case k = Binsize
        if Binsize(1) == 0
            continue;
        elseif Binsize(1) < 18
            k = Binsize(1);
        else
            k = 18;
        end

        % Sample k observations from each bin without replacement
        [addY, sampIdx] = datasample(binsY{i}, k, 'Replace', false);
        addX = binsX{i}(sampIdx,:);

        % Put sampled data back into vector and matrix form for y and X,
        % respectively
        Ybin = [Ybin addY'];
        Xbin = [Xbin; addX];
    end

    Ybin = Ybin';

    % Reassign X and y matrices to be the subset of the data we just
    % created
    X = Xbin;
    y = Ybin;
    
    % Find new size of our data
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

    disp(j)

    % Fit a CARS-PLS model to the data using 50 Monte Carlo sampling runs
    % and 10-fold cross validation
    CARS=carspls(X,y,50,10,'center',50,0,1,2);

    % Save results in vector
    R2CV = [R2CV CARS.Q2_max];
    nComp = [nComp CARS.optLV];
    nVar = [nVar size(CARS.vsel,2)];

    clearvars -except R2CV nComp nVar j str n
    
end

% Average results
R2CV = mean(R2CV);
nComp = mean(nComp);
nVar = mean(nVar);

% Print results and save to a .txt file
sprintf('CARS-PLS\n%s\nN: %g\nR2CV: %g\nnComp: %g\nnVar: %g', str, n, R2CV, nComp, nVar)

txtStr = strcat(str, ".txt");
fid = fopen(txtStr, 'wt');
fprintf(fid, 'CARS-PLS\n%s\nN: %g\nR2CV: %g\nnComp: %g\nnVar: %g', str, n, R2CV, nComp, nVar);
fclose(fid);




