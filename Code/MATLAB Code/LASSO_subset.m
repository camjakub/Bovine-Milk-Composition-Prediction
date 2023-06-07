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

% Set string for saving results
str = 'FA';

R2 = [];
sampleN = [];
coefNum = [];
lam = [];

for j=1:10
    
    disp(j)

    % Load data
    data = dlmread('/~path~/data.csv', ',', 1, 0);

    % Divide data into predictor matrix X and response vector y
    X = data(:,2:end);
    y = data(:,1);

    % Standardize spectroscopy data to have columnwise mean of 0 and standard
    % deviation of 1
    X = zscore(X);

    % Find size of input matrix
    n = size(X,1);

    % Set seed for reproducibilty
    seed = 99 + j; % Change this value every time to get different results
    rng(seed);

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
    y = Ybin;
    X = Xbin;

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

    % Save results in vector
    sampleN = [sampleN n];
    R2 = [R2 R2CV];
    coefNum = [coefNum nCoef];
    lam = [lam fit.lambda_min];

    clearvars -except str sampleN R2 coefNum lam j;

end

% Average values of 10 runs
R2CV = mean(R2);
nCoef = mean(coefNum);
lambda = mean(lam);
n = mean(sampleN);

% Print results and save to a .txt file
sprintf('LASSO\n%s\nR2cv: %g\n# nonzero coef: %g\nLambda: %g\nN: %g', str, R2CV, nCoef, lambda, n)

txtStr = strcat(str, ".txt");
fid = fopen(txtStr, 'wt');
fprintf(fid,'LASSO\n%s\nR2cv: %g\n# nonzero coef: %g\nLambda: %g\nN: %g', str, R2CV, nCoef, lambda, n);
fclose(fid);







