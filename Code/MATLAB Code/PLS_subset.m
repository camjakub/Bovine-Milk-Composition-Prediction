% Code used to run partial least squares regression with subset selection
% on fatty acid/MFG data
% Cameron Jakub

maxComp = [];
R2maxComp = [];
sampleN = [];

% Set string for saving results
str = 'FA';

for j = 1:10
    % Load data
    data = dlmread('/~path~/data.csv', ',', 1, 0);

    % Divide data into predictor matrix X and response vector y
    Xtemp = data(:,2:end);
    y = data(:,1);

    % Standardize spectroscopy data to have columnwise mean of 0 and standard
    % deviation of 1
    Xtemp = zscore(Xtemp);

    % Find size of input matrix
    n = size(Xtemp,1);

    % Sort data into 100 evenly spaced bins
    [Y,E] = discretize(y, 100);

    % Create a cell array of bins which stores the data in each bin
    for i = 1:100
        index = find(Y == i);
        binsY{i} = y(index);
        binsX{i} = Xtemp(index, :);
    end

    % Set a seed for reproducibility
    seed = 99 + j; % Change this value every time to get different results
    rng(seed);

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

    % Now that we have removed outliers, we will perform a PLS regression with
    % 100 components and 10-fold cross validation
    [XL, YL, XS, YS, betaCoef, pctVar, plsMSEP, stats] = plsregress(X, y, 100, 'CV', 10);
  
    % Find minimum MSEP & its corresponding index
    minMSEP = min(plsMSEP(2,:));
    idxMin = find(plsMSEP(2,:) == minMSEP) -  1;

    % Calculate fitted values of our model
    Fit = [ones(n,1) X]*betaCoef;

    % Calculate R2CV
    SStot = sum((y - mean(y)).^2);
    SSEP = n*minMSEP;
    R2CV = 1 - SSEP/SStot;
    
    % Save results in vector
    R2maxComp = [R2maxComp R2CV];
    maxComp = [maxComp idxMin];
    sampleN = [sampleN n];

    clearvars -except maxComp str R2maxComp i sampleN;
    
end

% Average values of 10 runs
idxMin = mean(maxComp);
R2CV = mean(R2maxComp);
n = mean(sampleN);

% Print results and save to a .txt file
sprintf("PLSR\n%s\nR2: %g\nnComp: %g\nN: %g", str, R2CV, idxMin, n)

txtStr = strcat(str, ".txt");
fid = fopen(txtStr, 'wt');
fprintf(fid, "PLSR\n%s\nR2: %g\nnComp: %g\nN: %g", str, R2CV, idxMin, n);
fclose(fid);




