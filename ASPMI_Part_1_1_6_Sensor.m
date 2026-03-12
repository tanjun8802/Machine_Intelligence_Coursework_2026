clc;
clear all;
close all;

T = readtable('AirQualityUCI.csv');

Y = T.C6H6_GT_;
X = [ ...
    T.CO_GT_, ...
    T.PT08_S1_CO_, ...
    T.NMHC_GT_, ...
    T.PT08_S2_NMHC_, ...
    T.NOx_GT_, ...
    T.PT08_S3_NOx_, ...
    T.NO2_GT_, ...
    T.PT08_S4_NO2_, ...
    T.PT08_S5_O3_, ...
    T.T, ...
    T.RH, ...
    T.AH ];

valid = all(~ismissing(X),2) & ~ismissing(Y);
X = X(valid,:);
Y = Y(valid);

% Test-train split
N = size(X,1);
idx = randperm(N);
Ntrain = round(0.7*N);
itrain = idx(1:Ntrain);
itest  = idx(Ntrain+1:end);

Xtrain = X(itrain,:);
Ytrain = Y(itrain,:);
Xtest  = X(itest,:);
Ytest  = Y(itest,:);

maxComp = 8; % Since only have 8 input variables
RMSE_PCR = zeros(maxComp,1);
RMSE_PLS = zeros(maxComp,1);

for r = 1:maxComp
    
    [U,S,V] = svd(Xtrain,'econ');
    Ur = U(:,1:r);
    Sr = S(1:r,1:r);
    Vr = V(:,1:r);
    
    Xtilde_train = Ur * Sr * Vr';
    Ttest        = Xtest * Vr;
    Xtilde_test  = Ttest * Vr';
    
    b_PCR = Vr * (Sr \ (Ur' * Ytrain));   
    Yhat_test_PCR = Xtilde_test * b_PCR; 
    
    err_PCR = Ytest - Yhat_test_PCR;
    RMSE_PCR(r) = sqrt(mean(err_PCR.^2)); 
    
    [~,~,~,~,betaPLS] = plsregress(Xtrain, Ytrain, r);
    Xtest_aug = [ones(size(Xtest,1),1) Xtest];
    Yhat_test_PLS = Xtest_aug * betaPLS;
    
    err_PLS = Ytest - Yhat_test_PLS;
    RMSE_PLS(r) = sqrt(mean(err_PLS.^2));
end

table((1:maxComp)', RMSE_PLS, RMSE_PCR, ...
    'VariableNames', {'Components','RMSE_PLS','RMSE_PCR'})

