clc;
clear all;
close all;

load("PCAPCR.mat")

S = svd(X);
Snoise = svd(Xnoise);
rankX = rank(X);             
rankX_noise = rank(Xnoise);

% figure;
% stem(S, 'filled'); hold on;
% stem(Snoise, 'filled');
% xlabel('Index');
% ylabel('\sigma_i');
% title('Singular values of X and X_{noise}');
% legend({'X','X_{noise}'}, 'Location','best');
% 
% sqErr = (S - Snoise).^2;
% 
% % Plot
% figure;
% stem(sqErr, 'filled');
% xlabel('Singular value index i');
% ylabel('(\sigma_i(X) - \sigma_i(X_{noise}))^2');
% title('Squared error between singular values of X and X_{noise}');
% grid on;

% [U,S_n,V] = svd(Xnoise,'econ');
% 
% r = rank(X);                
% 
% Ur = U(:,1:r);
% Sr = S_n(1:r,1:r);
% Vr = V(:,1:r);
% 
% Xtilde_noise = Ur * Sr * Vr';

% err_noise  = vecnorm(X - Xnoise);       
% err_denois = vecnorm(X - Xtilde_noise);
% 
% figure;
% stem(err_noise,'r','filled'); hold on;
% stem(err_denois,'b','filled');
% xlabel('Singular Value index in terms of magnitude');
% ylabel('Squared-error');
% legend('Error X vs X_{noise}','Error X vs X̃_{noise}');
% title('Singular value squared-error before and after low-rank denoising');
% grid on;

% OLS and PCR part
% 
% B_hat_OLS = Xnoise \ Y;          
% 
% Yhat_OLS = Xnoise * B_hat_OLS; 
% 
% B_hat_PCR = Vr * (Sr \ (Ur.' * Y));   % V1:r * (Σ1:r^{-1}) * U1:r.' * Y

% Yhat_PCR = Xtilde_noise * B_hat_PCR;
% T_test = Xtest * Vr;            
% Xtilde_test = T_test * Vr.';     
% 
% trainErr_OLS = norm(Y - Yhat_OLS, 'fro');
% trainErr_PCR = norm(Y - Yhat_PCR, 'fro');
% 
% fprintf('Train error OLS : %.4f\n', trainErr_OLS);
% fprintf('Train error PCR : %.4f\n', trainErr_PCR);
% 
% Yhat_test_OLS = Xtest       * B_hat_OLS;
% Yhat_test_PCR = Xtilde_test * B_hat_PCR;
% 
% testErr_OLS = norm(Ytest - Yhat_test_OLS, 'fro');
% testErr_PCR = norm(Ytest - Yhat_test_PCR, 'fro');
% 
% fprintf('Test error OLS  : %.4f\n', testErr_OLS);
% fprintf('Test error PCR  : %.4f\n', testErr_PCR);

% [Yhat_OLS, Y_OLS] = regval(B_hat_OLS);
% [Yhat_PCR, Y_PCR] = regval(B_hat_PCR);
% 
% MSE_OLS = mean( sum( (Y_OLS  - Yhat_OLS ).^2, 2 ) );
% MSE_PCR = mean( sum( (Y_PCR  - Yhat_PCR ).^2, 2 ) );
% 
% fprintf('MSE OLS : %.4f\n', MSE_OLS);
% fprintf('MSE PCR : %.4f\n', MSE_PCR);


%% 1.6.1 PLS part

maxComp = rankX_noise;

MSE_test_PLS = zeros(maxComp,1);
MSE_test_PCR = zeros(maxComp,1);

for r = 1:maxComp
    
    [~,~,~,~,betaPLS,~,~,~] = plsregress(Xnoise, Y, r);
    
    Xtest_aug = [ones(size(Xtest,1),1), Xtest];
    
    Yhat_test_PLS = Xtest_aug * betaPLS;
    
    diffPLS = Ytest - Yhat_test_PLS;
    MSE_test_PLS(r) = mean(sum(diffPLS.^2, 2));
    
    [U,S,V] = svd(Xnoise,'econ');
    
    Ur = U(:,1:r);
    Sr = S(1:r,1:r);
    Vr = V(:,1:r);
    
    Xtilde_noise = Ur * Sr * Vr';
    Ttest        = Xtest * Vr;       
    Xtilde_test  = Ttest * Vr';
    
    B_hat_PCR = Vr * (Sr \ (Ur' * Y));
    Yhat_test_PCR = Xtilde_test * B_hat_PCR;
    
    diffPCR = Ytest - Yhat_test_PCR;
    MSE_test_PCR(r) = mean(sum(diffPCR.^2, 2));
end

disp(table((1:maxComp)', MSE_test_PLS, MSE_test_PCR, ...
    'VariableNames', {'Components','MSE_PLS','MSE_PCR'}));








