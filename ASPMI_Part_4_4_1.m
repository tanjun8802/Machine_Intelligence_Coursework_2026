%% ASPMI Part 4.1 – Neural Networks for Prediction
% This script covers:
%   1. LMS adaptive predictor (AR(4) assumption) on the non-stationary time-series
%   2. Dynamical perceptron with tanh activation
%   3. Scaled activation: a * tanh(.)
%   4. Bias term to handle non-zero mean
%   5. Pre-training on first 20 samples to initialise the weights
%
% Note: Parts 6 and 7 (deep network training) require Python/PyTorch.
%       See: https://github.com/am5113/ASPMI_DeepLearning

clc; clear; close all;

%% =========================================================
%% Load and inspect the time-series
%% =========================================================
ts_file = 'time-series.mat';
if ~isfile(ts_file)
    error('File not found: %s\nDownload it from Blackboard.', ts_file);
end

data = load(ts_file);
% The mat file contains one variable (assumed 'y' or the first field)
fn   = fieldnames(data);
y_original = double(data.(fn{1}));
y_original = y_original(:);   % Ensure column vector

N   = length(y_original);
n_vec = (1:N)';   % Sample index

fprintf('=== Time-series info ===\n');
fprintf('Length: %d samples\n', N);
fprintf('Mean: %.4f, Std: %.4f\n', mean(y_original), std(y_original));

% Zero-mean version of the signal
y_mean = mean(y_original);
y      = y_original - y_mean;

%% =========================================================
%% 1. LMS adaptive predictor (AR(4) model)
%% =========================================================
% Predict y[n] from y[n-1], y[n-2], y[n-3], y[n-4] using LMS.
% Model order M=4, learning rate mu = 1e-5.

M  = 4;          % AR model order
mu = 1e-5;       % Learning rate (as specified in the coursework)

w_lms  = zeros(M, 1);   % Initial weights
y_pred_lms = zeros(N, 1);

for n = M+1 : N
    % Input vector: 4 most recent zero-mean samples
    xv           = y(n-1 : -1 : n-M);
    y_pred_lms(n) = w_lms' * xv;          % LMS prediction
    e             = y(n) - y_pred_lms(n); % Prediction error
    w_lms         = w_lms + mu * e * xv;  % LMS weight update
end

% MSE and prediction gain
MSE_lms = mean((y(M+1:end) - y_pred_lms(M+1:end)).^2);
sigma2_y = var(y_pred_lms(M+1:end));   % Variance of LMS output
sigma2_e = var(y(M+1:end) - y_pred_lms(M+1:end));   % Variance of error
Rp_lms   = 10 * log10(sigma2_y / sigma2_e);

fprintf('\n=== Part 1: LMS predictor (AR4) ===\n');
fprintf('MSE       = %.4f\n', MSE_lms);
fprintf('Prediction gain Rp = %.2f dB\n', Rp_lms);

figure;
plot(n_vec, y,           'b',  'LineWidth', 1.0); hold on;
plot(n_vec, y_pred_lms,  'r--', 'LineWidth', 1.0);
xlabel('Time step n');
ylabel('Amplitude');
title('LMS Adaptive Predictor – Zero-mean signal vs. Prediction');
legend({'Zero-mean signal y[n]', 'LMS prediction \hat{y}[n]'}, 'Location', 'best');
grid on;

%% =========================================================
%% 2. Dynamical perceptron with tanh
%% =========================================================
% Output: y_hat[n] = tanh(w^T x[n])
% Update: w(n+1) = w(n) + mu * e(n) * (1 - tanh^2(w^T x[n])) * x[n]
%         (gradient of tanh is sech^2 = 1 - tanh^2)

w_tanh = zeros(M, 1);
y_pred_tanh = zeros(N, 1);

for n = M+1 : N
    xv                = y(n-1 : -1 : n-M);
    a_n               = w_tanh' * xv;         % Pre-activation
    y_pred_tanh(n)    = tanh(a_n);            % Activation output
    e                 = y(n) - y_pred_tanh(n);
    % Gradient: d(tanh(a))/da = 1 - tanh^2(a) = sech^2(a)
    grad = (1 - tanh(a_n)^2);
    w_tanh = w_tanh + mu * e * grad * xv;     % Chain-rule update
end

MSE_tanh = mean((y(M+1:end) - y_pred_tanh(M+1:end)).^2);
fprintf('\n=== Part 2: Dynamical perceptron (tanh) ===\n');
fprintf('MSE with tanh = %.4f\n', MSE_tanh);
fprintf('tanh maps to (-1, 1). If y has amplitude >> 1, tanh saturates\n');
fprintf('and cannot represent the signal faithfully.\n');

figure;
plot(n_vec, y,            'b',  'LineWidth', 1.0); hold on;
plot(n_vec, y_pred_tanh,  'r--', 'LineWidth', 1.0);
xlabel('Time step n');
ylabel('Amplitude');
title('Dynamical Perceptron with tanh – Zero-mean signal vs. Prediction');
legend({'Zero-mean y[n]', 'tanh prediction'}, 'Location', 'best');
grid on;

%% =========================================================
%% 3. Scaled activation: a * tanh
%% =========================================================
% The tanh output range is (-1, 1). Scaling by 'a' extends it to (-a, a).
% A good choice of 'a' is the std deviation (or peak amplitude) of y, so
% the activation can represent the signal's full dynamic range.

a_scale = std(y) * 1.5;   % Scale to ~1.5 sigma of signal std dev
fprintf('\n=== Part 3: Scaled activation a*tanh, a = %.4f ===\n', a_scale);

w_scaled = zeros(M, 1);
y_pred_scaled = zeros(N, 1);

for n = M+1 : N
    xv                  = y(n-1 : -1 : n-M);
    a_n                 = w_scaled' * xv;
    y_pred_scaled(n)    = a_scale * tanh(a_n);        % Scaled activation
    e                   = y(n) - y_pred_scaled(n);
    % Gradient of (a * tanh(a_n)): a * sech^2(a_n)
    grad = a_scale * (1 - tanh(a_n)^2);
    w_scaled = w_scaled + mu * e * grad * xv;
end

MSE_scaled  = mean((y(M+1:end) - y_pred_scaled(M+1:end)).^2);
sigma2_e_sc = var(y(M+1:end) - y_pred_scaled(M+1:end));
Rp_scaled   = 10 * log10(var(y_pred_scaled(M+1:end)) / sigma2_e_sc);

fprintf('MSE with scaled tanh = %.4f (vs. LMS MSE = %.4f)\n', MSE_scaled, MSE_lms);
fprintf('Prediction gain Rp   = %.2f dB\n', Rp_scaled);

figure;
plot(n_vec, y,              'b',  'LineWidth', 1.0); hold on;
plot(n_vec, y_pred_scaled,  'r--', 'LineWidth', 1.0);
xlabel('Time step n');
ylabel('Amplitude');
title(sprintf('Scaled Perceptron (a=%.2f)*tanh – Zero-mean vs. Prediction', a_scale));
legend({'Zero-mean y[n]', sprintf('%.2f*tanh prediction', a_scale)}, 'Location', 'best');
grid on;

%% =========================================================
%% 4. Bias term for non-zero mean (predict original y_original)
%% =========================================================
% Augment input with a constant 1 to allow the model to learn the bias:
%   x_aug = [1; y(n-1); y(n-2); y(n-3); y(n-4)]
%   y_hat[n] = a_scale * tanh(w_aug^T x_aug)

M_bias   = M + 1;     % Augmented weight vector length (M weights + 1 bias)
w_bias   = zeros(M_bias, 1);
y_pred_bias = zeros(N, 1);

for n = M+1 : N
    % Augmented input: [1; y_original(n-1); ...; y_original(n-M)]
    xv_aug            = [1; y_original(n-1 : -1 : n-M)];
    a_n               = w_bias' * xv_aug;
    y_pred_bias(n)    = a_scale * tanh(a_n);
    e                 = y_original(n) - y_pred_bias(n);
    grad              = a_scale * (1 - tanh(a_n)^2);
    w_bias            = w_bias + mu * e * grad * xv_aug;
end

MSE_bias = mean((y_original(M+1:end) - y_pred_bias(M+1:end)).^2);
fprintf('\n=== Part 4: Bias term (predicting original y_original) ===\n');
fprintf('MSE with bias = %.4f\n', MSE_bias);
fprintf('The bias allows the model to track the non-zero mean automatically.\n');

figure;
plot(n_vec, y_original,     'b',  'LineWidth', 1.0); hold on;
plot(n_vec, y_pred_bias,    'r--', 'LineWidth', 1.0);
xlabel('Time step n');
ylabel('Amplitude');
title('Perceptron with Bias – Original signal (non-zero mean) vs. Prediction');
legend({'y_{original}[n]', 'Prediction with bias'}, 'Location', 'best');
grid on;

%% =========================================================
%% 5. Pre-training on first 20 samples
%% =========================================================
% Idea: overfit to the first 20 samples using 100 epochs to get a
% good weight initialisation, then use those weights for the full run.

N_pretrain = 20;     % Number of samples for pre-training
n_epochs   = 100;    % Epochs of over-fitting to the first 20 samples

w_pretrain = zeros(M_bias, 1);   % Start from zero

% Pre-train: iterate 100 times over the first 20 samples
for epoch = 1:n_epochs
    for n = M+1 : N_pretrain
        xv_aug   = [1; y_original(n-1 : -1 : n-M)];
        a_n      = w_pretrain' * xv_aug;
        y_hat    = a_scale * tanh(a_n);
        e        = y_original(n) - y_hat;
        grad     = a_scale * (1 - tanh(a_n)^2);
        w_pretrain = w_pretrain + mu * e * grad * xv_aug;
    end
end

fprintf('\n=== Part 5: Pre-trained weights ===\n');
fprintf('Pre-training completed: %d epochs on first %d samples.\n', n_epochs, N_pretrain);
fprintf('w_init = [%.4f, %.4f, %.4f, %.4f, %.4f]\n', w_pretrain);

% Now use w_pretrain as initialisation for the full prediction
w_pt         = w_pretrain;   % Initialise with pre-trained weights
y_pred_pt    = zeros(N, 1);

for n = M+1 : N
    xv_aug        = [1; y_original(n-1 : -1 : n-M)];
    a_n           = w_pt' * xv_aug;
    y_pred_pt(n)  = a_scale * tanh(a_n);
    e             = y_original(n) - y_pred_pt(n);
    grad          = a_scale * (1 - tanh(a_n)^2);
    w_pt          = w_pt + mu * e * grad * xv_aug;
end

MSE_pt = mean((y_original(M+1:end) - y_pred_pt(M+1:end)).^2);
fprintf('MSE with pre-trained weights = %.4f (vs. bias MSE = %.4f)\n', MSE_pt, MSE_bias);

figure;
plot(n_vec, y_original,   'b',  'LineWidth', 1.0); hold on;
plot(n_vec, y_pred_pt,    'r--', 'LineWidth', 1.0);
xlabel('Time step n');
ylabel('Amplitude');
title('Perceptron with Pre-trained Weights + Bias – Original vs. Prediction');
legend({'y_{original}[n]', 'Pre-trained prediction'}, 'Location', 'best');
grid on;

fprintf('\nNote: Pre-training gives a good initial estimate, reducing the adaptation\n');
fprintf('transient at the start of the online prediction phase.\n');
