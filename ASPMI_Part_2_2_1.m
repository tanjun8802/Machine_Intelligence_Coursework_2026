%% ASPMI Part 2.1 – The Least Mean Square (LMS) Algorithm
% This script covers:
%   (a) Correlation matrix of the AR(2) input and convergence range for mu
%   (b) LMS adaptive predictor: squared error and learning curve
%   (c) Misadjustment estimation and comparison with theory
%   (d) Steady-state weight estimates
%   (e) Derivation note for Leaky LMS cost function
%   (f) Leaky LMS implementation

clc; clear; close all;

%% =========================================================
%% System parameters (same throughout Part 2.1)
%% =========================================================
a1    = 0.1;    % AR(2) coefficient 1
a2    = 0.8;    % AR(2) coefficient 2
sigma2_eta = 0.25;  % Noise variance (sigma_eta^2)
N     = 1000;   % Samples per realization
L     = 100;    % Number of independent realizations (for ensemble average)

%% =========================================================
%% (a) Correlation matrix R and convergence range for mu
%% =========================================================
% For the AR(2) process x(n) = a1*x(n-1) + a2*x(n-2) + eta(n),
% the input to the LMS filter is x(n) = [x(n-1), x(n-2)]^T.
% The 2x2 autocorrelation matrix is R = [[r(0), r(1)]; [r(1), r(0)]].
%
% The Yule-Walker equations give (for k >= 1):
%   r(k) = a1*r(k-1) + a2*r(k-2),       k >= 2
%   r(0) = a1*r(1) + a2*r(2) + sigma2_eta
%
% Solving: r(1) = a1*r(0) + a2*r(1)  =>  r(1) = a1/(1-a2) * r(0)
% Then r(2) = a1*r(1) + a2*r(0)
% And  r(0) from r(0) = a1*r(1) + a2*r(2) + sigma2_eta

% Compute r(1) in terms of r(0): r(1) = a1*r(0)/(1-a2)
c = a1 / (1 - a2);          % r(1) = c * r(0)
% r(2) = a1*r(1) + a2*r(0) = (a1*c + a2)*r(0)
d = a1 * c + a2;             % r(2) = d * r(0)
% Substitute into r(0) equation: r(0) = a1*c*r(0) + a2*d*r(0) + sigma2_eta
% => r(0)*(1 - a1*c - a2*d) = sigma2_eta
r0 = sigma2_eta / (1 - a1*c - a2*d);
r1 = c * r0;

% Build 2x2 autocorrelation matrix
R = [r0, r1; r1, r0];

fprintf('=== Part (a): Correlation matrix R ===\n');
disp(R);

% LMS convergence in the mean requires: 0 < mu < 2/lambda_max
% where lambda_max is the largest eigenvalue of R.
lambda_max = max(eig(R));
mu_max = 2 / lambda_max;
fprintf('Largest eigenvalue of R: %.4f\n', lambda_max);
fprintf('Convergence range: 0 < mu < %.4f\n\n', mu_max);

%% =========================================================
%% (b) LMS adaptive predictor – squared error and learning curves
%% =========================================================
mu_vals = [0.05, 0.01];  % Two step sizes to compare

% Storage: rows = realizations, columns = time steps
e2_all = zeros(L, N, length(mu_vals));  % Squared error for each mu

for m = 1:length(mu_vals)
    mu = mu_vals(m);
    for l = 1:L
        % Generate AR(2) process of length N+2 (extra samples for initial conditions)
        eta = sqrt(sigma2_eta) * randn(1, N+2);
        x   = zeros(1, N+2);
        for n = 3:N+2
            x(n) = a1*x(n-1) + a2*x(n-2) + eta(n);
        end
        x = x(3:end);  % Discard first 2 initialisation samples

        % LMS adaptive predictor (filter order M=2)
        w = zeros(2, 1);    % Initial weights [w1; w2]
        for n = 3:N
            % Input vector: x at lags 1 and 2
            xv = [x(n-1); x(n-2)];
            % Prediction and error
            x_hat = w' * xv;
            e      = x(n) - x_hat;
            % Squared prediction error in dB
            e2_all(l, n, m) = 10 * log10(e^2 + eps);
            % LMS weight update
            w = w + mu * e * xv;
        end
    end
end

% Plot squared error for a single realisation and the ensemble-averaged learning curve
figure;
for m = 1:length(mu_vals)
    subplot(1, 2, m);
    % Single realization squared error
    plot(1:N, squeeze(e2_all(1,:,m)), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    hold on;
    % Ensemble-averaged learning curve
    plot(1:N, mean(squeeze(e2_all(:,:,m)), 1), 'b', 'LineWidth', 1.5);
    xlabel('Time step n');
    ylabel('10 log_{10}(e^2(n)) (dB)');
    title(sprintf('Learning Curve – \\mu = %.2f', mu_vals(m)));
    legend({'Single realisation', 'Ensemble average (L=100)'}, 'Location', 'best');
    grid on;
end
sgtitle('LMS Adaptive Predictor: Squared Prediction Error');

%% =========================================================
%% (c) Misadjustment – estimated vs. theoretical
%% =========================================================
% Theoretical misadjustment (small step-size approximation):
%   M_LMS ≈ (mu/2) * Tr{R}
% Tr{R} = r(0) + r(0) = 2*r(0) for a 2x2 matrix with equal diagonal entries.
TrR = trace(R);

fprintf('=== Part (c): Misadjustment ===\n');
for m = 1:length(mu_vals)
    mu = mu_vals(m);

    % Theoretical misadjustment
    M_theory = (mu / 2) * TrR;

    % Empirical misadjustment: use second half of each learning curve (steady state)
    % MSE_ss = mean of e^2 (not in dB) in steady state, averaged over L trials
    % First reconstruct e^2 (linear) from e2_all which is already in dB
    lc_db     = mean(squeeze(e2_all(:,:,m)), 1);    % ensemble-averaged learning curve (dB)
    lc_lin    = 10.^(lc_db / 10);                   % convert back to linear
    ss_region = round(0.5*N):N;                      % steady-state region (last 50%)
    MSE_ss    = mean(lc_lin(ss_region));             % steady-state MSE

    M_est = (MSE_ss - sigma2_eta) / sigma2_eta;

    fprintf('mu = %.2f: Theoretical M = %.4f, Estimated M = %.4f\n', mu, M_theory, M_est);
end
fprintf('\n');

%% =========================================================
%% (d) Steady-state weight estimates
%% =========================================================
fprintf('=== Part (d): Steady-state weight estimates ===\n');
for m = 1:length(mu_vals)
    mu = mu_vals(m);
    w_all = zeros(L, 2);  % Store final weights for each trial

    for l = 1:L
        eta = sqrt(sigma2_eta) * randn(1, N+2);
        x   = zeros(1, N+2);
        for n = 3:N+2
            x(n) = a1*x(n-1) + a2*x(n-2) + eta(n);
        end
        x = x(3:end);

        w = zeros(2, 1);
        % Run LMS and record the final weight vector
        for n = 3:N
            xv    = [x(n-1); x(n-2)];
            e     = x(n) - w' * xv;
            w     = w + mu * e * xv;
        end
        w_all(l, :) = w';
    end

    % Average final weights over all L trials
    w_mean = mean(w_all, 1);
    fprintf('mu = %.2f: w1_est = %.4f (true %.1f), w2_est = %.4f (true %.1f)\n', ...
        mu, w_mean(1), a1, w_mean(2), a2);
end
fprintf('Standard LMS converges to unbiased estimates of the AR coefficients.\n');
fprintf('Any small deviation from the true values is due to finite N and noise variance,\n');
fprintf('not a systematic bias. Leaky LMS (Part f) introduces genuine bias via the L2 term.\n\n');

%% =========================================================
%% (e) Leaky LMS derivation note (analytical – shown in comments)
%% =========================================================
% The leaky LMS minimises J2(n) = (1/2)(e^2(n) + gamma*||w(n)||^2).
% Taking the stochastic gradient:
%   dJ2/dw = -e(n)*x(n) + gamma*w(n)
% The LMS-type update is:
%   w(n+1) = w(n) - mu * dJ2/dw = w(n) + mu*e(n)*x(n) - mu*gamma*w(n)
%           = (1 - mu*gamma)*w(n) + mu*e(n)*x(n)
% which is exactly the Leaky LMS equation (Eq. 19 in the coursework).

%% =========================================================
%% (f) Leaky LMS implementation
%% =========================================================
gamma_vals = [0.001, 0.01, 0.1];   % Leakage coefficients to test
mu_leaky   = 0.05;                  % Fixed step size for leaky LMS

fprintf('=== Part (f): Leaky LMS steady-state weights ===\n');
fprintf('True coefficients: a1 = %.1f, a2 = %.1f\n', a1, a2);

% Plot weight trajectories for different gamma values
figure;
colors = lines(length(gamma_vals));

for g = 1:length(gamma_vals)
    gamma = gamma_vals(g);
    w_all_leaky = zeros(L, 2);

    for l = 1:L
        eta = sqrt(sigma2_eta) * randn(1, N+2);
        x   = zeros(1, N+2);
        for n = 3:N+2
            x(n) = a1*x(n-1) + a2*x(n-2) + eta(n);
        end
        x = x(3:end);

        w         = zeros(2, 1);
        w_traj    = zeros(2, N);  % Weight trajectory (for first realization only)
        for n = 3:N
            xv         = [x(n-1); x(n-2)];
            e          = x(n) - w' * xv;
            w          = (1 - mu_leaky*gamma) * w + mu_leaky * e * xv;
            w_traj(:, n) = w;
        end
        w_all_leaky(l, :) = w';
    end

    w_leaky_mean = mean(w_all_leaky, 1);
    fprintf('gamma = %.3f: w1_est = %.4f, w2_est = %.4f\n', gamma, w_leaky_mean(1), w_leaky_mean(2));

    % Plot weight trajectory for the last trial
    subplot(length(gamma_vals), 1, g);
    plot(1:N, w_traj(1,:), 'Color', colors(1,:), 'LineWidth', 1.2); hold on;
    plot(1:N, w_traj(2,:), 'Color', colors(2,:), 'LineWidth', 1.2);
    yline(a1, '--', 'Color', colors(1,:), 'LineWidth', 1);
    yline(a2, '--', 'Color', colors(2,:), 'LineWidth', 1);
    xlabel('Time step n');
    ylabel('Weight value');
    title(sprintf('Leaky LMS weight trajectories: \\gamma = %.3f, \\mu = %.2f', gamma, mu_leaky));
    legend({'w_1(n)', 'w_2(n)', 'True a_1', 'True a_2'}, 'Location', 'best');
    grid on;
end
sgtitle('Leaky LMS: Weight trajectories for different \gamma values');

fprintf('\nNote: The leaky LMS converges to a biased solution because the L2 penalty\n');
fprintf('in J2 shrinks the weights toward zero, not toward the true AR coefficients.\n');
