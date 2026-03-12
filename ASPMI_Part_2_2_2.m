%% ASPMI Part 2.2 – Adaptive Step Sizes
% This script covers:
%   (a) GASS algorithms: Benveniste, Ang & Farhang, Matthews & Xie
%       vs. standard LMS for MA(1) system identification
%   (b) Verification that the a posteriori error update = NLMS
%   (c) GNGD algorithm implementation and comparison with Benveniste

clc; clear; close all;

%% =========================================================
%% System parameters – MA(1) process
%% =========================================================
% x(n) = 0.9*eta(n-1) + eta(n),   eta ~ N(0, 0.5)
w_true   = 0.9;          % True MA(1) coefficient
sigma2_v = 0.5;          % Noise variance
N        = 1000;         % Samples per realization
L        = 100;          % Number of independent realizations

%% =========================================================
%% Helper: generate one MA(1) realization and its delayed version
%% =========================================================
% For system identification of x(n) = w_true * eta(n-1) + eta(n):
% Input to the adaptive filter: u(n) = eta(n-1)  (the delayed noise)
% Desired output: x(n) = w_true * u(n) + eta(n)
% This is a 1-tap FIR identification problem.

%% =========================================================
%% (a) GASS algorithms and standard LMS – weight error curves
%% =========================================================
% Weight error: w_tilde(n) = w_true - w(n)
% We run L independent realizations and average the weight error.

rho_gass = 0.005;   % Meta learning rate for GASS (common for all three)
alpha_af  = 0.9;    % Exponential forgetting factor for Ang & Farhang
mu_fixed  = [0.01, 0.1];   % Fixed step sizes for standard LMS

% Accumulated weight error squared over L trials, for each algorithm
% Algorithms: 1=LMS_0.01, 2=LMS_0.1, 3=Benveniste, 4=Ang&Farhang, 5=Matthews&Xie
nAlg = 5;
wtilde_sum = zeros(nAlg, N);   % Sum of |w_tilde|^2 across realizations

for l = 1:L
    % Generate noise eta and delayed version
    eta = sqrt(sigma2_v) * randn(1, N+1);
    u   = eta(1:N);       % input: eta(n-1)  (one-sample delayed noise)
    d   = w_true * u + eta(2:N+1);  % desired: x(n)

    % --- Standard LMS with mu = 0.01 ---
    w = 0; 
    for n = 1:N
        e  = d(n) - w * u(n);
        wtilde_sum(1, n) = wtilde_sum(1, n) + (w_true - w)^2;
        w  = w + mu_fixed(1) * e * u(n);
    end

    % --- Standard LMS with mu = 0.1 ---
    w = 0;
    for n = 1:N
        e  = d(n) - w * u(n);
        wtilde_sum(2, n) = wtilde_sum(2, n) + (w_true - w)^2;
        w  = w + mu_fixed(2) * e * u(n);
    end

    % --- Benveniste GASS ---
    % psi(n) = [I - mu(n-1)*u(n-1)*u(n-1)'] * psi(n-1) + e(n-1)*u(n-1)
    % mu(n+1) = mu(n) + rho * e(n) * u(n)' * psi(n)
    w = 0; mu = 0.1; psi = 0; e_prev = 0; u_prev = 0; mu_prev = mu;
    for n = 1:N
        e  = d(n) - w * u(n);
        wtilde_sum(3, n) = wtilde_sum(3, n) + (w_true - w)^2;
        % Update psi (scalar version for 1-tap filter)
        psi = (1 - mu_prev * u_prev^2) * psi + e_prev * u_prev;
        % Update mu (clamp to positive to keep stability)
        mu_new = mu + rho_gass * e * u(n) * psi;
        mu_new = max(1e-6, mu_new);
        % Update weight
        w = w + mu * e * u(n);
        % Shift state
        mu_prev = mu; mu = mu_new;
        u_prev = u(n); e_prev = e;
    end

    % --- Ang & Farhang GASS ---
    % psi(n) = alpha * psi(n-1) + e(n-1) * u(n-1)
    w = 0; mu = 0.1; psi = 0; e_prev = 0; u_prev = 0;
    for n = 1:N
        e  = d(n) - w * u(n);
        wtilde_sum(4, n) = wtilde_sum(4, n) + (w_true - w)^2;
        % Update psi
        psi = alpha_af * psi + e_prev * u_prev;
        % Update mu
        mu_new = mu + rho_gass * e * u(n) * psi;
        mu_new = max(1e-6, mu_new);
        % Update weight
        w = w + mu * e * u(n);
        mu = mu_new;
        u_prev = u(n); e_prev = e;
    end

    % --- Matthews & Xie GASS ---
    % psi(n) = e(n-1) * u(n-1)  (no memory)
    w = 0; mu = 0.1; e_prev = 0; u_prev = 0;
    for n = 1:N
        e  = d(n) - w * u(n);
        wtilde_sum(5, n) = wtilde_sum(5, n) + (w_true - w)^2;
        % Update psi (instantaneous)
        psi = e_prev * u_prev;
        % Update mu
        mu_new = mu + rho_gass * e * u(n) * psi;
        mu_new = max(1e-6, mu_new);
        % Update weight
        w = w + mu * e * u(n);
        mu = mu_new;
        u_prev = u(n); e_prev = e;
    end
end

% Average over realizations
wtilde_avg = wtilde_sum / L;

% Plot weight error curves
alg_names = {'LMS \mu=0.01', 'LMS \mu=0.1', 'Benveniste', 'Ang & Farhang', 'Matthews & Xie'};
colors     = lines(nAlg);

figure;
for k = 1:nAlg
    plot(1:N, 10*log10(wtilde_avg(k,:) + eps), 'Color', colors(k,:), 'LineWidth', 1.5);
    hold on;
end
xlabel('Time step n');
ylabel('Weight error |\tilde{w}(n)|^2 (dB)');
title('GASS algorithms vs. Standard LMS – Weight Error Curves');
legend(alg_names, 'Location', 'best');
grid on;

%% =========================================================
%% (b) NLMS = a posteriori error update (verification)
%% =========================================================
% The NLMS update is:
%   w(n+1) = w(n) + beta / (eps + ||x(n)||^2) * e(n) * x(n)
%
% The a posteriori error is ep(n) = d(n) - x^T(n)*w(n+1).
% Starting from Δw(n) = w(n+1)-w(n) = mu*ep(n)*x(n), and substituting:
%   ep(n) = d(n) - x^T(n)*(w(n) + mu*ep(n)*x(n))
%         = e(n) - mu*ep(n)*||x(n)||^2
%   => ep(n)*(1 + mu*||x(n)||^2) = e(n)
%   => ep(n) = e(n) / (1 + mu*||x(n)||^2)
% Then:
%   Δw(n) = mu / (1 + mu*||x(n)||^2) * e(n) * x(n)
%         = beta / (eps + ||x(n)||^2) * e(n) * x(n)
% where beta = mu / (1 + mu*eps) → mu and eps_nlms → mu * eps.
%
% Numerically verify: NLMS and a-posteriori-based update give identical trajectories.

eps_nlms = 1e-6;   % Regularisation factor
beta_nlms = 0.5;   % Normalised step size (0 < beta < 2 for convergence)
mu_apost  = beta_nlms;  % Equivalent standard step size in a-posteriori form

fprintf('=== Part (b): NLMS equivalence verification ===\n');

% Generate a single MA(1) trial
eta = sqrt(sigma2_v) * randn(1, N+1);
u   = eta(1:N);
d   = w_true * u + eta(2:N+1);

% Standard NLMS
w_nlms = zeros(1, N+1);
for n = 1:N
    e_n = d(n) - w_nlms(n) * u(n);
    w_nlms(n+1) = w_nlms(n) + beta_nlms / (eps_nlms + u(n)^2) * e_n * u(n);
end

% A-posteriori error NLMS (equivalent derivation)
% w(n+1) = w(n) + mu*ep(n)*u(n), ep(n) = e(n)/(1 + mu*u(n)^2)
w_apost = zeros(1, N+1);
for n = 1:N
    e_n  = d(n) - w_apost(n) * u(n);
    ep_n = e_n / (1 + mu_apost * u(n)^2);
    w_apost(n+1) = w_apost(n) + mu_apost * ep_n * u(n);
end

max_diff = max(abs(w_nlms - w_apost));
fprintf('Max absolute difference between NLMS and a-posteriori form: %.2e\n', max_diff);
fprintf('(Should be ~0 since they are analytically equivalent)\n\n');

figure;
plot(1:N+1, w_nlms,  'b',  'LineWidth', 1.5); hold on;
plot(1:N+1, w_apost, 'r--', 'LineWidth', 1.5);
yline(w_true, 'k:', 'LineWidth', 1.2);
xlabel('Time step n');
ylabel('Weight estimate w(n)');
title('NLMS vs. A-posteriori Error Form (should overlap exactly)');
legend({'NLMS', 'A-posteriori form', 'True w = 0.9'}, 'Location', 'best');
grid on;

%% =========================================================
%% (c) GNGD algorithm and comparison with Benveniste
%% =========================================================
% GNGD update:
%   w(n+1)     = w(n) + beta / (eps(n) + ||x(n)||^2) * e(n) * x(n)
%   eps(n+1)   = eps(n) - rho*mu * (e(n)*e(n-1)*u(n)'*u(n-1))
%                         / (eps(n-1) + ||u(n-1)||^2)^2

rho_gngd  = 0.05;     % GNGD meta learning rate
beta_gngd = 1;        % Fixed normalised step size for GNGD
eps0      = 1;        % Initial regularisation

wtilde_benv = zeros(1, N);
wtilde_gngd = zeros(1, N);

for l = 1:L
    eta = sqrt(sigma2_v) * randn(1, N+1);
    u   = eta(1:N);
    d   = w_true * u + eta(2:N+1);

    % --- Benveniste (repeated from part a for direct comparison) ---
    w = 0; mu_b = 0.1; psi = 0; e_prev = 0; u_prev = 0; mu_prev = mu_b;
    for n = 1:N
        e = d(n) - w * u(n);
        wtilde_benv(n) = wtilde_benv(n) + (w_true - w)^2;
        psi    = (1 - mu_prev * u_prev^2) * psi + e_prev * u_prev;
        mu_new = max(1e-6, mu_b + rho_gass * e * u(n) * psi);
        w      = w + mu_b * e * u(n);
        mu_prev = mu_b; mu_b = mu_new;
        u_prev = u(n); e_prev = e;
    end

    % --- GNGD ---
    w = 0; eps_n = eps0; eps_prev = eps0; e_prev = 0; u_prev = 0;
    for n = 1:N
        e = d(n) - w * u(n);
        wtilde_gngd(n) = wtilde_gngd(n) + (w_true - w)^2;
        % Update epsilon
        denom  = (eps_prev + u_prev^2)^2;
        eps_new = eps_n - rho_gngd * beta_gngd * (e * e_prev * u(n) * u_prev) / (denom + eps);
        eps_new = max(1e-6, eps_new);   % Keep epsilon non-negative
        % NLMS-style weight update
        w = w + beta_gngd / (eps_n + u(n)^2) * e * u(n);
        % Shift state
        eps_prev = eps_n; eps_n = eps_new;
        u_prev = u(n); e_prev = e;
    end
end

wtilde_benv = wtilde_benv / L;
wtilde_gngd = wtilde_gngd / L;

figure;
plot(1:N, 10*log10(wtilde_benv + eps), 'b', 'LineWidth', 1.5); hold on;
plot(1:N, 10*log10(wtilde_gngd + eps), 'r--', 'LineWidth', 1.5);
xlabel('Time step n');
ylabel('Weight error |\tilde{w}(n)|^2 (dB)');
title('Benveniste GASS vs. GNGD – Weight Error Comparison');
legend({'Benveniste', 'GNGD'}, 'Location', 'best');
grid on;

fprintf('=== Part (c): Computational complexity comparison ===\n');
fprintf('Benveniste GASS:\n');
fprintf('  Weight update:   1 multiply + 1 add (standard LMS)\n');
fprintf('  mu update:       3 multiplies + 1 add\n');
fprintf('  psi update:      3 multiplies + 2 adds\n');
fprintf('  Total ~7 mults, 4 adds per step.\n\n');
fprintf('GNGD:\n');
fprintf('  Weight update:   2 multiplies + 1 add (NLMS)\n');
fprintf('  eps update:      5 multiplies + 1 add + 1 divide\n');
fprintf('  Total ~7 mults + 1 divide, 2 adds per step.\n');
fprintf('  (GNGD is slightly more expensive due to the division for eps update)\n');
