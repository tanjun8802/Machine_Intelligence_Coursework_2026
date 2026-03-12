%% ASPMI Part 3.1 – Complex LMS (CLMS) and Augmented CLMS (ACLMS)
% This script covers:
%   (a) WLMA(1) system identification using CLMS and ACLMS
%       – learning curves and steady-state error comparison
%   (b) Complex wind data prediction (circularity plots, CLMS vs ACLMS)

clc; clear; close all;

%% =========================================================
%% (a) WLMA(1) system identification
%% =========================================================
% Process: y(n) = x(n) + b1*x(n-1) + b2*x*(n-1),  x ~ CN(0,1)
% where b1 = 1.5+1j, b2 = 2.5-0.5j

b1 = 1.5 + 1j;     % Strictly linear MA coefficient
b2 = 2.5 - 0.5j;   % Widely linear (conjugate) MA coefficient

% Simulation parameters
N  = 1000;   % Samples per trial
L  = 100;    % Number of independent trials (for ensemble average)
mu = 0.01;   % Learning rate (same for both CLMS and ACLMS)

% Storage for ensemble-averaged squared error (in linear scale)
e2_clms  = zeros(1, N);
e2_aclms = zeros(1, N);

for l = 1:L
    % Generate circular complex Gaussian input x(n)
    x = (randn(1, N+1) + 1j*randn(1, N+1)) / sqrt(2);  % CN(0,1)
    % Generate WLMA(1) output y(n) for n = 1..N
    y = x(2:end) + b1*x(1:end-1) + b2*conj(x(1:end-1));

    %% ---- CLMS (strictly linear, 1-tap) ----
    h_clms  = zeros(1, 1);    % Weight vector for CLMS (scalar here)
    for n = 2:N
        % Input: x(n-1)
        xv       = x(n);          % At time n, input is x(n-1) (index n in 1-based)
        y_hat    = h_clms * xv;   % Strictly linear prediction
        e        = y(n) - y_hat;
        e2_clms(n) = e2_clms(n) + abs(e)^2;
        % CLMS weight update: h(n+1) = h(n) + mu * e*(n) * x(n)
        h_clms   = h_clms + mu * conj(e) * xv;
    end

    %% ---- ACLMS (widely linear, 1-tap h and g) ----
    h_aclms = zeros(1, 1);  % Weight for x(n-1)
    g_aclms = zeros(1, 1);  % Weight for x*(n-1)
    for n = 2:N
        xv    = x(n);
        xv_c  = conj(xv);
        % Widely linear output: y_hat = h*x + g*x*
        y_hat    = h_aclms * xv + g_aclms * xv_c;
        e        = y(n) - y_hat;
        e2_aclms(n) = e2_aclms(n) + abs(e)^2;
        % ACLMS weight updates
        h_aclms = h_aclms + mu * conj(e) * xv;
        g_aclms = g_aclms + mu * conj(e) * xv_c;
    end
end

% Average over L trials
e2_clms  = e2_clms  / L;
e2_aclms = e2_aclms / L;

% Plot learning curves in dB
figure;
plot(1:N, 10*log10(e2_clms  + eps), 'b',  'LineWidth', 1.5); hold on;
plot(1:N, 10*log10(e2_aclms + eps), 'r--', 'LineWidth', 1.5);
xlabel('Time step n');
ylabel('10 log_{10}(|e(n)|^2)  (dB)');
title('CLMS vs ACLMS Learning Curves for WLMA(1) Identification (L=100 trials)');
legend({'CLMS', 'ACLMS'}, 'Location', 'best');
grid on;

% Print steady-state error (last 20% of samples)
ss_region = round(0.8*N):N;
fprintf('=== Part (a): Steady-state error (dB) ===\n');
fprintf('CLMS  : %.2f dB\n', mean(10*log10(e2_clms(ss_region)  + eps)));
fprintf('ACLMS : %.2f dB\n', mean(10*log10(e2_aclms(ss_region) + eps)));
fprintf('ACLMS should achieve lower error because the WLMA(1) process is non-circular.\n\n');

%% =========================================================
%% (b) Complex wind data prediction
%% =========================================================
% Load wind data; file contains veast and vnorth for three regimes:
%   low wind, medium wind, high wind (column vectors).
wind_file = 'wind-dataset\low-wind.mat';

if ~isfile(wind_file)
    warning('Wind data file not found. Skipping Part (b).');
else
    % Load all three wind regimes
    low_wind  = load('wind-dataset\low-wind.mat');
    med_wind  = load('wind-dataset\medium-wind.mat');
    high_wind = load('wind-dataset\high-wind.mat');

    regimes  = {low_wind, med_wind, high_wind};
    reg_names = {'Low wind', 'Medium wind', 'High wind'};

    % For each regime, form complex signal v[n] = veast + j*vnorth
    figure('Name', 'Wind circularity plots');
    for k = 1:3
        data  = regimes{k};
        v_cplx = data.veast + 1j * data.vnorth;

        subplot(1, 3, k);
        scatter(real(v_cplx), imag(v_cplx), 5, 'filled');
        axis equal;
        xlabel('East (m/s)');
        ylabel('North (m/s)');
        title(sprintf('%s – Circularity plot', reg_names{k}));
        grid on;
    end
    sgtitle('Complex wind signal: circularity (scatter) plots');

    % Prediction comparison: CLMS vs ACLMS for each regime
    mu_wind   = 0.01;
    M_vals    = [1, 5, 10];  % Filter lengths to experiment with

    fprintf('=== Part (b): Wind prediction – MSPE comparison ===\n');
    fprintf('%-12s  %-4s  %-12s  %-12s\n', 'Regime', 'M', 'CLMS MSPE', 'ACLMS MSPE');

    for k = 1:3
        data   = regimes{k};
        v      = data.veast + 1j * data.vnorth;
        Nw     = length(v);

        for mi = 1:length(M_vals)
            M = M_vals(mi);
            [mspe_c, mspe_a] = wind_predict(v, M, mu_wind);
            fprintf('%-12s  M=%-2d  %.6f      %.6f\n', reg_names{k}, M, mspe_c, mspe_a);
        end
    end
end

%% =========================================================
%% Local function: CLMS and ACLMS in prediction setting
%% =========================================================

function [mspe_clms, mspe_aclms] = wind_predict(v, M, mu)
% WIND_PREDICT  One-step ahead prediction using CLMS and ACLMS.
%   v    - complex-valued wind signal (column vector)
%   M    - filter length (taps)
%   mu   - step size

    Nw = length(v);
    e2_clms  = 0;
    e2_aclms = 0;
    count    = 0;

    h = zeros(M, 1);    % CLMS weights
    g = zeros(M, 1);    % ACLMS conjugate weights

    for n = M+1 : Nw
        % Input vector: M most recent samples
        xv    = v(n-1 : -1 : n-M);
        xv_c  = conj(xv);

        % CLMS prediction
        y_clms  = h' * xv;
        e_clms  = v(n) - y_clms;
        h       = h + mu * conj(e_clms) * xv;

        % ACLMS prediction (reuse same xv)
        y_aclms = h' * xv + g' * xv_c;   % Note: h here is already updated; use a copy
        % (Re-run with separate weight copies for fair comparison)
        e2_clms  = e2_clms  + abs(e_clms)^2;
        count    = count + 1;
    end

    % Rerun ACLMS separately to avoid entangling with CLMS updates
    ha = zeros(M, 1);
    ga = zeros(M, 1);
    count_a = 0;
    for n = M+1 : Nw
        xv    = v(n-1 : -1 : n-M);
        y_a   = ha' * xv + ga' * conj(xv);
        e_a   = v(n) - y_a;
        ha    = ha + mu * conj(e_a) * xv;
        ga    = ga + mu * conj(e_a) * conj(xv);
        e2_aclms = e2_aclms + abs(e_a)^2;
        count_a  = count_a + 1;
    end

    mspe_clms  = e2_clms  / count;
    mspe_aclms = e2_aclms / count_a;
end
