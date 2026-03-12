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
% b1 = 1.5+1j  (strictly linear coefficient)
% b2 = 2.5-0.5j (widely linear / conjugate coefficient)

b1 = 1.5 + 1j;
b2 = 2.5 - 0.5j;

N  = 1000;    % Samples per trial
L  = 100;     % Independent trials (for ensemble average)
mu = 0.01;    % Learning rate

% Accumulators for squared error (linear scale)
e2_clms  = zeros(1, N);
e2_aclms = zeros(1, N);

for l = 1:L
    % Circular complex Gaussian input: x ~ CN(0,1)
    x = (randn(1, N+1) + 1j*randn(1, N+1)) / sqrt(2);

    % WLMA(1) output: y(n) = x(n) + b1*x(n-1) + b2*x*(n-1)
    y = x(2:end) + b1*x(1:end-1) + b2*conj(x(1:end-1));

    %% ---- CLMS (strictly linear, 1-tap) ----
    % Model: y_hat(n) = h * x(n-1)  — misses the conjugate term
    h = 0 + 0j;
    for n = 1:N
        y_hat        = h * x(n);          % Use x(n) as proxy for x(n-1) at step n
        e            = y(n) - y_hat;
        e2_clms(n)   = e2_clms(n) + abs(e)^2;
        h            = h + mu * conj(e) * x(n);   % CLMS: h += mu * e*(n) * x(n)
    end

    %% ---- ACLMS (widely linear, 1-tap h and g) ----
    % Model: y_hat(n) = h*x(n-1) + g*x*(n-1)  — uses both x and x*
    h = 0 + 0j;
    g = 0 + 0j;
    for n = 1:N
        y_hat         = h * x(n) + g * conj(x(n));
        e             = y(n) - y_hat;
        e2_aclms(n)   = e2_aclms(n) + abs(e)^2;
        h             = h + mu * conj(e) * x(n);          % Update h
        g             = g + mu * conj(e) * conj(x(n));    % Update g
    end
end

% Normalise by number of trials
e2_clms  = e2_clms  / L;
e2_aclms = e2_aclms / L;

% Plot learning curves in dB
figure;
plot(1:N, 10*log10(e2_clms  + eps), 'b',  'LineWidth', 1.5); hold on;
plot(1:N, 10*log10(e2_aclms + eps), 'r--', 'LineWidth', 1.5);
xlabel('Time step n');
ylabel('10 log_{10}(|e(n)|^2)  (dB)');
title('CLMS vs ACLMS Learning Curves – WLMA(1) Identification (L=100 trials)');
legend({'CLMS (strictly linear)', 'ACLMS (widely linear)'}, 'Location', 'best');
grid on;

% Steady-state error summary
ss_idx = round(0.8*N) : N;
fprintf('=== Part (a): Steady-state error ===\n');
fprintf('CLMS  steady-state error: %.2f dB\n', ...
    mean(10*log10(e2_clms(ss_idx)  + eps)));
fprintf('ACLMS steady-state error: %.2f dB\n', ...
    mean(10*log10(e2_aclms(ss_idx) + eps)));
fprintf('ACLMS achieves lower error because the WLMA(1) process is\n');
fprintf('non-circular (b2 ~= 0 introduces a pseudo-covariance).\n\n');

%% =========================================================
%% (b) Complex wind data prediction
%% =========================================================
% Data files: low-wind.mat, medium-wind.mat, high-wind.mat
% Each contains veast (E-W speed) and vnorth (N-S speed).
% Form complex wind: v[n] = veast[n] + j*vnorth[n]

wind_files = {'wind-dataset\low-wind.mat', ...
              'wind-dataset\medium-wind.mat', ...
              'wind-dataset\high-wind.mat'};
reg_names  = {'Low wind', 'Medium wind', 'High wind'};

if ~isfile(wind_files{1})
    warning('Wind data files not found. Skipping Part (b).');
    return;
end

% --- Circularity (scatter) plots ---
figure('Name', 'Wind circularity plots');
for k = 1:3
    data  = load(wind_files{k});
    v     = data.veast + 1j * data.vnorth;

    subplot(1, 3, k);
    scatter(real(v), imag(v), 5, 'filled', 'MarkerFaceAlpha', 0.3);
    axis equal;
    xlabel('East component (m/s)');
    ylabel('North component (m/s)');
    title(sprintf('%s – Circularity plot', reg_names{k}));
    grid on;
end
sgtitle('Complex wind signal: scatter (circularity) diagrams');
% A circular distribution → CLMS and ACLMS perform equally well.
% An elongated (non-circular) distribution → ACLMS outperforms CLMS.

% --- Prediction with CLMS and ACLMS ---
mu_wind = 0.01;
M_vals  = [1, 5, 10];   % Filter lengths to compare

fprintf('=== Part (b): Wind prediction MSPE ===\n');
fprintf('%-14s  M  %-14s  %-14s\n', 'Regime', 'CLMS MSPE', 'ACLMS MSPE');

for k = 1:3
    data = load(wind_files{k});
    v    = data.veast(:) + 1j * data.vnorth(:);  % Column vector
    Nw   = length(v);

    for m = 1:length(M_vals)
        M = M_vals(m);

        % ---- CLMS prediction ----
        h_c = zeros(M, 1);
        e2c = 0;
        for n = M+1 : Nw
            xv    = v(n-1 : -1 : n-M);          % Input vector
            e_c   = v(n) - h_c' * xv;           % CLMS prediction error
            h_c   = h_c + mu_wind * conj(e_c) * xv;
            e2c   = e2c + abs(e_c)^2;
        end
        mspe_c = e2c / (Nw - M);

        % ---- ACLMS prediction ----
        h_a = zeros(M, 1);
        g_a = zeros(M, 1);
        e2a = 0;
        for n = M+1 : Nw
            xv    = v(n-1 : -1 : n-M);
            e_a   = v(n) - h_a' * xv - g_a' * conj(xv);  % ACLMS prediction error
            h_a   = h_a + mu_wind * conj(e_a) * xv;
            g_a   = g_a + mu_wind * conj(e_a) * conj(xv);
            e2a   = e2a + abs(e_a)^2;
        end
        mspe_a = e2a / (Nw - M);

        fprintf('%-14s  %-2d  %-14.6f  %-14.6f\n', reg_names{k}, M, mspe_c, mspe_a);
    end
end
fprintf('\nHint: ACLMS outperforms CLMS for non-circular (high) wind regimes.\n');
