%% ASPMI Part 2.3 – Adaptive Noise Cancellation
% This script covers:
%   (a) Adaptive Line Enhancer (ALE): minimum delay Delta, MATLAB program
%   (b) ALE with different filter orders M and delays Delta; MSPE analysis
%   (c) Adaptive Noise Cancellation (ANC) configuration
%   (d) EEG de-noising via ANC to remove 50 Hz mains interference

clc; clear; close all;

%% =========================================================
%% Signal generation
%% =========================================================
N      = 1000;           % Number of samples
omega0 = 0.01 * pi;      % Angular frequency of the sinusoid (rad/sample)
n_vec  = (0:N-1)';       % Sample index vector

% Clean sinusoid x(n): unit amplitude
x_clean = sin(omega0 * n_vec);

% Coloured noise: eta(n) = v(n) + 0.5*v(n-2),  v ~ N(0,1)
v    = randn(N+2, 1);                     % Extra samples to avoid edge effects
eta  = v(3:end) + 0.5 * v(1:end-2);      % Moving-average coloured noise (lag 2)

% Noisy signal: s(n) = x(n) + eta(n)
s = x_clean + eta;

%% =========================================================
%% (a) Adaptive Line Enhancer (ALE) – minimum delay
%% =========================================================
% The noise model is eta(n) = v(n) + 0.5*v(n-2).
% s(n - Delta) is correlated with s(n) through the sinusoid x(n).
% For the noise to be uncorrelated with s(n-Delta), we need Delta > 2
% (the noise autocorrelation r_eta(k) = 0 for |k| > 2).
% Therefore the MINIMUM delay is Delta_min = 3.

Delta_min = 3;
fprintf('=== Part (a): Minimum ALE delay ===\n');
fprintf('Noise: eta(n) = v(n) + 0.5*v(n-2).  r_eta(k)=0 for |k|>2.\n');
fprintf('Minimum delay Delta_min = %d samples.\n\n', Delta_min);

%% ---- ALE LMS function (defined at the bottom of this file) ----

%% Test ALE with M=5, Delta=Delta_min
M_demo  = 5;
mu_demo = 0.01;
[x_hat_demo, e_demo] = ale_lms(s, M_demo, Delta_min, mu_demo);

figure;
subplot(3,1,1); plot(n_vec, s);  title('Noisy signal s(n)');
xlabel('n'); ylabel('Amplitude'); grid on;
subplot(3,1,2); plot(n_vec, x_hat_demo); title('ALE output \hat{x}(n)  (M=5, \Delta=3, \mu=0.01)');
xlabel('n'); ylabel('Amplitude'); grid on;
subplot(3,1,3); plot(n_vec, x_clean); title('Clean reference x(n)');
xlabel('n'); ylabel('Amplitude'); grid on;
sgtitle('Adaptive Line Enhancer (ALE) – Demo');

%% =========================================================
%% (b) MSPE vs. delay Delta and filter order M
%% =========================================================
M_vals     = [5, 10, 15, 20];           % Filter orders to test
Delta_vals = Delta_min : 25;            % Delta from minimum to 25
mu_b       = 0.01;                      % Step size

MSPE = zeros(length(M_vals), length(Delta_vals));

for mi = 1:length(M_vals)
    for di = 1:length(Delta_vals)
        [x_hat, ~] = ale_lms(s, M_vals(mi), Delta_vals(di), mu_b);
        % MSPE = mean squared prediction error vs. clean signal
        MSPE(mi, di) = mean((x_clean - x_hat).^2);
    end
end

% Plot MSPE vs Delta for each M
figure;
for mi = 1:length(M_vals)
    plot(Delta_vals, 10*log10(MSPE(mi,:)), 'LineWidth', 1.5); hold on;
end
xlabel('Delay \Delta');
ylabel('MSPE (dB)');
title('ALE: MSPE vs. Delay \Delta for different filter orders M');
legend(arrayfun(@(m) sprintf('M = %d', m), M_vals, 'UniformOutput', false), 'Location', 'best');
grid on;

fprintf('=== Part (b): MSPE summary ===\n');
for mi = 1:length(M_vals)
    [best_mspe, best_idx] = min(MSPE(mi,:));
    fprintf('M=%2d: Best Delta=%2d, MSPE=%.4f (%.2f dB)\n', ...
        M_vals(mi), Delta_vals(best_idx), best_mspe, 10*log10(best_mspe));
end
fprintf('\nNote: Increasing M beyond ~10 yields diminishing MSPE reduction\n');
fprintf('because the sinusoidal component is captured well by a short filter.\n\n');

%% =========================================================
%% (c) Adaptive Noise Cancellation (ANC) configuration
%% =========================================================
% ANC uses a secondary reference input epsilon(n) that is correlated
% with the noise eta(n) but independent of x(n).
% Here epsilon(n) = v(n) (the underlying white noise) acts as the reference.
% The ANC filter learns to estimate eta(n) from epsilon(n), then subtracts.

% For this simulation, assume we have access to v(n) as the reference
% (in practice, a microphone capturing ambient noise plays this role).
epsilon = v(3:end);   % Reference noise (same underlying white noise as eta)

mu_anc = 0.01;
M_anc  = 5;

[x_hat_anc, e_anc] = anc_lms(s, epsilon, M_anc, mu_anc);

MSPE_anc = mean((x_clean - x_hat_anc).^2);
MSPE_ale = mean((x_clean - x_hat_demo).^2);  % Best ALE from demo

fprintf('=== Part (c): ANC vs. ALE comparison ===\n');
fprintf('ANC MSPE : %.4f (%.2f dB)\n', MSPE_anc, 10*log10(MSPE_anc));
fprintf('ALE MSPE : %.4f (%.2f dB)  (M=%d, Delta=%d)\n', ...
    MSPE_ale, 10*log10(MSPE_ale), M_demo, Delta_min);

figure;
subplot(3,1,1); plot(n_vec, s);  title('Noisy signal s(n)');
xlabel('n'); ylabel('Amplitude'); grid on;
subplot(3,1,2); plot(n_vec, x_hat_anc); title('ANC output \hat{x}(n)  (M=5, \mu=0.01)');
xlabel('n'); ylabel('Amplitude'); grid on;
subplot(3,1,3); plot(n_vec, x_clean); title('Clean reference x(n)');
xlabel('n'); ylabel('Amplitude'); grid on;
sgtitle('Adaptive Noise Cancellation (ANC)');

%% =========================================================
%% (d) EEG de-noising – remove 50 Hz mains interference
%% =========================================================
% Load single-channel EEG data from file EEG_Data_Assignment2.mat.
% The file contains variables Cz, POz, and fs.
% We use ANC to suppress the 50 Hz component using a synthetic reference.

eeg_file = 'EEG_Data\EEG_Data_Assignment2.mat';
if ~isfile(eeg_file)
    warning('EEG data file not found: %s\n  Skipping Part (d).', eeg_file);
else
    load(eeg_file);  % Loads: POz (or Cz), fs

    % Choose POz as the primary input
    primary = POz(:);
    Neeg    = length(primary);
    t_eeg   = (0:Neeg-1)' / fs;

    % Synthetic reference: 50 Hz sinusoid + small white noise
    % (models a power-line pickup channel)
    ref_50hz = sin(2*pi*50 * t_eeg) + 0.01*randn(Neeg,1);

    % ANC parameters (tune as needed)
    mu_eeg  = 0.01;
    M_eeg   = 32;   % Longer filter to handle phase offset of 50 Hz

    [eeg_clean, ~] = anc_lms(primary, ref_50hz, M_eeg, mu_eeg);

    % Spectrogram parameters
    win_len  = round(fs);        % 1-second window
    noverlap = round(win_len/2);
    nfft_sg  = 2 * win_len;

    figure;
    subplot(1,2,1);
    spectrogram(primary,  hamming(win_len), noverlap, nfft_sg, fs, 'yaxis');
    title('EEG Spectrogram – Original (with 50 Hz)');
    ylim([0 100]);
    colorbar;

    subplot(1,2,2);
    spectrogram(eeg_clean, hamming(win_len), noverlap, nfft_sg, fs, 'yaxis');
    title('EEG Spectrogram – After ANC De-noising');
    ylim([0 100]);
    colorbar;

    sgtitle('EEG 50 Hz Mains Removal Using Adaptive Noise Cancellation');
end

%% =========================================================
%% Local functions
%% =========================================================

function [x_hat, e] = ale_lms(s, M, Delta, mu)
% ALE_LMS  Adaptive Line Enhancer using LMS.
%   s     - noisy input signal (column vector)
%   M     - filter length (number of taps)
%   Delta - delay (integer >= 1)
%   mu    - step size
%
% Returns:
%   x_hat - estimated clean signal (filter output)
%   e     - prediction error (s(n) - x_hat(n))

    N     = length(s);
    x_hat = zeros(N, 1);
    e     = zeros(N, 1);
    w     = zeros(M, 1);    % Filter weights initialised to zero

    for n = M + Delta : N
        % Delayed input vector: [s(n-Delta), s(n-Delta-1), ..., s(n-Delta-M+1)]
        u      = s(n - Delta : -1 : n - Delta - M + 1);
        x_hat(n) = w' * u;     % Filter output (ALE estimate of x(n))
        e(n)     = s(n) - x_hat(n);  % Error = s(n) - estimated s(n)
        w        = w + mu * e(n) * u; % LMS update
    end
end

function [x_hat, e] = anc_lms(primary, reference, M, mu)
% ANC_LMS  Adaptive Noise Cancellation using LMS.
%   primary   - noisy primary signal s(n) = x(n) + eta(n)
%   reference - reference noise signal epsilon(n) (correlated with eta)
%   M         - filter length
%   mu        - step size
%
% Returns:
%   x_hat - estimated clean signal (primary minus estimated noise)
%   e     - estimation error (same as x_hat here)

    N     = length(primary);
    x_hat = zeros(N, 1);
    e     = zeros(N, 1);
    w     = zeros(M, 1);

    for n = M : N
        % Reference input vector
        u      = reference(n : -1 : n - M + 1);
        % Noise estimate
        eta_hat = w' * u;
        % Clean signal estimate: subtract noise estimate from primary
        e(n)    = primary(n) - eta_hat;
        x_hat(n) = e(n);
        % LMS update: minimise e(n)^2 w.r.t. w
        w = w + mu * e(n) * u;
    end
end
