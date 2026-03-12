%% ASPMI Part 3.2 – Adaptive AR Model Based Time-Frequency Estimation
% This script covers:
%   (a) FM signal generation; stationary AR(1) spectrum (aryule)
%   (b) CLMS-based online AR(1) coefficient estimation → time-frequency diagram

clc; clear; close all;

%% =========================================================
%% FM signal generation
%% =========================================================
% Instantaneous frequency:
%   f(n) = 100,                       1   <= n <= 500
%   f(n) = 100 + (n-500)/2,         501  <= n <= 1000
%   f(n) = 100 + ((n-1000)/25)^2,  1001 <= n <= 1500
%
% Phase: phi(n) = cumsum(f(n))   (discrete integral)
% Signal: y(n) = exp(j * 2*pi/fs * phi(n)) + noise

fs      = 1500;   % Nominal sampling frequency (samples / unit)
N_total = 1500;   % Total number of samples
sigma2_eta = 0.05; % Noise variance

% Build instantaneous frequency vector
n1  = 1:500;
n2  = 501:1000;
n3  = 1001:1500;

f_vec = [100 * ones(1, 500), ...
         100 + (n2 - 500) / 2, ...
         100 + ((n3 - 1000) / 25).^2];

% Phase = discrete integral of frequency (use cumsum)
phi = cumsum(f_vec);

% Complex FM signal
noise = sqrt(sigma2_eta/2) * (randn(1, N_total) + 1j*randn(1, N_total));
y     = exp(1j * 2*pi/fs * phi) + noise;

%% =========================================================
%% (a) Stationary AR(1) model using aryule
%% =========================================================
% Fit a single AR(1) model to the ENTIRE signal (treating it as stationary).
% This will miss the time-varying frequency content.

ar_order = 1;
a_yule   = aryule(y, ar_order);   % Returns [1, -a1_hat] for AR(1)

fprintf('=== Part (a): Stationary AR(1) coefficient ===\n');
fprintf('Estimated AR(1) coefficient: a_hat = %.4f + %.4fj\n', ...
    real(-a_yule(2)), imag(-a_yule(2)));
fprintf('Expected to be an average over the time-varying true coefficient.\n\n');

% Compute and plot the power spectrum of the stationary AR model
N_fft = 1024;
[h_ar, w_ar] = freqz(1, a_yule, N_fft);   % Frequency response of AR filter
P_ar         = abs(h_ar).^2;               % Power spectrum = |H(omega)|^2

figure;
plot(w_ar/(2*pi)*fs, 10*log10(P_ar + eps), 'b', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Stationary AR(1) Power Spectrum (aryule on full signal)');
grid on;
fprintf('Note: The stationary AR spectrum shows a blurred average frequency.\n');
fprintf('It does NOT capture the linear chirp or quadratic FM in Equation (37).\n\n');

%% =========================================================
%% (b) CLMS-based adaptive AR(1) → time-frequency diagram
%% =========================================================
% At each time step n, estimate the AR(1) coefficient a_hat(n) with CLMS,
% then compute the power spectrum using freqz with a_hat(n).
% This yields a time-varying (adaptive) spectral estimate.

mu_clms = 0.01;    % CLMS step size
N_fft   = 1024;    % Frequency bins for spectrum at each time step

% Storage matrix: H(k, n) = power at frequency bin k at time n
H = zeros(N_fft, N_total);

a_hat = 0 + 0j;   % AR(1) coefficient estimate, initialised to 0

for n = 2:N_total
    % ---- CLMS update for AR(1) ----
    % Input:  x(n)  = [y(n-1)]  (AR(1) has one lag)
    % Desired: y(n)
    y_pred  = a_hat * y(n-1);          % One-step prediction
    e       = y(n) - y_pred;           % Complex prediction error
    % CLMS: h(n+1) = h(n) + mu * e*(n) * x(n)
    a_hat   = a_hat + mu_clms * conj(e) * y(n-1);

    % ---- Compute power spectrum with current a_hat ----
    % freqz(1, [1; -a_hat*], N) gives the AR(1) frequency response
    [h_n, ~] = freqz(1, [1; -conj(a_hat)], N_fft);
    H(:, n)  = abs(h_n).^2;
end

% Remove outliers (cap large values) so the colour scale is informative
medianH = 50 * median(median(H));
H(H > medianH) = medianH;

% Frequency axis (0 to fs Hz)
[~, w_n] = freqz(1, [1; 0], N_fft);
freq_axis = w_n / (2*pi) * fs;

% Time axis
time_axis = 1:N_total;

% Plot time-frequency diagram
figure;
surf(time_axis, freq_axis, H, 'EdgeColor', 'none');
view(2);                    % 2D colour heat-map view
colormap jet;
colorbar;
xlabel('Time (samples)');
ylabel('Frequency (Hz)');
title('CLMS Adaptive AR(1) Time-Frequency Diagram');
ylim([0 500]);              % Zoom to relevant frequency range

fprintf('=== Part (b): CLMS time-frequency estimation ===\n');
fprintf('The time-frequency diagram shows the instantaneous frequency track:\n');
fprintf('  n=1..500  : constant ~100 Hz\n');
fprintf('  n=501..1000: linearly increasing (chirp)\n');
fprintf('  n=1001..1500: quadratically increasing\n');
fprintf('This is in contrast to the blurred stationary AR(1) spectrum in part (a).\n');
