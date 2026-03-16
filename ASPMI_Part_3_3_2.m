%% ASPMI Part 3.2 – Adaptive AR Model Based Time-Frequency Estimation
% This script covers:
%   (a) FM signal generation; stationary AR(1) spectrum using aryule
%   (b) CLMS-based adaptive AR(1) estimation → time-frequency diagram

clc; clear; close all;

%% =========================================================
%% FM signal generation (Equation 37 in the coursework)
%% =========================================================
% Instantaneous frequency:
%   f(n) = 100,                         1   ≤ n ≤ 500
%   f(n) = 100 + (n-500)/2,           501  ≤ n ≤ 1000
%   f(n) = 100 + ((n-1000)/25)^2,    1001  ≤ n ≤ 1500

fs      = 1500;    % Sampling frequency (determines frequency axis scaling)
N_total = 1500;    % Total number of samples

% Build instantaneous frequency vector
n2    = 501:1000;
n3    = 1001:1500;
f_vec = [100 * ones(1, 500), ...
         100 + (n2 - 500) / 2, ...
         100 + ((n3 - 1000) / 25).^2];

% Phase = discrete integral of f (cumulative sum approximates integral)
phi = cumsum(f_vec);

% Complex FM signal + circular complex Gaussian noise (variance 0.05)
sigma2_eta = 0.05;
noise = sqrt(sigma2_eta/2) * (randn(1, N_total) + 1j*randn(1, N_total));
y     = exp(1j * 2*pi / fs * phi) + noise;

%% =========================================================
%% (a) Stationary AR(1) model (aryule on the full signal)
%% =========================================================
% Fitting a single AR(1) to the entire signal treats it as stationary.
% The estimated coefficient will be an average over the changing frequency.

a_yule = aryule(y, 1);   % Returns [1, -a1_hat] for AR(1) order

fprintf('=== Part (a): Stationary AR(1) ===\n');
fprintf('AR(1) coefficient: a_hat = %.4f%+.4fj\n', real(-a_yule(2)), imag(-a_yule(2)));

% Compute the corresponding power spectrum: P(omega) = |H(omega)|^2
N_fft = 1024;
[h_ar, w_ar] = freqz(1, a_yule, N_fft);
P_ar         = abs(h_ar).^2;

% Map normalised frequency to Hz (w_ar in rad/sample, 0..pi → 0..fs/2)
freq_hz = w_ar / (2*pi) * fs;

figure;
plot(freq_hz, 10*log10(P_ar + eps), 'b', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Stationary AR(1) Power Spectrum (aryule, full signal)');
xlim([0 fs/2]);
grid on;

fprintf('The spectrum shows a single broad peak — it does NOT capture\n');
fprintf('the time-varying frequency (constant → linear → quadratic FM).\n\n');

%% =========================================================
%% (b) CLMS-based adaptive AR(1) → time-frequency diagram
%% =========================================================
% At each sample n, the CLMS updates a single AR(1) coefficient a_hat(n).
% We then compute the local power spectrum using freqz(1, [1; -conj(a_hat)], N_fft).
% Stacking these spectra over time gives a time-frequency representation.

mu_clms = 0.01;    % CLMS step size (tune: too large → unstable, too small → slow tracking)
N_fft   = 1024;    % Frequency resolution per time step

% H(k, n) stores the power at frequency bin k at time n
H = zeros(N_fft, N_total);

a_hat = 0 + 0j;   % AR(1) coefficient initialised to zero

for n = 2:N_total
    % ----- CLMS weight update -----
    % Desired: y(n), Input: y(n-1) (AR(1))
    y_pred = a_hat * y(n-1);           % One-step ahead prediction
    e      = y(n) - y_pred;            % Complex prediction error
    % CLMS update: a_hat(n+1) = a_hat(n) + mu * e*(n) * y(n-1)
    a_hat  = a_hat + mu_clms * conj(e) * y(n-1);

    % ----- Power spectrum from current coefficient -----
    % AR(1): H(z) = 1 / (1 - a_hat * z^{-1})
    % Note: freqz evaluates |1 / (1 - a_hat*e^{-jω})|^2
    [h_n, ~]  = freqz(1, [1; -conj(a_hat)], N_fft);
    H(:, n)   = abs(h_n).^2;
end

% Cap large values (outliers) to improve colour-scale visibility
medH = 50 * median(median(H));
H(H > medH) = medH;

% Frequency axis: 0..fs/2 for positive frequencies
[~, w_n]   = freqz(1, [1; 0], N_fft);
freq_axis  = w_n / (2*pi) * fs;    % Hz
time_axis  = 1 : N_total;          % Sample index

% Plot time-frequency diagram as a 2D colour heat-map
figure;
surf(time_axis, freq_axis, H, 'EdgeColor', 'none');
view(2);               % Collapse 3D surf to 2D colour image
colormap jet;
colorbar;
xlabel('Time (samples)');
ylabel('Frequency (Hz)');
title('CLMS Adaptive AR(1) – Time-Frequency Diagram (FM signal)');
ylim([0 600]);         % Zoom to relevant range (0 to ~600 Hz)

fprintf('=== Part (b): CLMS time-frequency ===\n');
fprintf('The colour heat-map should clearly show three segments:\n');
fprintf('  n=1..500   : horizontal stripe at ~100 Hz (constant)\n');
fprintf('  n=501..1000: upward-sloping stripe (linear chirp)\n');
fprintf('  n=1001..1500: rapidly rising stripe (quadratic FM)\n');
fprintf('Compared to the blurred stationary spectrum in (a), the adaptive\n');
fprintf('AR(1) CLMS tracks the instantaneous frequency in real time.\n');
