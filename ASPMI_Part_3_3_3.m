%% ASPMI Part 3.3 – A Real-Time Spectrum Analyser Using the DFT-CLMS
% This script covers:
%   (a) Analytical: LS solution of DFT basis = DFT formula (comments only)
%   (b) Analytical: DFT as change-of-basis / projection (comments only)
%   (c) DFT-CLMS applied to the FM signal from Part 3.2
%   (d) DFT-CLMS applied to EEG signal POz (1200-sample segment)

clc; clear; close all;

%% =========================================================
%% Analytical notes – Parts (a) and (b)
%% =========================================================
% (a) LS solution of DFT problem
% The LS cost is: min_w ||y - F*w||^2,  solved by  w = (F^H F)^{-1} F^H y.
% The DFT basis matrix F satisfies F^H F = N * I  (orthonormal columns scaled by sqrt(N)).
% Therefore:  w = (1/N) * F^H * y = DFT coefficients (divided by N).
% This is exactly the Discrete Fourier Transform formula, confirming that
% the LS solution is the DFT.
%
% (b) DFT as change-of-basis / projection
% The DFT decomposes signal y into the orthogonal frequency basis
% {e_k | e_k(n) = e^{j2πkn/N}, k=0,...,N-1}.
% Each Fourier coefficient w_k = <y, e_k> / ||e_k||^2 is the projection
% of y onto the k-th complex exponential basis vector.
% This changes the representation from the time-domain (standard basis)
% to the frequency domain (DFT basis).

%% =========================================================
%% Re-generate FM signal (same as Part 3.2)
%% =========================================================
fs         = 1500;
N_fm       = 1500;
sigma2_eta = 0.05;

n2    = 501:1000;   n3 = 1001:1500;
f_vec = [100*ones(1,500), 100+(n2-500)/2, 100+((n3-1000)/25).^2];
phi   = cumsum(f_vec);
noise = sqrt(sigma2_eta/2) * (randn(1, N_fm) + 1j*randn(1, N_fm));
y_fm  = exp(1j * 2*pi/fs * phi) + noise;

%% =========================================================
%% (c) DFT-CLMS on FM signal
%% =========================================================
% The CLMS is configured with:
%   desired signal y(n)  →  time-domain FM signal
%   input vector x(n)    →  complex phasor at time n (length N_dft)
%
% x(n) = [1, e^{j2πn/N}, e^{j4πn/N}, ..., e^{j2π(N-1)n/N}]^T
%
% Weight update: w(n+1) = w(n) + mu * e*(n) * x(n)
% With mu=1 and w(0)=0, w(n) converges to DFT coefficients (Widrow et al.).

N_dft  = N_fm;   % DFT size equals signal length
mu_dft = 1;      % Step size = 1 for DFT convergence guarantee

% Weight magnitude matrix: W(k, n) = |w_k(n)| at time step n
W_fm = zeros(N_dft, N_fm);
w    = zeros(N_dft, 1);   % Initialise all weights to zero

for n = 0 : N_fm - 1
    % Build phasor input vector at time n
    k_vec   = (0 : N_dft - 1)';
    x_n     = exp(1j * 2*pi * k_vec * n / N_dft);  % Column vector

    % CLMS: compute output, error, and update
    y_hat   = w' * x_n;              % Current DFT estimate of y(n)
    e_n     = y_fm(n+1) - y_hat;    % Complex error
    w       = w + mu_dft * conj(e_n) * x_n;  % Weight update

    W_fm(:, n+1) = abs(w);          % Store weight magnitude
end

% Clip extreme values for better colourmap visualisation
medW = 50 * median(median(W_fm));
W_fm(W_fm > medW) = medW;

% Axes
freq_fm  = (0 : N_dft-1) * fs / N_dft;   % Hz
time_fm  = 1 : N_fm;                      % Sample index

figure;
surf(time_fm, freq_fm, W_fm, 'EdgeColor', 'none');
view(2);
colormap jet;
colorbar;
xlabel('Time (samples)');
ylabel('Frequency (Hz)');
title('DFT-CLMS Time-Frequency Diagram – FM signal (Part c)');
ylim([0 600]);

fprintf('=== Part (c): DFT-CLMS vs AR-CLMS ===\n');
fprintf('The DFT-CLMS diagram shows a blurry time-frequency track compared\n');
fprintf('to the AR-CLMS (Part 3.2) because:\n');
fprintf('  - DFT-CLMS updates N=%d weights jointly; effective per-bin step = 1/N.\n', N_dft);
fprintf('  - Weights accumulate a running average of DFT coefficients over\n');
fprintf('    all past samples, not just the current instant.\n');
fprintf('  - The weight magnitude |w_k(n)|^2 is NOT a power spectrum;\n');
fprintf('    it lacks the 1/N normalisation and includes cross-terms.\n');
fprintf('  - AR-CLMS tracks instantaneous frequency faster because it adapts\n');
fprintf('    only ONE coefficient per sample.\n\n');

%% =========================================================
%% (d) DFT-CLMS on EEG signal (POz, 1200-sample segment)
%% =========================================================
eeg_file = 'EEG_Data\EEG_Data_Assignment1.mat';
if ~isfile(eeg_file)
    warning('EEG file not found: %s\nSkipping Part (d).', eeg_file);
    return;
end

load(eeg_file);   % Loads: POz (EEG samples), fs (sampling frequency)

% Select 1200-sample segment (adjust start index 'a' as desired)
a       = 1;
y_eeg   = POz(a : a + 1200 - 1);
N_eeg   = length(y_eeg);           % = 1200

% DFT-CLMS on EEG segment
W_eeg = zeros(N_eeg, N_eeg);
w_eeg = zeros(N_eeg, 1);

for n = 0 : N_eeg - 1
    k_vec   = (0 : N_eeg - 1)';
    x_n     = exp(1j * 2*pi * k_vec * n / N_eeg);
    y_hat   = w_eeg' * x_n;
    e_n     = y_eeg(n+1) - y_hat;
    w_eeg   = w_eeg + mu_dft * conj(e_n) * x_n;
    W_eeg(:, n+1) = abs(w_eeg);
end

% Clip outliers
medW_eeg = 50 * median(median(W_eeg));
W_eeg(W_eeg > medW_eeg) = medW_eeg;

% Axes (frequency in Hz, time in seconds)
freq_eeg = (0 : N_eeg-1) * fs / N_eeg;   % Hz
time_eeg = (0 : N_eeg-1) / fs;            % Seconds

figure;
surf(time_eeg, freq_eeg, W_eeg, 'EdgeColor', 'none');
view(2);
colormap jet;
colorbar;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('DFT-CLMS Time-Frequency Diagram – EEG POz (1200 samples, Part d)');
ylim([0 100]);   % Focus on 0-100 Hz (EEG-relevant range)

fprintf('=== Part (d): DFT-CLMS on EEG ===\n');
fprintf('Expected observations in the time-frequency diagram:\n');
fprintf('  - Alpha rhythm (~8-10 Hz): persists if subject is in relaxed state\n');
fprintf('  - SSVEP peak (integer 11-20 Hz from Assignment 1): time-locked response\n');
fprintf('  - 50 Hz power-line artefact: bright horizontal stripe\n');
fprintf('The DFT-CLMS provides a sliding spectral estimate useful for tracking\n');
fprintf('non-stationary EEG dynamics, though slower than AR-CLMS.\n');
