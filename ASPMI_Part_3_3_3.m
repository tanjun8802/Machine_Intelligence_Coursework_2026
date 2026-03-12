%% ASPMI Part 3.3 – A Real-Time Spectrum Analyser Using the DFT-CLMS
% This script covers:
%   (a) Analytical note: LS solution is the DFT (shown in comments)
%   (b) Analytical note: DFT as change-of-basis / projection (shown in comments)
%   (c) DFT-CLMS applied to the FM signal from Part 3.2, compared to AR-CLMS
%   (d) DFT-CLMS applied to EEG signal POz (segment of length 1200)

clc; clear; close all;

%% =========================================================
%% Re-generate FM signal (same parameters as Part 3.2)
%% =========================================================
fs         = 1500;
N_fm       = 1500;
sigma2_eta = 0.05;

n2 = 501:1000;  n3 = 1001:1500;
f_vec = [100*ones(1,500), 100+(n2-500)/2, 100+((n3-1000)/25).^2];
phi   = cumsum(f_vec);
noise = sqrt(sigma2_eta/2) * (randn(1, N_fm) + 1j*randn(1, N_fm));
y_fm  = exp(1j * 2*pi/fs * phi) + noise;

%% =========================================================
%% (a & b) Analytical notes (no simulation needed)
%% =========================================================
% (a) The LS solution w = (F^H F)^{-1} F^H y satisfies:
%     F^H F = N * I  (because the DFT matrix F is orthogonal: F^H F / N = I)
%     Therefore w = (1/N) * F^H y  = DFT{y} (scaled by 1/N).
%     This is the definition of the Discrete Fourier Transform.
%
% (b) The DFT changes the basis from the standard time-domain basis
%     {delta(n-k)} to the frequency-domain basis {e^{j2πkn/N}}.
%     w_k = <y, e_k> / ||e_k||^2 is the projection of y onto
%     the k-th complex exponential (frequency bin k).
%     The DFT thus decomposes the signal into its frequency components
%     by orthogonal projection onto the DFT basis vectors.

%% =========================================================
%% (c) DFT-CLMS on FM signal
%% =========================================================
% The CLMS is run with mu = 1 and the phasor input x(n):
%   x(n) = [1, e^{j2πn/N}, e^{j4πn/N}, ..., e^{j2π(N-1)n/N}]^T / N
%
% The weight w_k(n) at each time step is an online DFT Fourier coefficient.
% Plotting |w(n)| over time produces a time-frequency diagram.

N_dft = N_fm;     % DFT size = signal length
mu_dft = 1;       % Learning rate (= 1 gives convergence to DFT per Widrow)

% Weight matrix: W(k, n) = |w_k(n)|, shape [N_dft x N_fm]
W = zeros(N_dft, N_fm);

w_dft = zeros(N_dft, 1);  % Initial weights (all zero)

for n = 0 : N_fm - 1
    % Phasor input vector at time n (1-based: n corresponds to n_vec = n/N)
    k_vec = (0 : N_dft - 1)';
    x_n   = exp(1j * 2*pi * k_vec * n / N_dft);   % Column vector of complex phasors

    % CLMS output (estimate of y(n))
    y_hat = w_dft' * x_n;
    % Error
    e_n   = y_fm(n+1) - y_hat;
    % CLMS weight update: w(n+1) = w(n) + mu * e*(n) * x(n)
    w_dft = w_dft + mu_dft * conj(e_n) * x_n;

    % Store magnitude of weight vector at this time step
    W(:, n+1) = abs(w_dft);
end

% Clip outliers for better visualisation
medW = 50 * median(median(W));
W(W > medW) = medW;

% Frequency axis: bin k corresponds to frequency k * fs / N_dft
freq_dft  = (0 : N_dft-1) * fs / N_dft;
time_axis = 1 : N_fm;

figure;
surf(time_axis, freq_dft, W, 'EdgeColor', 'none');
view(2);
colormap jet;
colorbar;
xlabel('Time (samples)');
ylabel('Frequency (Hz)');
title('DFT-CLMS Time-Frequency Diagram (FM signal)');
ylim([0 600]);

fprintf('=== Part (c): DFT-CLMS vs AR-CLMS ===\n');
fprintf('DFT-CLMS weights converge slowly because mu=1 is used globally\n');
fprintf('across N=%d weights – the effective per-bin step size is 1/N.\n', N_dft);
fprintf('This results in a blurry time-frequency map (the weights average\n');
fprintf('DFT coefficients over past samples, not just the current instant).\n');
fprintf('AR-CLMS (Part 3.2) tracks instantaneous frequency more sharply\n');
fprintf('because it only updates one coefficient per sample.\n\n');
fprintf('The DFT-CLMS weights |w_k(n)|^2 represent a running spectral estimate\n');
fprintf('rather than a true power spectrum, hence they do not match the AR power\n');
fprintf('spectrum directly.\n\n');

%% =========================================================
%% (d) DFT-CLMS on EEG signal (POz segment of length 1200)
%% =========================================================
eeg_file = 'EEG_Data\EEG_Data_Assignment1.mat';
if ~isfile(eeg_file)
    warning('EEG file not found: %s\nSkipping Part (d).', eeg_file);
else
    load(eeg_file);   % Loads: POz, fs

    % Select a segment of 1200 samples (adjust start index 'a' as desired)
    a      = 1;
    y_eeg  = POz(a : a + 1200 - 1);
    N_eeg  = length(y_eeg);   % = 1200

    % DFT-CLMS on EEG segment
    W_eeg = zeros(N_eeg, N_eeg);
    w_eeg = zeros(N_eeg, 1);

    for n = 0 : N_eeg - 1
        k_vec  = (0 : N_eeg - 1)';
        x_n    = exp(1j * 2*pi * k_vec * n / N_eeg);
        y_hat  = w_eeg' * x_n;
        e_n    = y_eeg(n+1) - y_hat;
        w_eeg  = w_eeg + mu_dft * conj(e_n) * x_n;
        W_eeg(:, n+1) = abs(w_eeg);
    end

    % Clip outliers
    medW_eeg = 50 * median(median(W_eeg));
    W_eeg(W_eeg > medW_eeg) = medW_eeg;

    % Frequency axis: 0 to fs Hz (for EEG, fs is typically 1200 Hz)
    freq_eeg  = (0 : N_eeg - 1) * fs / N_eeg;
    time_eeg  = (0 : N_eeg - 1) / fs;

    figure;
    surf(time_eeg, freq_eeg, W_eeg, 'EdgeColor', 'none');
    view(2);
    colormap jet;
    colorbar;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('DFT-CLMS Time-Frequency Diagram (EEG POz, 1200 samples)');
    ylim([0 100]);  % Zoom to EEG-relevant frequencies (0-100 Hz)

    fprintf('=== Part (d): DFT-CLMS on EEG ===\n');
    fprintf('The time-frequency diagram should reveal:\n');
    fprintf('  - Alpha rhythm (~8-10 Hz) if the subject is resting\n');
    fprintf('  - SSVEP peak at the stimulus frequency (from Assignment 1)\n');
    fprintf('  - 50 Hz power-line artefact\n');
    fprintf('The DFT-CLMS provides a continuous-time spectral snapshot,\n');
    fprintf('useful for tracking non-stationary EEG dynamics.\n');
end
