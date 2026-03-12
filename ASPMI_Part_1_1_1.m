clear; clc; close all;

N    = 4096;     % Number of samples per realization (time length)
L    = 200;      % Number of independent realizations (for ensemble averaging)
nfft = N;        % FFT length (use N for 1:1 mapping of bins to samples)

% Choose which case to run:
run_case = 'Converge';          % Options: 'AR1', 'RandomWalk', 'Custom'

% Parameters for AR(1)
a_AR     = 0.01;            % AR(1) coefficient (|a_AR|<1 => stationary, fast ACF decay)
sigma_AR = 1;              % Std dev of driving white noise

% Parameters for Random Walk
sigma_RW = 1;              % Std dev of step noise

%% ============== DEFINE PROCESS GENERATOR ==============
% Here we define a function handle gen_process(N) that returns one realization
% x[0..N-1] of the chosen process.

% AR(1) process: x[n] - a_AR * x[n-1] = w[n], w ~ N(0, sigma_AR^2)
% filter(1, [1 -a_AR], w) implements this recursion
gen_process = @(N) filter(1, [1 -a_AR], sigma_AR*randn(1,N));
process_name = sprintf('Converge, a=%.2f', a_AR);

% We know the theoretical ACF and PSD for AR(1), so we set these
use_theoretical = true;
sigma_x2 = sigma_AR^2 / (1 - a_AR^2);               % Variance of AR(1)
% Theoretical ACF: r(k) = sigma_x2 * a_AR^{|k|}
r_theo_fun = @(k) sigma_x2 * (abs(a_AR).^abs(k));
% Frequency grid in rad/sample
omega = 2*pi*(0:nfft-1)/nfft;
% Theoretical PSD: P(ω) = σ_w^2 / |1 - a e^{-jω}|^2
P_theo_fun = @(omega) sigma_x2 * (1 - a_AR^2) ./ ...
                      (abs(1 - a_AR*exp(-1j*omega)).^2);

%% ============== GENERATE REALIZATIONS ==============
% Generate L independent realizations of length N using gen_process
x_all = zeros(L, N);                % Preallocate matrix [L x N]
for r = 1:L
    x_all(r,:) = gen_process(N);    % Row r = one realization
end

%% ============== ESTIMATE ACF (ENSEMBLE) ==============
% We want the ensemble ACF r(k) ≈ E{x[n] x*(n-k)}.
% Approximate it by averaging xcorr() over all realizations.

maxLag = N-1;                       % Compute ACF for lags -N+1..N-1
r_est  = zeros(1, 2*maxLag+1);      % Will hold average ACF over realizations

for r = 1:L
    % xcorr with 'biased' normalization: divides by N instead of N-|k|
    [acf, lags] = xcorr(x_all(r,:), maxLag, 'biased');
    r_est = r_est + acf;            % Accumulate ACF from each realization
end
r_est = r_est / L;                  % Average across realizations → ensemble estimate

%% ============== PSD VIA DTFT OF ACF (DEF. 1) ==============
% Definition 1: P(ω) = Σ_k r(k) e^{-jωk}
% We approximate this DTFT by an N-point FFT of r(k) for k = 0..N-1.

% Index of lag k=0 is at the center of r_est
center_idx = (length(r_est)+1)/2;

% Extract lags k = 0..N-1 into a length-N vector
r0_to_Nm1  = r_est(center_idx : center_idx + N - 1);

% FFT of r(k) approximates samples of P(ω)
P1         = real(fft(r0_to_Nm1));  % Real part; imaginary part is numerical noise

%% ============== PSD VIA AVERAGED PERIODOGRAM (DEF. 2) ==============
% Definition 2: P(ω) = lim_{N→∞} E{ (1/N) |Σ_n x(n)e^{-jωn}|^2 }
% Here we approximate the expectation with an average over L realizations.

P2 = zeros(1, nfft);                % Will hold averaged periodogram
for r = 1:L
    x = x_all(r,:);                 % One realization
    X = fft(x, nfft);               % Discrete-time Fourier transform of that realization
    % Periodogram: (1/N) * |Σ x(n)e^{-jωn}|^2, evaluated at DFT bins
    P2 = P2 + (1/N)*abs(X).^2;
end
P2 = P2 / L;                        % Average over all L realizations (≈ expectation)

%% ============== PLOTS: ACF AND PSDs ==============

% -------- ACF plot --------
figure;
stem(lags, r_est, '.'); grid on;
xlabel('Lag k');
ylabel('r(k)');
title(sprintf('Estimated ACF r(k) for %s run', process_name));
xlim([-100 100]);                   % Zoom into small lags to visualize decay

% -------- PSD comparison plot --------
figure;
% PSD from DTFT of ACF (Definition 1)
plot(omega, 10*log10(P1+eps), 'LineWidth', 1.5); hold on;
% PSD from averaged periodogram (Definition 2)
plot(omega, 10*log10(P2+eps), '--', 'LineWidth', 1.3);

if use_theoretical
    % If we have a closed-form PSD (e.g. AR(1)), overlay it
    P_theo = P_theo_fun(omega);
    plot(omega, 10*log10(P_theo+eps), ':', 'LineWidth', 1.3);
    legend('DTFT of ACF (Equation 7)', 'Avg. periodogram (Equation 9)', ...
           'Theoretical PSD', 'Location','Best');
else
    legend('DTFT of ACF (Equation 7)', 'Avg. periodogram (Equation 9)', ...
           'Location','Best');
end

xlabel('\omega (rad/sample)');
ylabel('PSD / power (dB)');
title(sprintf('PSD estimates for %s run', process_name));
grid on;

%% ============== OPTIONAL: DECAY CONDITION TERM (11) ==============
% We approximate: (1/N) Σ_{k=-(N-1)}^{N-1} |k| |r(k)| for several N values.
% If this term → 0 as N grows, the WK condition is satisfied.

Ns  = [100, 200, 400, 800, 1600];   % Different N values for the condition check
val = zeros(size(Ns));              % Will store the condition value for each N

for i = 1:length(Ns)
    Ni = Ns(i);                     % Current N
    k  = -(Ni-1):(Ni-1);            % Lags from -(Ni-1) to +(Ni-1)

    if use_theoretical
        % If we know theoretical r(k) (e.g. AR(1)), use it
        r_use = r_theo_fun(k);
    else
        % Otherwise, use empirical r_est sliced around 0 lag
        mid = (length(r_est)+1)/2;          % index for lag 0
        idx = mid-(Ni-1):mid+(Ni-1);        % indices for lags -(Ni-1)..(Ni-1)
        r_use = r_est(idx);
    end

    % Compute (1/N) * Σ |k| |r(k)|
    val(i) = (1/Ni) * sum(abs(k).*abs(r_use));
end

figure;
plot(Ns, val, 'o-', 'LineWidth', 1.5); grid on;
xlabel('N');
ylabel('Error');
title(sprintf('Decay-condition term for %s run', process_name));
