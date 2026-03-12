%% Core part

clc;
clear all;
close all;

% A   = 1.0;              % amplitude
% f   = 1;                % Hz
% phi = pi/3;        % phase in radians
% fs  = 200;              % sampling frequency (Hz)
% N   = 0:200-1;             % number of samples
% t = N/fs;

% Synthetic signal: x[n] = A cos(2π f t + φ)
% Part 1.5.1
% x = A * cos(2*pi*f*t + phi);
% Part 1.5.2
% x = cos(2*pi*8*t) + 0.5*cos(2*pi*20*t);
% Part 1.5.4
% fs = 100;
% N = 0:1000-1;
% t = N/fs;
% 
% x1 = sin(2*pi*1.2*t);
% x2 = sin(2*pi*1.25*t);

%% Parameters
T  = 8;              % duration (s)
fs = 256;            % sampling frequency (Hz)
dt = 1/fs;
t  = 0:dt:T-dt;      % time vector (length = T*fs)

% Frequencies
f_delta = 2;         % Hz (δ-wave)
f_alpha = 10;        % Hz (α-wave)

A_delta = 5 - 4 ./ (1 + exp(-5*(t - 2)));   % Aδ(t)
A_alpha = 1 + 4 ./ (1 + exp(-5*(t - 6)));   % Aα(t)

x_delta = A_delta .* cos(2*pi*f_delta*t);
x_alpha = A_alpha .* cos(2*pi*f_alpha*t);

sigma_eta = 0.8;
eta = sigma_eta * randn(size(t));

x = x_delta + x_alpha + eta;

% Part 1.5.5 segmentation

t_cp1 = 2.5;
t_cp2 = 5.7;

% Convert changepoints to indices (first sample with t >= t_cp)
idx_cp1 = find(t >= t_cp1, 1, 'first');
idx_cp2 = find(t >= t_cp2, 1, 'first');

% Three segments:
% 1: [0, 2.5 s)
% 2: [2.5, 6.7 s)
% 3: [6.7, end]
x_seg1 = x(1:idx_cp1-1);
x_seg2 = x(idx_cp1:idx_cp2-1);
x_seg3 = x(idx_cp2:end);

% Apply iAAFT separately to each quasi-stationary segment
s_seg1 = iAAFT(x_seg1);
s_seg2 = iAAFT(x_seg2);
s_seg3 = iAAFT(x_seg3);

% Concatenate into a "nonstationary surrogate”
x_nsurr = [s_seg1(:); s_seg2(:); s_seg3(:)];
t_ns    = (0:length(x_nsurr)-1)/fs;

% 
% figure; 
% plot(t, x);
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('1 Hz sinusoid, A = 1, phase = 60^\circ');
% IAAFT Algorithm
% x_surr = iAAFT(x);
% x2_surr = iAAFT(x2);


%% 2x2 Visualization of IAAFT result PART 1.5.2

% Frequency axis for magnitude spectrum
% Nsig = length(x);
% k    = 0:Nsig-1;
% faxis = k*fs/Nsig;    % 0..fs-Δf
% 
% X_orig  = abs(fft(x));
% X_surr  = abs(fft(x_surr));
% 
% % X2_orig  = abs(fft(x2));
% % X2_surr  = abs(fft(x2_surr));
% 
% figure('Position',[100 100 1200 700]);
% tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
% 
% % (a) Top-left: Original signal in time domain
% nexttile;
% plot(t, x1, 'b');hold on;
% plot(t, x2, 'r');
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Original signal (time domain)');
% legend({'Original-1.2Hz','Original-1.25Hz'});
% 
% % (b) Top-right: Surrogate signal in time domain
% nexttile;
% plot(t, x1_surr, 'b');hold on;
% plot(t, x2_surr, 'r');
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Surrogate signal (time domain)');
% legend({'Surrogate-1.2Hz','Surrogate-1.25Hz'});
% 
% % (c) Bottom-left: Magnitude spectrum comparison (overlay)
% nexttile;
% stem(faxis, X1_orig, 'b', 'LineWidth', 1.2); hold on;
% stem(faxis, X1_surr, 'b--', 'LineWidth', 1.2);hold on;
% stem(faxis, X2_orig, 'r', 'LineWidth', 1.2); hold on;
% stem(faxis, X2_surr, 'r--', 'LineWidth', 1.2);              % single-sided view
% xlabel('Frequency (Hz)');
% ylabel('|X[k]|');
% xlim([0.5,2]);
% title('Magnitude spectrum: original vs surrogate');
% legend({'Original-1.2Hz','Surrogate-1.2Hz','Original-1.25Hz','Surrogate-1.25Hz'}, 'Location','best');
% grid on;

% % (d) Bottom-right: Amplitude distribution comparison (overlay histograms)
% bw = 0.02;  % bin width in amplitude units
% nexttile;
% hold on;
% histogram(x,      'BinWidth', bw, 'Normalization','pdf', ...
%           'FaceAlpha',0.4, 'EdgeColor','none');
% histogram(x_surr, 'BinWidth', bw, 'Normalization','pdf', ...
%           'FaceAlpha',0.4, 'EdgeColor','none');
% xlabel('Amplitude');
% ylabel('PDF (empirical)');
% title('Amplitude distribution: original vs surrogate');
% legend({'Original','Surrogate'}, 'Location','best');
% grid on;

% %% Autocovariance plot part 1.5.3
% fs = 200;         % sampling frequency (Hz)
% maxLag = 100;
% 
% % Remove mean for autocovariance
% x0      = x(:)      - mean(x);
% x_surr0 = x_surr(:) - mean(x_surr);
% 
% % Autocovariance via unbiased xcorr
% [Rx, lags]      = xcorr(x0,      maxLag, 'unbiased');
% [Rsurr, lags2]  = xcorr(x_surr0, maxLag, 'unbiased');
% 
% % Keep only nonnegative lags k = 0..maxLag
% rk_x     = Rx(lags >= 0);
% rk_surr  = Rsurr(lags2 >= 0);
% k        = 0:maxLag;
% 
% % Plot autocovariance comparison
% figure;
% stem(k, rk_x, 'b', 'filled'); hold on;
% stem(k, rk_surr, 'r--', 'filled');
% xlabel('Lag k');
% ylabel('Autocovariance r[k]');
% title('Autocovariance: original vs iAAFT surrogate');
% legend({'Original','Surrogate'}, 'Location','best');
% grid on;
% 
% %% Phase plot
% 
% fs = 200;
% N  = 200;
% n  = 0:N-1;
% t  = n/fs;
% 
% % Original signal (for completeness)
% x = cos(2*pi*8*t) + 0.5*cos(2*pi*20*t);
% 
% % FFTs
% X      = fft(x);
% X_surr = fft(x_surr);
% 
% % Bin indices for 8 Hz and 20 Hz (since fs=200, N=200 → Δf=1 Hz)
% k8  = 8;         % 8 Hz bin
% k20 = 20;        % 20 Hz bin
% 
% % Phases
% phi8_orig  = angle(X(k8+1));       % +1 for MATLAB 1-based indexing
% phi20_orig = angle(X(k20+1));
% 
% phi8_sur   = angle(X_surr(k8+1));
% phi20_sur  = angle(X_surr(k20+1));
% 
% % Magnitudes (for plotting as radius = 1 scaled)
% mag8_orig  = abs(X(k8+1));
% mag20_orig = abs(X(k20+1));
% mag8_sur   = abs(X_surr(k8+1));
% mag20_sur  = abs(X_surr(k20+1));
% 
% % Normalize radius to 1 for clean phase visualization
% r8_orig  = 1;
% r20_orig = 1;
% r8_sur   = 1;
% r20_sur  = 1;
% 
% figure;
% 
% % (a) Original: two frequency components in polar
% subplot(1,2,1);
% polarplot([0 phi8_orig],  [0 r8_orig],  'b-', 'LineWidth', 2); hold on;
% polarplot([0 phi20_orig], [0 r20_orig], 'r-', 'LineWidth', 2);
% rlim([0 1.2]);
% title('Original: phases at 8 Hz (blue) and 20 Hz (red)');
% 
% % (b) Surrogate: two frequency components in polar
% subplot(1,2,2);
% polarplot([0 phi8_sur],  [0 r8_sur],  'b-', 'LineWidth', 2); hold on;
% polarplot([0 phi20_sur], [0 r20_sur], 'r-', 'LineWidth', 2);
% rlim([0 1.2]);
% title('Surrogate: phases at 8 Hz (blue) and 20 Hz (red)');

%% Assume: x (original), x_surr (surrogate), fs = 256, t is time vector

% N  = length(x);
% t  = (0:N-1)/fs;
% 
% % FFT frequency axis
% X      = fft(x);
% X_nsurr = fft(x_nsurr);
% faxis  = (0:N-1)*fs/N;
% 
% % STFT parameters
% win     = hamming(128);
% noverlap = 64;
% nfft    = 256;
% 
% figure('Position',[100 100 1400 600]);
% tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
% 
% % (1,1) time domain
% nexttile;
% plot(t, x, 'b');
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Original: time domain');
% 
% % (1,2) magnitude spectrum (single-sided)
% nexttile;
% Xmag = abs(X);
% N2   = floor(N/2)+1;
% plot(faxis(1:N2), Xmag(1:N2), 'b');
% xlabel('Frequency (Hz)');
% ylabel('|X(f)|');
% title('Original: magnitude spectrum');
% xlim([0 20]);
% grid on;
% 
% % (1,3) spectrogram
% nexttile;
% spectrogram(x, win, noverlap, nfft, fs, 'yaxis');
% title('Original: spectrogram');
% ylabel('Frequency (Hz)');
% xlabel('Time (s)');
% colormap turbo;
% 
% % (2,1) time domain
% nexttile;
% plot(t, x_nsurr, 'b');
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Surrogate: time domain');
% 
% % (2,2) magnitude spectrum (single-sided)
% nexttile;
% Xmag_s = abs(X_nsurr);
% plot(faxis(1:N2), Xmag_s(1:N2), 'b');
% xlabel('Frequency (Hz)');
% ylabel('|X(f)|');
% title('Surrogate: magnitude spectrum');
% xlim([0 20]);
% grid on;
% 
% % (2,3) spectrogram
% nexttile;
% spectrogram(x_nsurr, win, noverlap, nfft, fs, 'yaxis');
% title('Surrogate: spectrogram');
% ylabel('Frequency (Hz)');
% xlabel('Time (s)');
% colormap turbo;






function x_surr = iAAFT(x)

thres = 0.1;
Nmax = 1000;
mse_last = +inf;
i = 0;
mse = 0;


    x_sort = sort(x);   % Target amplitude distribution
    X_fft = abs(fft(x));  % Target magnitude spectrum
    x_surr = x(randperm(length(x)));  % initial surrogate
    
    while (abs(mse - mse_last) > thres && (i < Nmax))
    
        mse_last = mse;
        % Spectra matching
        sk = fft(x_surr);
        phi_k = angle(sk);
        sk_prime  = X_fft.*exp(1j*phi_k);
        sn_prime = ifft(sk_prime,'symmetric');
    
        [s, idx] = sort(sn_prime);
        x_surr(idx) = x_sort;
    
        mse = mean(X_fft - abs(fft(x_surr)));
        i = i+ 1;
    end
end 

