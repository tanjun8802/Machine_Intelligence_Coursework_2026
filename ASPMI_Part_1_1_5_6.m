%% Extra script for part 1.5.6

%% Script: eeg_twochannel_surrogates_xcorr.m

clc; clear; close all;

%% Settings
dataDir  = "C:\Users\user\Desktop\MSc_Applied_Machine_Learning\Repos\Adaptive Signal Processing and Machine Intelligence\EEG_Data\brainwave_samples";  % <-- change this
fs       = 100;                          % EEG sampling frequency (Hz)
maxLagSec = 1;                           % cross-corr lag window (seconds)
maxLag    = round(maxLagSec * fs);

%% List all .npy files
files  = dir(fullfile(dataDir, '*.npy'));
nFiles = numel(files);

if nFiles == 0
    error('No .npy files found in %s', dataDir);
end

% Choose how to call readNPY:
usePackage = false;  % true if you use npy_matlab.readNPY

if usePackage
    import npy_matlab.*;
end

for f = 1:nFiles
    fprintf('Processing file %d / %d: %s\n', f, nFiles, files(f).name);

    %% ---- Load EEG recording (2 channels) ----
    fname = files(f).name;
    fpath = fullfile(files(f).folder, fname);

    if usePackage
        data = npy_matlab.readNPY(fpath);
    else
        data = readNPY(fpath);  % assumes readNPY on path
    end

    % Ensure data is [T x 2]
    if size(data,1) < size(data,2)
        data = data.';  % transpose if [2 x T]
    end
    [T, C] = size(data);
    if C ~= 2
        warning('File %s: expected 2 channels, found %d.', files(f).name, C);
    end

    t = (0:T-1)/fs;

    % Channel labels (for clarity)
    chanNames = {'FPz-Cz','Pz-Oz'};

    %% ---- Generate iAAFT surrogate per channel ----
    data_surr = zeros(size(data));
    for c = 1:C
        x = data(:,c);
        data_surr(:,c) = iAAFT(x);   % your iAAFT function
    end


    %% ---- Cross-correlation between the two channels: original vs surrogate ----
    x_orig = data(:,1);       % FPz-Cz
    y_orig = data(:,2);       % Pz-Oz
    
    x_surr = data_surr(:,1);
    y_surr = data_surr(:,2);
    
    [c_orig, lags] = xcorr(x_orig, y_orig, maxLag, 'coeff');
    [c_surr, ~]    = xcorr(x_surr, y_surr, maxLag, 'coeff');
    
    %% ---- Single figure with 3x1 subplots ----
    figure('Name', sprintf('EEG original, surrogate, xcorr (%s)', fname), ...
           'Position',[100 100 1600 450]);
    
    % (1) Original time domain
    subplot(3,1,1);
    plot(t, data(:,1), 'b'); hold on;
    plot(t, data(:,2), 'r');
    xlabel('Time (s)');
    ylabel('Amplitude (\muV)');
    title(sprintf('Original EEG (%s)', fname), 'Interpreter','none');
    legend({'FPz-Cz','Pz-Oz'}, 'Location','best');
    grid on;
    
    % (2) Surrogate time domain
    subplot(3,1,2);
    plot(t, data_surr(:,1), 'b'); hold on;
    plot(t, data_surr(:,2), 'r');
    xlabel('Time (s)');
    ylabel('Amplitude (\muV)');
    title(sprintf('Surrogate EEG (%s)', fname), 'Interpreter','none');
    legend({'FPz-Cz sur','Pz-Oz sur'}, 'Location','best');
    grid on;
    
    % (3) Cross-correlation original vs surrogate
    subplot(3,1,3);
    plot(lags/fs, c_orig, 'b', 'LineWidth', 1.4); hold on;
    plot(lags/fs, c_surr, 'r--', 'LineWidth', 1.4);
    xlabel('Lag (s)');
    ylabel('Norm. cross-corr');
    title('FPz-Cz vs Pz-Oz xcorr');
    legend({'Original','Surrogate'}, 'Location','best');
    grid on;


end


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

