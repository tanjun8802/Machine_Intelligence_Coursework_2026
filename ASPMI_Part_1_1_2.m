clc;
clear all;
close all;

load("EEG_Data\EEG_Data_Assignment1.mat")

nfft  = 5*fs;   % 5 DFT samples per Hz

% Normal Periodogram
[Pxx,f] = periodogram(POz, [], nfft, fs);

figure;
plot(f, 10*log10(Pxx/max(Pxx)));
xlabel('Frequency (Hz)');
ylabel('Normalized PSD (dB)');
title('Standard Periodogram of the EEG data');
xlim([0 80]);
grid on;

% Windowed Average Periodogram

win10 = 10*fs;
win5 = 5*fs;
win1 = 1*fs;

figure;
[P10,f] = pwelch(POz, win10, win10/2, nfft, fs);  
plot(f, 10*log10(P10/max(P10)));
xlim([0 100]);
xlabel('Frequency (Hz)'); ylabel('Normalized PSD (dB)');
title('Average Periodogram with 10s window size');
grid on;

figure;
[P5,f] = pwelch(POz, win5, win5/2, nfft, fs);  
plot(f, 10*log10(P5/max(P5)));
xlim([0 100]);
xlabel('Frequency (Hz)'); ylabel('Normalized PSD (dB)');
title('Average Periodogram with 5s window size');
grid on;

figure;
[P1,f] = pwelch(POz, win1, win1/2, nfft, fs);  
plot(f, 10*log10(P1/max(P1)));
xlim([0 100]);
xlabel('Frequency (Hz)'); ylabel('Normalized PSD (dB)');
title('Average Periodogram with 1s window size');
grid on;







