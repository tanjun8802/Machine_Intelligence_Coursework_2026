clc;
clear all;
close all;

% Task 1.3

n = 0:200;
noise = 0.2/sqrt(2)*(randn(size(n))+1j*randn(size(n)));
x = exp(1j*2*pi*0.3*n)+exp(1j*2*pi*0.32*n)+ noise;
nfft = 128;

% Signal Generation and Visualisation

% figure;
% subplot(2,2,1); plot(n, real(x)); title('Real Part of Complex Signal x');xlabel('n');  ylabel('Amplitude');
% subplot(2,2,2); plot(n, imag(x)); title('Imaginary Part of Complex Signal x'); xlabel('n'); ylabel('Amplitude');
% subplot(2,2,3); plot(n, abs(x)); title('Magnitude of Complex Signal |x|');xlabel('n');  ylabel('|x|');
% subplot(2,2,4); plot(n, unwrap(angle(x))); title('Phase \angle of Complex Signal x'); xlabel('n'); ylabel('Phase (rad)');
% 
% set(gcf, 'Position', [100, 100, 1200, 600]);
% % Reduce margins
% p = get(gcf, 'Position');
% set(gcf, 'PaperPositionMode', 'auto');

[Pxx,f] = periodogram(x, [], nfft, 1);

% figure;
% plot(f, 10*log10(Pxx));
% xlabel('Frequency (Hz)');
% ylabel('Normalized PSD (dB)');
% title('Periodogram of signal x when N = 200');
% xlim([0 0.6]);
% grid on;
 
figure;
[X,R] = corrmtx(x,14,'mod');
[S,F] = pmusic(R,2,[ ],1,'corr');
plot(F,S,'linewidth',2); set(gca,'xlim',[0.25 0.40]);
title('Peak Identification using the MUSIC method');
grid on; xlabel('Hz'); ylabel('Pseudospectrum');
