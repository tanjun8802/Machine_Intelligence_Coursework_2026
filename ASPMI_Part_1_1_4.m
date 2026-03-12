clc;
clear all;
close all;

load("Data\RRI-DATA.mat")

fs = 500;

% RRI signals (column vectors)
rri = {xRRI1(:), xRRI2(:), xRRI3(:)};

winLenSec = [50, 150, 200];

for k = 1:numel(rri)
    x = rri{k};
    
    figure;
    
    for w = 1:numel(winLenSec)
        nperseg = round(winLenSec(w) * fs);
        nperseg = min(nperseg, numel(x));
        
        noverlap = floor(nperseg/2); 
        window   = hamming(nperseg);
        
        [Pxx,f] = pwelch(x, window, noverlap, [], fs);
        
        subplot(1, numel(winLenSec), w);
        plot(f, 10*log10(Pxx/max(Pxx)), 'LineWidth', 1.2);
        grid on;
        xlim([0,100]);
        xlabel('Frequency (Hz)');
        ylabel('PSD (dB/Hz), Normalised');
        title(sprintf('Trial %d, window = %d s', k, winLenSec(w)));
    end
end




