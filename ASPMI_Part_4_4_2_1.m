%% ASPMI Part 4.2.1 – Sinusoidal Signal Generation and Convolution
% This script covers:
%   (a) Auto-convolution of y1, and cross-convolutions of y1*y2, y1*y3
%   (b) A simple classifier using convolution to distinguish y1, y2, y3

clc; clear; close all;

%% =========================================================
%% Generate three sinusoidal signals
%% =========================================================
% Signal generation code provided in the coursework:
t  = ((2*pi)/100) : ((2*pi)/100) : 10*pi;   % Time vector

y1 = sin(t);           % Frequency 1 (reference)
y2 = sin(0.5 * t);     % Half frequency of y1
y3 = sin(4   * t);     % Four times frequency of y1

N1 = length(y1);       % All three have the same length

figure;
subplot(3,1,1); plot(t, y1); title('y1 = sin(t)');         xlabel('t'); grid on;
subplot(3,1,2); plot(t, y2); title('y2 = sin(0.5t)');      xlabel('t'); grid on;
subplot(3,1,3); plot(t, y3); title('y3 = sin(4t)');        xlabel('t'); grid on;
sgtitle('Three sinusoidal signals');

%% =========================================================
%% (a) Auto-convolution and cross-convolutions
%% =========================================================
% Auto-convolution of y1 with itself: conv(y1, y1)
% Cross-convolution y1 * y2, and y1 * y3

c_y1y1 = conv(y1, y1);   % Auto-convolution
c_y1y2 = conv(y1, y2);   % Cross-convolution: different frequency
c_y1y3 = conv(y1, y3);   % Cross-convolution: higher frequency

% Lag axes for the three convolution outputs
lag_auto  = -(N1-1) : (N1-1);   % All output lengths are 2*N1-1

figure;
subplot(3,1,1);
plot(lag_auto, c_y1y1, 'b', 'LineWidth', 1.2);
xlabel('Lag (samples)');
ylabel('Amplitude');
title('Auto-convolution: y1 \star y1');
grid on;

subplot(3,1,2);
plot(lag_auto, c_y1y2, 'r', 'LineWidth', 1.2);
xlabel('Lag (samples)');
ylabel('Amplitude');
title('Cross-convolution: y1 \star y2  (different frequency)');
grid on;

subplot(3,1,3);
plot(lag_auto, c_y1y3, 'g', 'LineWidth', 1.2);
xlabel('Lag (samples)');
ylabel('Amplitude');
title('Cross-convolution: y1 \star y3  (higher frequency)');
grid on;

sgtitle('Convolution results – auto and cross');

fprintf('=== Part (a): Observations ===\n');
fprintf('y1 ★ y1 (auto-conv): a large peak at lag 0 indicating self-similarity;\n');
fprintf('  the envelope follows a triangular shape decaying towards the edges.\n');
fprintf('y1 ★ y2: Lower-frequency oscillation in the convolution output;\n');
fprintf('  the two signals have different frequencies so the peak is weaker.\n');
fprintf('y1 ★ y3: High-frequency oscillation; convolution of signals with\n');
fprintf('  very different frequencies produces rapid oscillation with small amplitude.\n\n');
fprintf('This illustrates the matched-filter property of convolution:\n');
fprintf('max response occurs when the kernel and signal share the same frequency.\n\n');

%% =========================================================
%% (b) Simple frequency classifier using convolution
%% =========================================================
% Principle: treat y1, y2, y3 as kernels (templates / matched filters).
% For an unknown signal y, compute:
%   score_k = max(abs(conv(y, y_k)))
% The class with the highest score identifies the signal.
%
% This mimics a 1-layer CNN with fixed (template) kernels:
%   - Conv1D layer  →  computes cross-correlation with each template
%   - Abs           →  |output|  (similar to ReLU or magnitude response)
%   - MaxPool1d     →  takes maximum over the sequence
%   - argmax        →  assigns class label

function class_id = classify_signal(y, y1, y2, y3)
% CLASSIFY_SIGNAL  Classify input y as class 1 (y1), 2 (y2), or 3 (y3)
% using cross-correlation (matched filter) with each template signal.
%
%   y       - input signal to classify (1D vector)
%   y1,y2,y3- template signals (one per class)
%   class_id - predicted class (1, 2, or 3)

    % Cross-correlate (equivalent to conv with flipped kernel)
    score1 = max(abs(conv(y, fliplr(y1))));
    score2 = max(abs(conv(y, fliplr(y2))));
    score3 = max(abs(conv(y, fliplr(y3))));

    scores   = [score1, score2, score3];
    [~, class_id] = max(scores);   % Class with highest correlation score
end

% Test the classifier on the original signals (should predict correct class)
c1 = classify_signal(y1, y1, y2, y3);
c2 = classify_signal(y2, y1, y2, y3);
c3 = classify_signal(y3, y1, y2, y3);

fprintf('=== Part (b): Classifier results ===\n');
fprintf('y1 classified as class %d  (expected 1) – %s\n', c1, check(c1==1));
fprintf('y2 classified as class %d  (expected 2) – %s\n', c2, check(c2==2));
fprintf('y3 classified as class %d  (expected 3) – %s\n', c3, check(c3==3));

fprintf('\nComment on similarity with CNNs:\n');
fprintf('  - The three template signals y1, y2, y3 act as convolutional KERNELS.\n');
fprintf('  - conv(y, kernel)  corresponds to the 1D convolution (Conv1d) layer.\n');
fprintf('  - abs(.)           corresponds to the absolute-value / ReLU activation.\n');
fprintf('  - max(.)           corresponds to global MaxPool1d.\n');
fprintf('  - argmax over scores assigns the class, like Softmax + argmax.\n');
fprintf('  In CNNs, the kernels are LEARNED via backpropagation, while here\n');
fprintf('  they are manually set to the class prototypes (domain knowledge).\n');

%% =========================================================
%% Visualise convolution scores for all three inputs
%% =========================================================
signals      = {y1, y2, y3};
signal_names = {'y1', 'y2', 'y3'};
templates    = {y1, y2, y3};

figure;
for s = 1:3
    sig = signals{s};
    for k = 1:3
        templ = templates{k};
        subplot(3, 3, (s-1)*3 + k);
        plot(abs(conv(sig, fliplr(templ))), 'LineWidth', 1.0);
        title(sprintf('%s ★ kernel_%d', signal_names{s}, k));
        xlabel('Lag'); ylabel('|conv|');
        grid on;
    end
end
sgtitle('Abs. cross-correlation of each signal with each template kernel');

%% =========================================================
%% Helper function
%% =========================================================
function s = check(cond)
    if cond; s = 'CORRECT'; else; s = 'WRONG'; end
end
