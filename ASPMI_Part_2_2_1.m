%% ASPMI Part 2.1 – The Least Mean Square (LMS) Algorithm
clc; clear; close all;

a1    = 0.1;    % AR(2) coefficient 1
a2    = 0.8;    % AR(2) coefficient 2
sigma2_eta = 0.25;  % Noise variance (sigma_eta^2)
N     = 1000;   % Samples per realization
L     = 100;    % Number of independent realizations (for ensemble average)

%% (a) Correlation matrix R and convergence range for mu

% Compute r(1) in terms of r(0): r(1) = a1*r(0)/(1-a2)
% c = a1 / (1 - a2);          % r(1) = c * r(0)
% % r(2) = a1*r(1) + a2*r(0) = (a1*c + a2)*r(0)
% d = a1 * c + a2;             % r(2) = d * r(0)
% % Substitute into r(0) equation: r(0) = a1*c*r(0) + a2*d*r(0) + sigma2_eta
% % => r(0)*(1 - a1*c - a2*d) = sigma2_eta
% r0 = sigma2_eta / (1 - a1*c - a2*d);
% r1 = c * r0;
% 
% % Build 2x2 autocorrelation matrix
% R = [r0, r1; r1, r0];
% % 
% fprintf('Part (a): Correlation matrix R\n');
% disp(R);
% 
% lambda_max = max(eig(R));
% mu_max = 2 / lambda_max;
% fprintf('Largest eigenvalue of R: %.4f\n', lambda_max);
% fprintf('Convergence range: 0 < mu < %.4f\n\n', mu_max);


% %% (b) LMS adaptive predictor – squared error and learning curves

% mu_vals = [0.05, 0.01];  % Two step sizes to compare
% 
% e2_db_all  = zeros(L, N, length(mu_vals));   % squared error in dB
% e2_lin_all = zeros(L, N, length(mu_vals));   % squared error linear
% 
% %% LMS adaptive predictor simulations
% for m = 1:length(mu_vals)
%     mu = mu_vals(m);
%     for l = 1:L
%         % Generate AR(2) process of length N+2 (for initial conditions)
%         eta = sqrt(sigma2_eta) * randn(1, N+2);
%         x   = zeros(1, N+2);
%         for n = 3:N+2
%             x(n) = a1*x(n-1) + a2*x(n-2) + eta(n);
%         end
%         x = x(3:end);  % length N
% 
%         % LMS adaptive predictor (order M = 2)
%         w = zeros(2, 1);   % initial weights
% 
%         for n = 3:N
%             xv    = [x(n-1); x(n-2)];   % regressor
%             x_hat = w.' * xv;           % prediction
%             e     = x(n) - x_hat;       % error
% 
%             % Store squared error
%             e2_lin_all(l, n, m) = e^2;
%             e2_db_all(l,  n, m) = 10*log10(e^2 + eps);
% 
%             % LMS update
%             w = w + mu * e * xv;
%         end
%     end
% end
% 
% %% Learning curve plots (in dB)
% figure;
% for m = 1:length(mu_vals)
%     mu = mu_vals(m);
% 
%     subplot(1, 2, m);
%     % Single realization (first trial)
%     plot(1:N, squeeze(e2_db_all(1,:,m)), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
%     hold on;
% 
%     % Ensemble-averaged learning curve (dB)
%     lc_db = mean(squeeze(e2_db_all(:,:,m)), 1);
%     plot(1:N, lc_db, 'b', 'LineWidth', 1.5);
% 
%     xlabel('Time step n');
%     ylabel('10 \log_{10}(e^2(n)) (dB)');
%     title(sprintf('Learning Curve – \\mu = %.2f', mu));
%     legend({'Single realization', 'Ensemble average (L=100)'}, 'Location', 'best');
%     grid on;
% end
% sgtitle('LMS Adaptive Predictor: Squared Prediction Error');
% 
% %% Empirical J_min (optimal AR(2) predictor)
% N_long = 50000;
% eta_long = sqrt(sigma2_eta) * randn(1, N_long+2);
% x_long   = zeros(1, N_long+2);
% for n = 3:N_long+2
%     x_long(n) = a1*x_long(n-1) + a2*x_long(n-2) + eta_long(n);
% end
% x_long = x_long(3:end);
% 
% e_opt = zeros(1, N_long);
% for n = 3:N_long
%     x_hat_opt = a1*x_long(n-1) + a2*x_long(n-2);
%     e_opt(n)  = x_long(n) - x_hat_opt;
% end
% Jmin_emp = mean(e_opt(3:end).^2);
% 
% %% Correlation matrix R and its trace (from Part a)
% r0 = 0.9259;  % replace by your r(0)
% r1 = 0.4630;  % replace by your r(1)
% R  = [r0 r1; r1 r0];
% TrR = trace(R);
% 
% %% Misadjustment estimation
% fprintf('=== Part (c): Misadjustment ===\n');
% ss_region = max(3, round(0.5*N)) : N;   % steady-state region (last 50%)
% 
% for m = 1:length(mu_vals)
%     mu = mu_vals(m);
% 
%     % Theoretical misadjustment
%     M_theory = (mu / 2) * TrR;
% 
%     % Steady-state MSE from ensemble-avg linear errors
%     lc_lin = mean(squeeze(e2_lin_all(:,:,m)), 1);    % ensemble avg MSE
%     MSE_ss = mean(lc_lin(ss_region));
% 
%     % Empirical misadjustment (using empirical J_min)
%     M_est = (MSE_ss - Jmin_emp) / Jmin_emp;
% 
%     fprintf('mu = %.2f: Theoretical M = %.4f, Estimated M = %.4f\n', ...
%             mu, M_theory, M_est);
% end
% 
% % %% =========================================================
% % %% (d) Steady-state weight estimates
% % %% =========================================================
% fprintf('=== Part (d): Steady-state weight estimates ===\n');
% for m = 1:length(mu_vals)
%     mu = mu_vals(m);
%     w_all = zeros(L, 2);  % Store final weights for each trial
% 
%     for l = 1:L
%         eta = sqrt(sigma2_eta) * randn(1, N+2);
%         x   = zeros(1, N+2);
%         for n = 3:N+2
%             x(n) = a1*x(n-1) + a2*x(n-2) + eta(n);
%         end
%         x = x(3:end);
% 
%         w = zeros(2, 1);
%         % Run LMS and record the final weight vector
%         for n = 3:N
%             xv = [x(n-1); x(n-2)];
%             e  = x(n) - w' * xv;
%             w = w + mu * e * xv;
%         end
%         w_all(l, :) = w';
%     end
% 
%     % Average final weights over all L trials
%     w_mean = mean(w_all, 1);
%     fprintf('mu = %.2f: w1_est = %.4f (true %.1f), w2_est = %.4f (true %.1f)\n', ...
%         mu, w_mean(1), a1, w_mean(2), a2);
% end
% fprintf('Note: biased estimates are expected when signal variance is high.\n\n');
% 
% 
%% (f) Leaky LMS implementation

gamma_vals = [0.1, 0.5, 0.001];   % Leakage coefficients to test
mu_leaky   = 0.005;                  % Fixed step size for leaky LMS

fprintf('=== Part (f): Leaky LMS steady-state weights ===\n');
fprintf('True coefficients: a1 = %.1f, a2 = %.1f\n', a1, a2);

% Plot weight trajectories for different gamma values
figure;
colors = lines(length(gamma_vals));

for g = 1:length(gamma_vals)
    gamma = gamma_vals(g);
    w_all_leaky = zeros(L, 2);

    for l = 1:L
        eta = sqrt(sigma2_eta) * randn(1, N+2);
        x   = zeros(1, N+2);
        for n = 3:N+2
            x(n) = a1*x(n-1) + a2*x(n-2) + eta(n);
        end
        x = x(3:end);

        w = zeros(2, 1);
        w_traj = zeros(2, N);  % Weight trajectory (for first realization only)
        for n = 3:N
            xv = [x(n-1); x(n-2)];
            e = x(n) - w' * xv;
            w = (1 - mu_leaky*gamma) * w + mu_leaky * e * xv;
            w_traj(:, n) = w;
        end
        w_all_leaky(l, :) = w';
    end

    w_leaky_mean = mean(w_all_leaky, 1);
    fprintf('gamma = %.3f: w1_est = %.4f, w2_est = %.4f\n', gamma, w_leaky_mean(1), w_leaky_mean(2));

    % Plot weight trajectory for the last trial
    subplot(length(gamma_vals), 1, g);
    plot(1:N, w_traj(1,:), 'Color', colors(1,:), 'LineWidth', 1.2); hold on;
    plot(1:N, w_traj(2,:), 'Color', colors(2,:), 'LineWidth', 1.2);
    yline(a1, '--', 'Color', colors(1,:), 'LineWidth', 1);
    yline(a2, '--', 'Color', colors(2,:), 'LineWidth', 1);
    xlabel('Time step n');
    ylabel('Weight value');
    title(sprintf('Leaky LMS weight trajectories: \\gamma = %.3f, \\mu = %.2f', gamma, mu_leaky));
    legend({'w_1(n)', 'w_2(n)', 'True a_1', 'True a_2'}, 'Location', 'best');
    grid on;
end
sgtitle('Leaky LMS: Weight trajectories for different \gamma values');

fprintf('\nNote: The leaky LMS converges to a biased solution because the L2 penalty\n');
fprintf('in J2 shrinks the weights toward zero, not toward the true AR coefficients.\n');
