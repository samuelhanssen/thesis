clear
clc

A = [0.7 0.2;
     0.1 0.9];

B = [1;
     0.5];

C = [1 0];

nx = size(A,1);
nu = size(B,2);
ny = size(C,1);

Tini = 10;
N = 40;

noiseStd = 0.3;

rng(1)

Tdata = 100;

U = sign(randn(Tdata,nu));

x = randn(nx,1);

Y = zeros(Tdata,ny);

for k = 1:Tdata

    Y(k,:) = (C*x + noiseStd*randn(ny,1))';

    x = A*x + B*U(k,:)'+noiseStd*randn(nx,1);

end

UT = U.';
YT = Y.';

Tcols = Tdata - Tini - N + 1;

[Up,Yp,Uf,Yf] = Hankel_func(N,Tini,Tcols,nu,ny,UT,YT);

Theta = Yf*pinv([Up;Yp;Uf]);

nup = size(Up,1);
nyp = size(Yp,1);

P1 = Theta(:,1:nup);
P2 = Theta(:,nup+1:nup+nyp);
Gamma = Theta(:,nup+nyp+1:end);

Nmarkov = Tini + N + nx + 50;

G = cell(Nmarkov,1);

for k = 1:Nmarkov
    G{k} = C*A^(k-1)*B;
end

Tp = zeros(Tini*ny,Tini*nu);

for i = 2:Tini
    for j = 1:i-1

        Tp((i-1)*ny+1:i*ny,...
           (j-1)*nu+1:j*nu) = G{i-j};

    end
end

Tf = zeros(N*ny,N*nu);

for i = 2:N
    for j = 1:i-1

        Tf((i-1)*ny+1:i*ny,...
           (j-1)*nu+1:j*nu) = G{i-j};

    end
end

Tfp = zeros(N*ny,Tini*nu);

for i = 1:N
    for j = 1:Tini

        idx = Tini + i - j;

        if idx >= 1

            Tfp((i-1)*ny+1:i*ny,...
                (j-1)*nu+1:j*nu) = G{idx};

        end

    end
end

Hh = zeros(Tini*ny,Tini*nu);

for i = 1:Tini
    for j = 1:Tini

        Hh((i-1)*ny+1:i*ny,...
           (j-1)*nu+1:j*nu) = G{i+j-1};

    end
end

Hf = zeros(N*ny,Tini*nu);

for i = 1:N
    for j = 1:Tini

        Hf((i-1)*ny+1:i*ny,...
           (j-1)*nu+1:j*nu) = G{Tini+i+j-1};

    end
end

Tfull = [Tp zeros(Tini*ny,N*nu);
         Tfp Tf];

Hfull = [Hh;
         Hf];

Cc = zeros(nx,nx*nu);

for k = 1:nx

    cols = (k-1)*nu + (1:nu);

    Cc(:,cols) = A^(k-1)*B;

end

Nsamp = 100;

err_markov = zeros(Nsamp,1);
err_spc = zeros(Nsamp,1);

for s = 1:Nsamp

    alpha0 = randn(nx*nu,1);

    x0 = Cc*alpha0;

    uini_mat = sign(randn(Tini,nu));
    uf_mat   = sign(randn(N,nu));

    x = x0;

    yini_meas = zeros(Tini,ny);

    for k = 1:Tini

        yini_meas(k,:) = (C*x + noiseStd*randn(ny,1))';

        x = A*x + B*uini_mat(k,:)' + noiseStd*randn(nx,1);

    end

    xk = x;

    x = xk;

    yf_true = zeros(N,ny);

    for k = 1:N

        yf_true(k,:) = (C*x)';

        x = A*x + B*uf_mat(k,:)' + noiseStd*randn(nx,1);

    end

    uini = reshape(uini_mat.',[],1);
    yini = reshape(yini_meas.',[],1);
    uf   = reshape(uf_mat.',[],1);

    b = yini - Tp*uini;

    alpha = pinv(Hh)*b;

    spc_future = Gamma*uf + P1*uini + P2*yini;

    markov_future = Tfp*uini + Tf*uf + Hf*alpha;

    yf_true_vec = reshape(yf_true.',[],1);

    err_spc(s) = norm(spc_future-yf_true_vec);

    err_markov(s) = norm(markov_future-yf_true_vec);

end

fprintf('\n');
fprintf('Noise standard deviation = %.4f\n',noiseStd);

fprintf('\n');
fprintf('Mean SPC prediction error     = %.16e\n',...
        mean(err_spc));

fprintf('Mean Markov prediction error  = %.16e\n',...
        mean(err_markov));

fprintf('\n');
fprintf('Max SPC prediction error      = %.16e\n',...
        max(err_spc));

fprintf('Max Markov prediction error   = %.16e\n',...
        max(err_markov));

alpha0 = randn(nx*nu,1);

x0 = Cc*alpha0;

uini_mat = sign(randn(Tini,nu));
uf_mat   = sign(randn(N,nu));

x = x0;

yini_true = zeros(Tini,ny);

for k = 1:Tini

    yini_true(k,:) = (C*x)'+ noiseStd*randn(ny,1);

    x = A*x + B*uini_mat(k,:)'+ noiseStd*randn(nx,1);

end

xk = x;

x = xk;

yf_true = zeros(N,ny);

for k = 1:N

    yf_true(k,:) = (C*x)'+ noiseStd*randn(ny,1);

    x = A*x + B*uf_mat(k,:)'+ noiseStd*randn(nx,1);

end

yini_meas = yini_true + noiseStd*randn(size(yini_true));

uini = reshape(uini_mat.',[],1);
yini = reshape(yini_meas.',[],1);
uf   = reshape(uf_mat.',[],1);

b = yini - Tp*uini;

alpha = pinv(Hh)*b;

spc_future = Gamma*uf + P1*uini + P2*yini;

markov_future = Tfp*uini + Tf*uf + Hf*alpha;

yf_true_vec = reshape(yf_true.',[],1);

fprintf('\n');
fprintf('Single-shot SPC error    = %.16e\n',...
        norm(spc_future-yf_true_vec));

fprintf('Single-shot Markov error = %.16e\n',...
        norm(markov_future-yf_true_vec));

timeFuture = 1:N;

figure

subplot(3,1,1)
plot(timeFuture,yf_true_vec,'k','LineWidth',2)
hold on
plot(timeFuture,spc_future,'b--','LineWidth',2)
grid on
legend('Ground truth','SPC')
title('SPC prediction vs ground truth')

subplot(3,1,2)
plot(timeFuture,yf_true_vec,'k','LineWidth',2)
hold on
plot(timeFuture,markov_future,'r--','LineWidth',2)
grid on
legend('Ground truth','Markov')
title('Markov prediction vs ground truth')

subplot(3,1,3)
plot(timeFuture,yf_true_vec-spc_future,'b','LineWidth',2)
hold on
plot(timeFuture,yf_true_vec-markov_future,'r--','LineWidth',2)
grid on
legend('Ground truth - SPC',...
       'Ground truth - Markov')
title('Prediction errors')

function [Up,Yp,Uf,Yf] = Hankel_func(N,Tini,T,nu,ny,U,Y)

L = Tini + N;

Hu = zeros(L*nu,T);
Hy = zeros(L*ny,T);

for k = 1:T

    Hu(:,k) = reshape(U(:,k:k+L-1),[],1);
    Hy(:,k) = reshape(Y(:,k:k+L-1),[],1);

end

Up = Hu(1:Tini*nu,:);
Uf = Hu(Tini*nu+1:end,:);

Yp = Hy(1:Tini*ny,:);
Yf = Hy(Tini*ny+1:end,:);

end


%%

figure

subplot(2,1,1)

plot(spc_future,'b','LineWidth',2)
hold on
plot(markov_future,'r--','LineWidth',2)

grid on
legend('SPC','Markov')
title('SPC vs Markov')

subplot(2,1,2)

plot(spc_future-markov_future,'k','LineWidth',2)

grid on
title(sprintf('SPC - Markov (norm = %.4e)', ...
    norm(spc_future-markov_future)))
xlabel('Prediction step')

%%

mse_spc = mean((spc_future - yf_true_vec).^2);
mse_markov = mean((markov_future - yf_true_vec).^2);

mae_spc = mean(abs(spc_future - yf_true_vec));
mae_markov = mean(abs(markov_future - yf_true_vec));

fprintf('\n');
fprintf('SPC MSE    = %.16e\n', mse_spc);
fprintf('Markov MSE = %.16e\n', mse_markov);

fprintf('SPC MAE    = %.16e\n', mae_spc);
fprintf('Markov MAE = %.16e\n', mae_markov);

figure

subplot(1,2,1)

bar([mse_spc mse_markov])
set(gca,'XTickLabel',{'SPC','Markov'})
ylabel('MSE')
title('Mean Squared Error')

subplot(1,2,2)

bar([mae_spc mae_markov])
set(gca,'XTickLabel',{'SPC','Markov'})
ylabel('MAE')
title('Mean Absolute Error')

%%

%% ==========================================================
%  Test theoretical equivalence:
%
%  Gamma = Tf
%  P2    = Hf*pinv(Hh)
%  P1    = Tfp - P2*Tp
% ===========================================================

P2_theory = Hf*pinv(Hh);

P1_theory = Tfp - P2_theory*Tp;

Gamma_theory = Tf;

fprintf('\n');
fprintf('=========================================\n');
fprintf('Theoretical matrix equivalence checks\n');
fprintf('=========================================\n');

fprintf('||Gamma - Tf||         = %.16e\n', ...
        norm(Gamma - Gamma_theory,'fro'));

fprintf('||P2 - Hf*pinv(Hh)||   = %.16e\n', ...
        norm(P2 - P2_theory,'fro'));

fprintf('||P1 - (Tfp-P2*Tp)||   = %.16e\n', ...
        norm(P1 - P1_theory,'fro'));

fprintf('\n');

fprintf('Relative errors:\n');

fprintf('Gamma : %.16e\n', ...
        norm(Gamma-Gamma_theory,'fro')/norm(Gamma,'fro'));

fprintf('P2    : %.16e\n', ...
        norm(P2-P2_theory,'fro')/norm(P2,'fro'));

fprintf('P1    : %.16e\n', ...
        norm(P1-P1_theory,'fro')/norm(P1,'fro'));

%%

%% Recover LS-optimal SPC matrices from Markov predictor

P2_eff = Hf*pinv(Hh);
P1_eff = Tfp - P2_eff*Tp;

Yf_markov_dataset = ...
      P1_eff*Up ...
    + P2_eff*Yp ...
    + Tf*Uf;

Theta_markov = ...
    (Yf_markov_dataset - Tf*Uf) ...
    * pinv([Up;Yp]);

nup = size(Up,1);

P1_ls = Theta_markov(:,1:nup);
P2_ls = Theta_markov(:,nup+1:end);

fprintf('\n');
fprintf('Recover SPC matrices from Markov predictor\n');

fprintf('||P1_ls*Up + P2_ls*Yp - (Yf_markov-Tf*Uf)|| = %.4e\n', ...
    norm(P1_ls*Up + P2_ls*Yp ...
       - (Yf_markov_dataset - Tf*Uf),'fro'));

fprintf('||P1_ls-P1||_F = %.4e\n', ...
    norm(P1_ls-P1,'fro'));

fprintf('||P2_ls-P2||_F = %.4e\n', ...
    norm(P2_ls-P2,'fro'));

%%

%% ==========================================================
% Nullspace test for SPC predictor matrices
%% ==========================================================

Phi = [Up;Yp;Uf];

Nphi = null(Phi.');

fprintf('\n');
fprintf('Nullspace dimension of Phi'' = %d\n', size(Nphi,2));

if ~isempty(Nphi)

    % take random nullspace direction
    z = randn(size(Nphi,2),1);

    dtheta = Nphi*z;

    % reshape into perturbation of predictor matrix
    Delta = 1e-2*randn(size(Yf,1),1)*dtheta.';

    % split into blocks
    dP1 = Delta(:,1:size(Up,1));
    dP2 = Delta(:,size(Up,1)+(1:size(Yp,1)));
    dG  = Delta(:,size(Up,1)+size(Yp,1)+(1:size(Uf,1)));

    % perturb predictor
    P1p = P1 + dP1;
    P2p = P2 + dP2;
    Gp  = Gamma + dG;

    Yf_nom = P1*Up + P2*Yp + Gamma*Uf;
    Yf_pert = P1p*Up + P2p*Yp + Gp*Uf;

    fprintf('||Delta*Phi||_F      = %.4e\n', ...
        norm(Delta*Phi,'fro'));

    fprintf('Prediction mismatch  = %.4e\n', ...
        norm(Yf_nom - Yf_pert,'fro'));

    fprintf('Relative mismatch    = %.4e\n', ...
        norm(Yf_nom-Yf_pert,'fro')/norm(Yf_nom,'fro'));

else
    fprintf('No nullspace found.\n');
end

%% Perturb Markov parameters in null directions


% Use one prediction scenario already available
y_nom = markov_future;

% Perturb one Markov parameter
epsi = 1e-3;

Gpert = G;
Gpert{5} = Gpert{5} + epsi;

% Rebuild only affected matrices
Tp_p = Tp;
Tf_p = Tf;
Tfp_p = Tfp;
Hh_p = Hh;
Hf_p = Hf;

for i = 2:Tini
    for j = 1:i-1
        if i-j == 5
            Tp_p((i-1)*ny+1:i*ny,...
                 (j-1)*nu+1:j*nu) = Gpert{5};
        end
    end
end

for i = 2:N
    for j = 1:i-1
        if i-j == 5
            Tf_p((i-1)*ny+1:i*ny,...
                 (j-1)*nu+1:j*nu) = Gpert{5};
        end
    end
end

alpha_p = pinv(Hh_p)*(yini-Tp_p*uini);

y_pert = Tfp_p*uini + Tf_p*uf + Hf_p*alpha_p;

fprintf('\n');
fprintf('Markov perturbation test\n');
fprintf('Prediction change = %.4e\n', ...
    norm(y_nom-y_pert));


%%
clear
clc
close all

rng(1)

nx = 8;
nu = 3;
ny = 1;

A = randn(nx);
A = 0.95*A/max(abs(eig(A)));

B = randn(nx,nu);
C = randn(ny,nx);
D = zeros(ny,nu);

Tini = 8;
N = 25;

Tdata = 15000;

noiseStd = 0;

U = sign(randn(Tdata,nu));

x = randn(nx,1);

Y = zeros(Tdata,ny);

for k = 1:Tdata

    Y(k,:) = (C*x + D*U(k,:)' + noiseStd*randn(ny,1))';

    x = A*x + B*U(k,:)' + noiseStd*randn(nx,1);

end

UT = U.';
YT = Y.';

Tcols = Tdata - Tini - N + 1;

[Up,Yp,Uf,Yf] = Hankel_func( ...
    N,Tini,Tcols,nu,ny,UT,YT);

Theta = Yf * pinv([Up;Yp;Uf]);

nup = size(Up,1);
nyp = size(Yp,1);

P1_data = Theta(:,1:nup);

P2_data = Theta(:, ...
    nup+1:nup+nyp);

Gamma_data = Theta(:, ...
    nup+nyp+1:end);

Nmarkov = Tini + N + nx + 50;

G = cell(Nmarkov,1);

G{1} = D;

for k = 2:Nmarkov

    G{k} = C*A^(k-2)*B;

end

Tp = zeros(Tini*ny,Tini*nu);

for i = 1:Tini
    for j = 1:Tini

        kk = i-j+1;

        if kk >= 1

            Tp((i-1)*ny+1:i*ny,...
               (j-1)*nu+1:j*nu) = G{kk};

        end

    end
end

Tf = zeros(N*ny,N*nu);

for i = 1:N
    for j = 1:N

        kk = i-j+1;

        if kk >= 1

            Tf((i-1)*ny+1:i*ny,...
               (j-1)*nu+1:j*nu) = G{kk};

        end

    end
end

Tfp = zeros(N*ny,Tini*nu);

for i = 1:N
    for j = 1:Tini

        kk = Tini + i - j + 1;

        if kk >= 1

            Tfp((i-1)*ny+1:i*ny,...
                (j-1)*nu+1:j*nu) = G{kk};

        end

    end
end

Hh = zeros(Tini*ny,nx);

for i = 1:Tini

    Hh((i-1)*ny+1:i*ny,:) = C*A^(i-1);

end

Hf = zeros(N*ny,nx);

for i = 1:N

    Hf((i-1)*ny+1:i*ny,:) = C*A^(Tini+i-1);

end

P2_theory = Hf*pinv(Hh);

P1_theory = Tfp - P2_theory*Tp;

Gamma_theory = Tf;

fprintf('\n');
fprintf('========================================\n');
fprintf('DIRECT DATA vs MARKOV/THEORY\n');
fprintf('========================================\n');

fprintf('||Gamma_data-Gamma_theory||_F = %.4e\n',...
    norm(Gamma_data-Gamma_theory,'fro'));

fprintf('||P1_data-P1_theory||_F      = %.4e\n',...
    norm(P1_data-P1_theory,'fro'));

fprintf('||P2_data-P2_theory||_F      = %.4e\n',...
    norm(P2_data-P2_theory,'fro'));

fprintf('\n');

fprintf('Relative errors\n');

fprintf('Gamma : %.4e\n',...
    norm(Gamma_data-Gamma_theory,'fro') ...
    / norm(Gamma_theory,'fro'));

fprintf('P1    : %.4e\n',...
    norm(P1_data-P1_theory,'fro') ...
    / norm(P1_theory,'fro'));

fprintf('P2    : %.4e\n',...
    norm(P2_data-P2_theory,'fro') ...
    / norm(P2_theory,'fro'));

fprintf('\n');
fprintf('========================================\n');
fprintf('CHECK P2*Hh = Hf\n');
fprintf('========================================\n');

R = P2_data*Hh - Hf;

fprintf('||P2_data*Hh - Hf||_F = %.16e\n',...
    norm(R,'fro'));

fprintf('Relative error        = %.16e\n',...
    norm(R,'fro')/norm(Hf,'fro'));

fprintf('Max abs entry         = %.16e\n',...
    max(abs(R(:))));

DeltaP2 = P2_data - P2_theory;

fprintf('||P2_data-P2_theory||_F = %.16e\n',...
    norm(DeltaP2,'fro'));

fprintf('||DeltaP2*Hh||_F        = %.16e\n',...
    norm(DeltaP2*Hh,'fro'));

fprintf('Contraction ratio      = %.16e\n',...
    norm(DeltaP2*Hh,'fro')/norm(DeltaP2,'fro'));

fprintf('\n');
fprintf('========================================\n');
fprintf('RANDOM PREDICTION TEST\n');
fprintf('========================================\n');

Nsamp = 100;

predErr = zeros(Nsamp,1);

for s = 1:Nsamp

    x0 = randn(nx,1);

    uini_mat = sign(randn(Tini,nu));
    uf_mat   = sign(randn(N,nu));

    x = x0;

    yini_meas = zeros(Tini,ny);

    for k = 1:Tini

        yini_meas(k,:) = ...
            (C*x + noiseStd*randn(ny,1))';

        x = A*x ...
          + B*uini_mat(k,:)' ...
          + noiseStd*randn(nx,1);

    end

    yf_true = zeros(N,ny);

    for k = 1:N

        yf_true(k,:) = (C*x)';

        x = A*x ...
          + B*uf_mat(k,:)' ...
          + noiseStd*randn(nx,1);

    end

    uini_vec = reshape(uini_mat.',[],1);
    yini_vec = reshape(yini_meas.',[],1);
    uf_vec   = reshape(uf_mat.',[],1);

    yf_spc = ...
          P1_data*uini_vec ...
        + P2_data*yini_vec ...
        + Gamma_data*uf_vec;

    yf_true_vec = reshape(yf_true.',[],1);

    predErr(s) = norm(yf_spc-yf_true_vec);

end

fprintf('Mean prediction error = %.4e\n',...
    mean(predErr));

fprintf('Max prediction error  = %.4e\n',...
    max(predErr));

tol = 1e-8;

if norm(P1_data-P1_theory,'fro') < tol && ...
   norm(P2_data-P2_theory,'fro') < tol && ...
   norm(Gamma_data-Gamma_theory,'fro') < tol

    fprintf('\nSUCCESS:\n');
    fprintf('SPC matrices recovered from data ');
    fprintf('match Toeplitz/Hankel theory.\n');

else

    fprintf('\nWARNING:\n');
    fprintf('Mismatch detected.\n');

end


Phi = [Up;Yp;Uf];

Theta_theory = [P1_theory P2_theory Gamma_theory];

Yf_theory = Theta_theory*Phi;

Theta_ls = Yf_theory*pinv(Phi);

P1_ls = Theta_ls(:,1:size(Up,1));

P2_ls = Theta_ls(:, ...
         size(Up,1)+1 : ...
         size(Up,1)+size(Yp,1));

Gamma_ls = Theta_ls(:, ...
            size(Up,1)+size(Yp,1)+1:end);

fprintf('\n');
fprintf('LS-projected theory vs data\n');

fprintf('||Gamma_data-Gamma_ls|| = %.4e\n', ...
    norm(Gamma_data-Gamma_ls,'fro'));

fprintf('||P1_data-P1_ls||      = %.4e\n', ...
    norm(P1_data-P1_ls,'fro'));

fprintf('||P2_data-P2_ls||      = %.4e\n', ...
    norm(P2_data-P2_ls,'fro'));


predErrSPC = zeros(Nsamp,1);
predErrMarkov = zeros(Nsamp,1);
predMismatch = zeros(Nsamp,1);

for s = 1:Nsamp

    x0 = randn(nx,1);

    uini_mat = sign(randn(Tini,nu));
    uf_mat   = sign(randn(N,nu));

    x = x0;

    yini_meas = zeros(Tini,ny);

    for k = 1:Tini

        yini_meas(k,:) = ...
            (C*x + noiseStd*randn(ny,1))';

        x = A*x ...
          + B*uini_mat(k,:)' ...
          + noiseStd*randn(nx,1);

    end

    yf_true = zeros(N,ny);

    for k = 1:N

        yf_true(k,:) = (C*x)';

        x = A*x ...
          + B*uf_mat(k,:)' ...
          + noiseStd*randn(nx,1);

    end

    uini_vec = reshape(uini_mat.',[],1);
    yini_vec = reshape(yini_meas.',[],1);
    uf_vec   = reshape(uf_mat.',[],1);

    b = yini_vec - Tp*uini_vec;

    alpha = pinv(Hh)*b;

    yf_markov = ...
          Tfp*uini_vec ...
        + Tf*uf_vec ...
        + Hf*alpha;

    yf_spc = ...
          P1_data*uini_vec ...
        + P2_data*yini_vec ...
        + Gamma_data*uf_vec;

    yf_true_vec = reshape(yf_true.',[],1);

    predErrSPC(s) = ...
        norm(yf_spc - yf_true_vec);

    predErrMarkov(s) = ...
        norm(yf_markov - yf_true_vec);

    predMismatch(s) = ...
        norm(yf_spc - yf_markov);

end

fprintf('\n');
fprintf('========================================\n');
fprintf('SPC vs MARKOV PREDICTION TEST\n');
fprintf('========================================\n');

fprintf('Mean SPC error      = %.16e\n', ...
    mean(predErrSPC));

fprintf('Mean Markov error   = %.16e\n', ...
    mean(predErrMarkov));

fprintf('Mean SPC-Markov     = %.16e\n', ...
    mean(predMismatch));

fprintf('\n');

fprintf('Max SPC error       = %.16e\n', ...
    max(predErrSPC));

fprintf('Max Markov error    = %.16e\n', ...
    max(predErrMarkov));

fprintf('Max SPC-Markov      = %.16e\n', ...
    max(predMismatch));

fprintf('\n');

fprintf('Relative SPC-Markov = %.16e\n', ...
    mean(predMismatch)/mean(predErrSPC));



%% ============================================
% Prediction using ONLY Markov-derived matrices
%% ============================================

uini_vec = reshape(uini_mat.',[],1);
yini_vec = reshape(yini.',[],1);
uf_vec   = reshape(uf_mat.',[],1);

yf_markov_P = ...
      P1_theory*uini_vec ...
    + P2_theory*yini_vec ...
    + Gamma_theory*uf_vec;

fprintf('\n');
fprintf('Prediction using Markov-only P1,P2,Gamma\n');

fprintf('||yf_markov_P - yf_true|| = %.16e\n', ...
    norm(yf_markov_P - yf_true_vec));

fprintf('Relative error = %.16e\n', ...
    norm(yf_markov_P-yf_true_vec)/norm(yf_true_vec));

%%

yf_spc = ...
      P1_data*uini_vec ...
    + P2_data*yini_vec ...
    + Gamma_data*uf_vec;

fprintf('\n');
fprintf('SPC vs Markov-derived predictor\n');

fprintf('||yf_spc - yf_markov_P|| = %.16e\n', ...
    norm(yf_spc-yf_markov_P));

%%

P1_from_P2 = Tfp - P2_data*Tp;

fprintf('\n');
fprintf('Consistency check using DATA matrices\n');
fprintf('||P1_data - (Tfp - P2_data*Tp)||_F = %.16e\n', ...
    norm(P1_data - P1_from_P2,'fro'));

fprintf('Relative error = %.16e\n', ...
    norm(P1_data - P1_from_P2,'fro')/norm(P1_data,'fro'));



%%

clear
clc

rng(1)

% Simple system
A = [0.8 0.1;
     0.0 0.9];

B = [1;
     0.5];

C = [1 0];

nx = 2;
nu = 1;
ny = 1;

Tini = 8;
N    = 10;

% process noise gain
K = 0.05*eye(nx);

Tdata = 5000;

U = sign(randn(Tdata,nu));

x = randn(nx,1);

Y = zeros(Tdata,ny);

% generate data with process noise
for k = 1:Tdata

    Y(k,:) = (C*x)';

    e = randn(nx,1);

    x = A*x + B*U(k,:)' + K*e;

end

[Up,Yp,Uf,Yf] = Hankel_func( ...
    N,Tini,Tdata-Tini-N+1,...
    nu,ny,U.',Y.');

% SPC matrices from data
Theta = Yf*pinv([Up;Yp;Uf]);

nup = size(Up,1);
nyp = size(Yp,1);

P1 = Theta(:,1:nup);
P2 = Theta(:,nup+(1:nyp));
Gamma = Theta(:,nup+nyp+1:end);

% Markov parameters of deterministic plant
G = cell(Tini+N+20,1);

for k = 1:length(G)
    G{k} = C*A^(k-1)*B;
end

Tp = zeros(Tini,Tini);

for i = 2:Tini
    for j = 1:i-1
        Tp(i,j) = G{i-j};
    end
end

Tf = zeros(N,N);

for i = 2:N
    for j = 1:i-1
        Tf(i,j) = G{i-j};
    end
end

Tfp = zeros(N,Tini);

for i = 1:N
    for j = 1:Tini

        idx = Tini+i-j;

        if idx >= 1
            Tfp(i,j) = G{idx};
        end

    end
end

Hh = zeros(Tini,nx);

for i = 1:Tini
    Hh(i,:) = C*A^(i-1);
end

Hf = zeros(N,nx);

for i = 1:N
    Hf(i,:) = C*A^(Tini+i-1);
end

P2_theory = Hf*pinv(Hh);
P1_theory = Tfp - P2_theory*Tp;
Gamma_theory = Tf;

fprintf('\n');
fprintf('Process noise level = %.3f\n\n', K(1,1));

fprintf('||Gamma-Gamma_theory|| = %.4e\n', ...
    norm(Gamma-Gamma_theory,'fro'));

fprintf('||P1-P1_theory||       = %.4e\n', ...
    norm(P1-P1_theory,'fro'));

fprintf('||P2-P2_theory||       = %.4e\n', ...
    norm(P2-P2_theory,'fro'));

fprintf('\nRelative errors\n');

fprintf('Gamma : %.4e\n', ...
    norm(Gamma-Gamma_theory,'fro') ...
    / norm(Gamma_theory,'fro'));

fprintf('P1    : %.4e\n', ...
    norm(P1-P1_theory,'fro') ...
    / norm(P1_theory,'fro'));

fprintf('P2    : %.4e\n', ...
    norm(P2-P2_theory,'fro') ...
    / norm(P2_theory,'fro'));


%%
clear
clc

rng(1)


A = [0.8 0.1;
     0.0 0.9];

B = [1;
     0.5];

C = [1 0];

nx = 2;
nu = 1;
ny = 1;

Tini = 8;
N    = 10;


noiseStd = 0.05;

Tdata = 5000;

U = sign(randn(Tdata,nu));

x = randn(nx,1);

Y = zeros(Tdata,ny);


for k = 1:Tdata

    Y(k,:) = (C*x + noiseStd*randn(ny,1))';

    x = A*x + B*U(k,:)';

end


[Up,Yp,Uf,Yf] = Hankel_func( ...
    N,Tini,Tdata-Tini-N+1,...
    nu,ny,U.',Y.');


Theta = Yf*pinv([Up;Yp;Uf]);

nup = size(Up,1);
nyp = size(Yp,1);

P1 = Theta(:,1:nup);
P2 = Theta(:,nup+(1:nyp));
Gamma = Theta(:,nup+nyp+1:end);


G = cell(Tini+N+20,1);

for k = 1:length(G)

    G{k} = C*A^(k-1)*B;

end

Tp = zeros(Tini,Tini);

for i = 2:Tini

    for j = 1:i-1

        Tp(i,j) = G{i-j};

    end

end

Tf = zeros(N,N);

for i = 2:N

    for j = 1:i-1

        Tf(i,j) = G{i-j};

    end

end

Tfp = zeros(N,Tini);

for i = 1:N

    for j = 1:Tini

        idx = Tini + i - j;

        if idx >= 1

            Tfp(i,j) = G{idx};

        end

    end

end


Hh = zeros(Tini,nx);

for i = 1:Tini

    Hh(i,:) = C*A^(i-1);

end

Hf = zeros(N,nx);

for i = 1:N

    Hf(i,:) = C*A^(Tini+i-1);

end


P2_theory = Hf*pinv(Hh);

P1_theory = Tfp - P2_theory*Tp;

Gamma_theory = Tf;


fprintf('\n');
fprintf('Measurement noise std = %.3f\n\n', noiseStd);

fprintf('||Gamma-Gamma_theory|| = %.4e\n', ...
    norm(Gamma-Gamma_theory,'fro'));

fprintf('||P1-P1_theory||       = %.4e\n', ...
    norm(P1-P1_theory,'fro'));

fprintf('||P2-P2_theory||       = %.4e\n', ...
    norm(P2-P2_theory,'fro'));

fprintf('\nRelative errors\n');

fprintf('Gamma : %.4e\n', ...
    norm(Gamma-Gamma_theory,'fro') ...
    / norm(Gamma_theory,'fro'));

fprintf('P1    : %.4e\n', ...
    norm(P1-P1_theory,'fro') ...
    / norm(P1_theory,'fro'));

fprintf('P2    : %.4e\n', ...
    norm(P2-P2_theory,'fro') ...
    / norm(P2_theory,'fro'));


x0 = randn(nx,1);

uini = sign(randn(Tini,1));
uf   = sign(randn(N,1));

x = x0;

yini_true = zeros(Tini,1);

for k = 1:Tini

    yini_true(k) = C*x;

    x = A*x + B*uini(k);

end

yf_true = zeros(N,1);

for k = 1:N

    yf_true(k) = C*x;

    x = A*x + B*uf(k);

end

yini_meas = yini_true + noiseStd*randn(size(yini_true));

yf_pred = ...
      P1*uini ...
    + P2*yini_meas ...
    + Gamma*uf;

fprintf('\nPrediction error = %.4e\n', ...
    norm(yf_pred-yf_true));


%%


x0 = randn(nx,1);

uini_mat = sign(randn(Tini,nu));
uf_mat   = sign(randn(N,nu));

x = x0;

yini = zeros(Tini,ny);

for k = 1:Tini

    yini(k,:) = (C*x)';

    e = randn(nx,1);

    x = A*x + B*uini_mat(k,:)' + K*e;

end

yf_true = zeros(N,ny);

for k = 1:N

    yf_true(k,:) = (C*x)';

    e = randn(nx,1);

    x = A*x + B*uf_mat(k,:)' + K*e;

end

uini_vec = reshape(uini_mat.',[],1);
yini_vec = reshape(yini.',[],1);
uf_vec   = reshape(uf_mat.',[],1);


yf_spc = ...
      P1*uini_vec ...
    + P2*yini_vec ...
    + Gamma*uf_vec;


xhat = pinv(Hh)*(yini_vec - Tp*uini_vec);

yf_markov = ...
      Tfp*uini_vec ...
    + Tf*uf_vec ...
    + Hf*xhat;

yf_true_vec = reshape(yf_true.',[],1);

fprintf('\nPrediction errors\n');

fprintf('||SPC - truth||    = %.4e\n', ...
    norm(yf_spc-yf_true_vec));

fprintf('||Markov - truth|| = %.4e\n', ...
    norm(yf_markov-yf_true_vec));

fprintf('||SPC - Markov||   = %.4e\n', ...
    norm(yf_spc-yf_markov));


t = 1:N;

figure

subplot(3,1,1)

plot(t,yf_true_vec,'k','LineWidth',2)
hold on
plot(t,yf_spc,'b--','LineWidth',2)
plot(t,yf_markov,'r-.','LineWidth',2)

grid on

legend('Truth','SPC','Markov')

title('Predictions')

subplot(3,1,2)

plot(t,yf_true_vec-yf_spc,...
    'b','LineWidth',2)

grid on

title(sprintf('SPC error (norm = %.4e)', ...
    norm(yf_true_vec-yf_spc)))

ylabel('Error')

subplot(3,1,3)

plot(t,yf_true_vec-yf_markov,...
    'r','LineWidth',2)

grid on

title(sprintf('Markov error (norm = %.4e)', ...
    norm(yf_true_vec-yf_markov)))

ylabel('Error')
xlabel('Prediction step')

%%

fprintf('\n');
fprintf('========================================\n');
fprintf('P2*Hh = Hf test\n');
fprintf('========================================\n');

R = P2*Hh - Hf;

fprintf('||P2*Hh - Hf||_F = %.16e\n', ...
    norm(R,'fro'));

fprintf('max abs residual = %.16e\n', ...
    max(abs(R(:))));

fprintf('relative error = %.16e\n', ...
    norm(R,'fro')/norm(Hf,'fro'));

P2_theory = Hf*pinv(Hh);

R_theory = P2_theory*Hh - Hf;

fprintf('\n');
fprintf('========================================\n');
fprintf('P2_theory*Hh = Hf test\n');
fprintf('========================================\n');

fprintf('||P2_theory*Hh - Hf||_F = %.16e\n', ...
    norm(R_theory,'fro'));

DeltaP2 = P2 - P2_theory;

fprintf('\n');
fprintf('========================================\n');
fprintf('Nullspace investigation\n');
fprintf('========================================\n');

fprintf('||DeltaP2||_F = %.16e\n', ...
    norm(DeltaP2,'fro'));

fprintf('||DeltaP2*Yp||_F = %.16e\n', ...
    norm(DeltaP2*Yp,'fro'));

fprintf('||DeltaP2*Hh||_F = %.16e\n', ...
    norm(DeltaP2*Hh,'fro'));


%%

r_markov = yf_true_vec - yf_markov;

maxLag = 20;

[racf,lags] = xcorr(r_spc,maxLag,'normalized');

figure
stem(lags,racf,'filled')
grid on
xlabel('Lag')
ylabel('Autocorrelation')
title('SPC residual autocorrelation')

conf = 1.96/sqrt(length(r_spc));

hold on
plot([lags(1) lags(end)],[conf conf],'r--')
plot([lags(1) lags(end)],[-conf -conf],'r--')

%%

fprintf('\n');
fprintf('========================================\n');
fprintf('Check P2*Hh = Hf\n');
fprintf('========================================\n');

R = P2*Hh - Hf;

fprintf('||P2*Hh - Hf||_F = %.16e\n', ...
    norm(R,'fro'));

fprintf('Relative error = %.16e\n', ...
    norm(R,'fro')/norm(Hf,'fro'));

fprintf('max abs entry = %.16e\n', ...
    max(abs(R(:))));