function EKI(N_En, Model, dataTag, seed)
%EKI  Ensemble-based inversion with EnKF / EKI.
%
%   EKI(N_En, Model, dataTag, seed)
%
%   N_En    : ensemble size
%   Model   : RTM model struct
%   dataTag : string flag for synthetic data set:
%               '_100'      -> use syndata_100.mat   (100 sensors)
%               '_for_real' -> use syndata_for_real.mat (lab sensors)
%             if omitted, defaults to '_for_real'
%   seed    : (optional) RNG seed for reproducibility

    % -------------------------
    % Handle arguments
    % -------------------------
    if nargin < 2
        error('EKI: not enough input arguments. Need at least N_En and Model.');
    end
    if nargin < 3 || isempty(dataTag)
        dataTag = '_for_real';
    end
    if nargin < 4
        seed = [];
    end

    % -------------------------
    % RNG seed for reproducibility
    % -------------------------
    if ~isempty(seed)
        rng(seed);
    end

    % -------------------------
    % Hyper-parameter limits
    % -------------------------
    K_center   = [2.5e-10, 6.5e-10];
    K_def      = [0.25e-10, 2.5e-10];
    por_center = [0.6, 0.8];
    por_def    = [0.55, 0.7];
    K_RT       = [20e-10, 50e-10];
    por_RT     = [0.9, 0.96];
    mu_lim     = [0.085, 0.12];
    P_lim      = [92e3, 120e3];
    gamma_lim  = [0.6, 1.25];
    beta_lim   = [0.2, 0.7];
    frac_lim   = [0.35, 0.75];

    % -------------------------
    % Sobol sampling of parameters
    % -------------------------
    sobol = sobolset(13, 'Skip', 1e3, 'Leap', 1e2);
    sobol = scramble(sobol, 'MatousekAffineOwen');

    X = net(sobol, N_En);  % N_En x 13

    % Scale to desired ranges
    perm_C        = K_center(1)   + (K_center(2)   - K_center(1))   * X(:,1);
    perm_def      = K_def(1)      + (K_def(2)      - K_def(1))      * X(:,2);
    perm_RT1      = K_RT(1)       + (K_RT(2)       - K_RT(1))       * X(:,3);
    perm_RT2      = K_RT(1)       + (K_RT(2)       - K_RT(1))       * X(:,4);
    por_C         = por_center(1) + (por_center(2) - por_center(1)) * X(:,5);
    por_def_v     = por_def(1)    + (por_def(2)    - por_def(1))    * X(:,6);
    por_RT_top    = por_RT(1)     + (por_RT(2)     - por_RT(1))     * X(:,7);
    por_RT_bottom = por_RT(1)     + (por_RT(2)     - por_RT(1))     * X(:,8);
    mu            = mu_lim(1)     + (mu_lim(2)     - mu_lim(1))     * X(:,9);
    gamma         = gamma_lim(1)  + (gamma_lim(2)  - gamma_lim(1))  * X(:,10);
    P             = P_lim(1)      + (P_lim(2)      - P_lim(1))      * X(:,11);
    beta          = beta_lim(1)   + (beta_lim(2)   - beta_lim(1))   * X(:,12);
    frac          = frac_lim(1)   + (frac_lim(2)   - frac_lim(1))   * X(:,13);

    % -------------------------
    % Grid / covariance
    % -------------------------
    Nx = Model.Nx;
    Ny = Model.Nx;

    % Load or build covariance matrices (reused across calls)
    if isfile('Covariance_Matrices.mat')
        load('Covariance_Matrices.mat', 'C_LS', 'C_RT', 'C_def', 'C_geo');
    else
        C_LS  = GetCova(Nx, Ny, 0.075, 1,   2);
        C_RT  = GetCova(Nx, Ny, 0.1,   0.3, 2);
        C_def = GetCova(Nx, Ny, 0.1,   0.3, 2);
        C_geo = GetCova1D(Nx,  0.15,  0.9*(0.3/20), 1.5);
        save('Covariance_Matrices.mat', 'C_LS', 'C_RT', 'C_def', 'C_geo');
    end

    % Random fields
    LS_field       = C_LS'  * randn(Nx*Ny, N_En);
    Perm_field_def = log(perm_def') + C_def' * randn(Nx*Ny, N_En);
    RT_field_top   = log(perm_RT1') + C_RT'  * randn(Nx*Ny, N_En);
    RT_field_bottom= log(perm_RT2') + C_RT'  * randn(Nx*Ny, N_En);

    RT_top    = C_geo' * randn(Nx, N_En);
    RT_bottom = C_geo' * randn(Nx, N_En);

    % -------------------------
    % Preallocation
    % -------------------------
    nTimes = numel(Model.ob_times);
    nMeas  = numel(Model.nodes_meas);  % number of measurement points

    Input3 = zeros(Nx*Ny, N_En);  % LS field
    Input4 = zeros(Nx,     N_En); % RT_top geom
    Input5 = zeros(Nx,     N_En); % RT_bottom geom
    Input6 = zeros(Nx*Ny, N_En);  % RT_field_top
    Input7 = zeros(Nx*Ny, N_En);  % RT_field_bottom
    Input8 = zeros(11,     N_En); % global parameters
    Input9 = zeros(Nx*Ny, N_En);  % Perm_field_def

    % flattened obs: time x measurement nodes
    Output = zeros(nTimes * nMeas, N_En);

    % Log-limits for transformation (for global parameters)
    Lim = {
        [log(2.5e-10),  log(6.5e-10)];   % log(perm_C)
        [log(0.25e-10), log(2.5e-10)];   % log(perm_def)
        [0.6,   0.8];                    % por_C
        [0.55,  0.7];                    % por_def
        [0.9,   0.96];                   % por_RT_top
        [0.9,   0.96];                   % por_RT_bottom
        [0.085, 0.12];                   % mu
        [log(92e3), log(120e3)];         % log(P)
        [0.6,   1.25];                   % gamma
        [0.2,   0.7];                    % beta
        [0.35,  0.75];                   % frac
    };

    % Switch to log-space for some parameters
    perm_C   = log(perm_C);
    perm_def = log(perm_def);

    % -------------------------
    % Synthetic data selection based on dataTag
    % -------------------------
    switch lower(strtrim(dataTag))
        case '_100'
            dataFile = 'MATLAB_files_for_emulator/syndata_100.mat';
            dataTag  = '_100';
        case '_for_real'
            dataFile = 'MATLAB_files_for_emulator/syndata_for_real.mat';
            dataTag  = '_for_real';
        otherwise
            error('Unknown dataTag "%s". Use "_100" or "_for_real".', dataTag);
    end

    if ~isfile(dataFile)
        error('Synthetic data file "%s" not found.', dataFile);
    end

    dataStruct = load(dataFile);
    syn_data   = dataStruct.syn_data;
    Error_std  = dataStruct.Error_std;

    % -------------------------
    % Inversion loop
    % -------------------------
    iter   = 1;
    t      = 0;
    flag   = false;
    Misfit = [];

    % combined suffix for all outputs: e.g. '_100_NEn500' or '_for_real_NEn500'
    suffix = sprintf('%s_NEn%d', dataTag, N_En);
    if ~isempty(seed)
        suffix = sprintf('%s_seed%d', suffix, seed);
    end

    tic
    while ~flag

        % Forward model for each ensemble member
        for en = 1:N_En
            [perm_for_RTM, poro_for_RTM] = get_perm( ...
                exp(perm_C), por_C, por_def_v, por_RT_top, por_RT_bottom, ...
                LS_field(:,en), Model, en, ...
                RT_field_top(:,en), RT_field_bottom(:,en), ...
                RT_top(:,en), RT_bottom(:,en), ...
                Perm_field_def(:,en));

            % Store inputs
            Input3(:,en) = LS_field(:,en);
            Input4(:,en) = RT_top(:,en);
            Input5(:,en) = RT_bottom(:,en);
            Input6(:,en) = RT_field_top(:,en);
            Input7(:,en) = RT_field_bottom(:,en);
            Input8(:,en) = [ ...
                perm_C(en); perm_def(en); ...
                por_C(en);  por_def_v(en); ...
                por_RT_top(en); por_RT_bottom(en); ...
                mu(en); log(P(en)); gamma(en); ...
                beta(en); frac(en)];
            Input9(:,en) = Perm_field_def(:,en);

            meas = fwd_model(perm_for_RTM, poro_for_RTM, ...
                             Model, mu(en), P(en), gamma(en), beta(en), frac(en));

            % meas: [nTimes x nNodes]; restrict to measurement nodes
            Output(:,en)  = meas(:);
        end

        % Transform global params for inversion step
        Input8_for_inversion = TransformAll(Input8, Lim);

        % Save prior ensemble only at first iteration
        if iter == 1
            priorFile = ['prior_ensemble' suffix '.h5'];
            if isfile(priorFile), delete(priorFile); end

            writeEnsembleH5(priorFile, ...
                {'/Input3','/Input4','/Input5','/Input6','/Input7','/Input8','/Input9'}, ...
                {Input3,   Input4,   Input5,   Input6,   Input7,   Input8,   Input9});
        end

        % -------------------------
        % Build covariance and misfit
        % -------------------------
        Cova = diag(Error_std(:).^2);

        % Eigen decomposition for inverse sqrt
        [U, D]    = eig(Cova);
        lambda    = diag(D);
        tol       = 1e-12 * max(lambda);
        lambda(lambda < tol) = 0;

        inv_sqrt_lambda      = zeros(size(lambda));
        nz_idx               = lambda > tol;
        inv_sqrt_lambda(nz_idx) = 1 ./ sqrt(lambda(nz_idx));

        Cova_inv_sqrt = U * diag(inv_sqrt_lambda) * U';

        % mismatch in obs space
        Z      = Cova_inv_sqrt * (syn_data(:) - Output);
        M      = length(syn_data(:));
        Z_m    = mean(Z, 2);
        Misfit(iter) = norm(Z_m)^2 / M;

        alpha = mean(rmoutliers(vecnorm(Z).^2) / M);

        Delta_Z = Z - Z_m;
        C       = (Delta_Z * Delta_Z.') / (N_En - 1);
        C_tilde = C + alpha * eye(M);
        rho     = 0.6;

        TT      = rho * norm(Z_m) - alpha * norm(C_tilde \ Z_m);

        if TT > 0
            alpha   = 2 * alpha;
            C_tilde = C + alpha * eye(M);
            TT      = rho * norm(Z_m) - alpha * norm(C_tilde \ Z_m);
        end

        if t(iter) + 1/alpha > 1
            alpha = 1 / (1 - t(iter));
        end

        % Kalman update
        Delta_Z_scaled = Delta_Z / sqrt(N_En - 1);
        E              = sqrt(alpha) * randn(M, N_En);
        E              = E - mean(E, 2);
        Z              = Z + E;

        B = (Delta_Z_scaled * Delta_Z_scaled.' + alpha * eye(M)) \ Z;

        Input3               = Update_Kalman(N_En, Input3,              B, Delta_Z_scaled);
        Input4               = Update_Kalman(N_En, Input4,              B, Delta_Z_scaled);
        Input5               = Update_Kalman(N_En, Input5,              B, Delta_Z_scaled);
        Input6               = Update_Kalman(N_En, Input6,              B, Delta_Z_scaled);
        Input7               = Update_Kalman(N_En, Input7,              B, Delta_Z_scaled);
        Input8_for_inversion = Update_Kalman(N_En, Input8_for_inversion, B, Delta_Z_scaled);
        Input9               = Update_Kalman(N_En, Input9,              B, Delta_Z_scaled);

        Input8 = InvTransformAll(Input8_for_inversion, Lim);

        % unpack updated fields
        LS_field        = Input3;
        RT_top          = Input4;
        RT_bottom       = Input5;
        RT_field_top    = Input6;
        RT_field_bottom = Input7;

        perm_C         = Input8(1,:);
        perm_def       = Input8(2,:);
        por_C          = Input8(3,:);
        por_def_v      = Input8(4,:);
        por_RT_top     = Input8(5,:);
        por_RT_bottom  = Input8(6,:);
        mu             = Input8(7,:);
        P              = exp(Input8(8,:));
        gamma          = Input8(9,:);
        beta           = Input8(10,:);
        frac           = Input8(11,:);
        Perm_field_def = Input9;

        % time-like parameter
        t(iter+1) = t(iter) + 1/alpha;
        iter      = iter + 1;

        % save intermediate inversion state
        save(['Inversion_out' suffix '.mat'], 'Misfit', 'alpha', 't', 'iter', '-v7.3');

        disp('Iteration update done');

        if abs(t(iter) - 1) < 1e-5
            flag = true;
        end
    end

    toc

    % -------------------------
    % Save posterior ensemble
    % -------------------------
    posteriorFile = ['posterior_ensemble' suffix '.h5'];
    if isfile(posteriorFile), delete(posteriorFile); end

    writeEnsembleH5(posteriorFile, ...
        {'/Input3','/Input4','/Input5','/Input6','/Input7','/Input8','/Input9'}, ...
        {Input3,   Input4,   Input5,   Input6,   Input7,   Input8,   Input9});
end


function [perm_for_RTM, poro_for_RTM] = get_perm( ...
    perm_C, por_C, por_def, por_RT_top, por_RT_bottom, ...
    LS, Model, en, ...
    RT_field_top, RT_field_bottom, ...
    RT_top, RT_bottom, ...
    Perm_field_def)

    x = linspace(0, Model.Dx, Model.Nx);
    RT_geo_top    = griddedInterpolant(x, RT_top,    'spline');
    RT_geo_bottom = griddedInterpolant(x, RT_bottom, 'spline');

    perm_for_deep = mask(perm_C(en), ...
                         exp(Perm_field_def), ...
                         exp(RT_field_top), ...
                         exp(RT_field_bottom), ...
                         RT_geo_top, RT_geo_bottom, LS);

    poro_for_deep = mask(por_C(en), ...
                         por_def(en), ...
                         por_RT_top(en), ...
                         por_RT_bottom(en), ...
                         RT_geo_top, RT_geo_bottom, LS);

    y = linspace(0, Model.Dy, Model.Nx);
    [X, Y] = meshgrid(x, y);

    perm_for_RTM = interp2(X, Y, perm_for_deep, ...
        Model.opt.mesh.elem_center(:,1), ...
        Model.opt.mesh.elem_center(:,2), 'nearest');

    poro_for_RTM = interp2(X, Y, poro_for_deep, ...
        Model.opt.mesh.elem_center(:,1), ...
        Model.opt.mesh.elem_center(:,2), 'nearest');
end


function mask = mask(Central, Defect, RT_field_top_deep, RT_field_bottom_deep, ...
                     RT_geo_top, RT_geo_bottom, LS)

    N  = 120;
    Dx = 0.3;

    x = linspace(0, Dx, N);
    y = linspace(0, Dx, N);
    [X, Y] = meshgrid(x, y);

    LS = reshape(LS, N, N);

    mask = Central * ones(N);

    if ~isscalar(Defect)
        Defect2D = reshape(Defect, N, N);
        mask = mask + (Defect2D - mask) .* (LS > 1.0);
    else
        mask(LS > 1.0) = Defect;
    end

    if ~isscalar(RT_field_top_deep)
        RT_field_top_deep    = reshape(RT_field_top_deep,    N, N);
        RT_field_bottom_deep = reshape(RT_field_bottom_deep, N, N);
    end

    mask = mask ...
        + (RT_field_top_deep    - mask) .* (Y <  RT_geo_top(X)) ...
        + (RT_field_bottom_deep - mask) .* (Y > (Dx - RT_geo_bottom(X)));
end


function C = GetCova(Nx, Ny, L, sigma, nu)
    D  = 1;
    hx = D / Nx;
    hy = D / Ny;

    [X, Y] = meshgrid(hx/2:hx:hx*Nx-hx/2, ...
                      hy/2:hy:hy*Ny-hy/2);

    N  = Nx * Ny;
    x  = reshape(X, N, 1);
    y  = reshape(Y, N, 1);

    Lx = L;
    Ly = L;

    C_prior = zeros(N, N);

    for i = 1:N
        v = [ (x(i) - x(:))'; (y(i) - y(:))' ];
        h = sqrt( (v(1,:).^2) / Lx^2 + (v(2,:).^2) / Ly^2 );
        C_prior(:, i) = sigma^2 * 2^(1-nu)/gamma(nu) .* (h.^nu) .* besselk(nu, h);
    end

    for i = 1:N
        C_prior(i,i) = sigma^2;
    end

    C = chol(C_prior);
end


function C = GetCova1D(Nx, L, sigma, nu)
    D  = 1;
    hx = D / Nx;
    x  = linspace(hx/2, D - hx/2, Nx);

    N  = Nx;
    Lx = L;

    C_prior = zeros(N, N);

    for i = 1:N
        v = x(i) - x(:);
        h = sqrt((v.^2) / Lx^2);
        C_prior(:, i) = sigma^2 * 2^(1-nu)/gamma(nu) .* (h.^nu) .* besselk(nu, h);
    end

    for i = 1:N
        C_prior(i,i) = sigma^2;
    end

    C = chol(C_prior);
end


function Un = Update_Kalman(N_En, Un, B, Delta_Z_scaled)
    Un_m    = mean(Un, 2);
    Delta_U = (Un - Un_m) / sqrt(N_En - 1);
    C_uz    = Delta_U * Delta_Z_scaled';
    Un      = Un - C_uz * B;
end


function y = logTrans(x, lim)
    y = log((lim(2) - x) ./ (x - lim(1)));
end

function x = inverse_log_transform(y, lim)
    e_y = exp(y);
    x   = (lim(2) + e_y * lim(1)) ./ (1 + e_y);
end

function transformed = TransformAll(variable, lim)
    [D, N] = size(variable);
    transformed = zeros(D, N);
    for n = 1:D
        transformed(n, :) = logTrans(variable(n, :), lim{n});
    end
end

function recovered = InvTransformAll(variable, lim)
    [D, N] = size(variable);
    recovered = zeros(D, N);
    for n = 1:D
        recovered(n, :) = inverse_log_transform(variable(n, :), lim{n});
    end
end



function writeEnsembleH5(filename, datasetNames, dataCells)
    for k = 1:numel(datasetNames)
        h5create(filename, datasetNames{k}, size(dataCells{k}));
        h5write(filename, datasetNames{k}, dataCells{k});
    end
end
