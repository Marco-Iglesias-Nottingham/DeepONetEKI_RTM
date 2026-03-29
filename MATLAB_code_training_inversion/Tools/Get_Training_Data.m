function Get_Training_Data(N_En, Model, label, seed)
%GET_TRAINING_DATA  Generate training inputs/outputs and save to HDF5.
%
%   Get_Training_Data(N_En, Model, label, seed)
%
%   N_En  : number of training samples
%   Model : RTM model struct (from set_model_RTM)
%   label : string or numeric label for this batch (e.g. 'seed1', 42)
%   seed  : (optional) RNG seed for reproducibility
%
%   This function writes:
%     input_data_<label>.h5  - contains Input1..Input9
%     output_data_<label>.h5 - contains Output1..Output4
%   and stores the label and N_En as HDF5 attributes.

    if nargin < 3 || isempty(label)
        label = 'default';
    end
    if nargin < 4
        seed = [];   % no reset of RNG unless provided
    end

    % Normalise label to a string without spaces for filenames
    if isnumeric(label)
        label_str = num2str(label);
    else
        label_str = char(label);
    end
    label_str = regexprep(label_str, '\s+', '_');

    % Optional: set RNG seed for reproducibility of this batch
    if ~isempty(seed)
        rng(seed);
    end

    % -------------------------
    % Parameter ranges
    % -------------------------
    K_center   = [2.0e-10, 6.5e-10];
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
    % Sobol sampling
    % -------------------------
    sobol = sobolset(13,'Skip',1e3,'Leap',1e2);
    sobol = scramble(sobol,'MatousekAffineOwen');

    X = net(sobol, N_En);   % [N_En x 13] in [0,1]

    % Scale to physical ranges
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
    % Grid / covariance setup
    % -------------------------
    Nx = Model.Nx;
    Ny = Model.Nx;           % square grid

    % Covariance (Cholesky factors)
    C_LS  = GetCova(Nx, Ny, 0.075, 1,   2);
    C_RT  = GetCova(Nx, Ny, 0.1,   0.3, 2);
    C_def = GetCova(Nx, Ny, 0.1,   0.3, 2);

 

    % Geometry random fields (1D)
    C_geo    = GetCova1D(Nx, 0.15, 0.9*(0.3/20), 1.5);

    % -------------------------
    % Preallocate inputs/outputs
    % -------------------------
    nTimes = numel(Model.ob_times);
    nNodes = size(Model.RTM.Mesh.Nodes, 2);

    Input1  = zeros(Nx, Ny, N_En);
    Input2  = zeros(Nx, Ny, N_En);
    Input8  = zeros(5,     N_En);

    Output1 = zeros(nTimes, nNodes, N_En);
    Output2 = zeros(nTimes, nNodes, N_En);
   

    % -------------------------
    % Main ensemble loop
    % -------------------------
    parfor en = 1:N_En
        en
           % Random fields (per ensemble)
        LS_field       = C_LS'  * randn(Nx*Ny, 1);
        Perm_field_def = log(perm_def(en)) + C_def' * randn(Nx*Ny, 1);
        RT_field_top   = log(perm_RT1(en)) + C_RT'  * randn(Nx*Ny, 1);
        RT_field_bottom= log(perm_RT2(en)) + C_RT'  * randn(Nx*Ny, 1);
        RT_top   = C_geo' * randn(Nx, 1);
        RT_bottom= C_geo' * randn(Nx, 1);

        [perm_for_RTM, poro_for_RTM, ...         
         perm_for_deep, poro_for_deep] = get_perm( ...
            perm_C(en),  por_C(en), por_def_v(en), ...
            por_RT_top(en), por_RT_bottom(en), ...
            LS_field, Model, ...
            RT_field_top, RT_field_bottom, ...
            RT_top, RT_bottom, ...
            Perm_field_def);

        % --- Inputs (for ML) ---
        Input1(:,:,en) = reshape(log(perm_for_deep), Nx, Ny)';    % log-perm
        Input2(:,:,en) = reshape(poro_for_deep,     Nx, Ny)';     % porosity
        % Global scalar parameters (11-dim)
        Input8(:,en) = [ mu(en);           ...
            log(P(en));       ...
            gamma(en);        ...
            beta(en);         ...
            frac(en)];

        

        % --- Forward model runs ---
        [~,pressure, fill] = fwd_model(perm_for_RTM,  poro_for_RTM,  ...
                                       Model, mu(en), P(en), gamma(en), ...
                                       beta(en), frac(en));

        % --- Outputs ---
        Output1(:,:,en) = pressure';           % pressure or saturation field
        Output2(:,:,en) = fill';           % fill / saturation
    end

    % -------------------------
    % Save INPUTS to HDF5
    % -------------------------
    inputFile = sprintf('input_data_%s.h5', label_str);
    if isfile(inputFile)
        delete(inputFile);
    end

    %inputDatasets = { ...
    %    '/Input1','/Input2','/Input3','/Input4','/Input5', ...
    %    '/Input6','/Input7','/Input8','/Input9'};
    %inputVars = { ...
    %    Input1, Input2, Input3, Input4, Input5, ...
    %    Input6, Input7, Input8, Input9};

    
    
    inputDatasets = { ...
        '/Input1','/Input2','/Input8',};
    inputVars = { ...
        Input1, Input2, Input8};

    for k = 1:numel(inputDatasets)
        h5create(inputFile, inputDatasets{k}, size(inputVars{k}));
        h5write(inputFile, inputDatasets{k}, inputVars{k});
    end

    % Add metadata / label
    h5writeatt(inputFile, '/', 'label',   label_str);
    h5writeatt(inputFile, '/', 'N_En',    int32(N_En));
    if ~isempty(seed)
        h5writeatt(inputFile, '/', 'seed', seed);
    end

    % -------------------------
    % Save OUTPUTS to HDF5
    % -------------------------
    outputFile = sprintf('output_data_%s.h5', label_str);
    if isfile(outputFile)
        delete(outputFile);
    end

    outputDatasets = {'/Output1','/Output2'};
    outputVars     = {Output1, Output2};

    for k = 1:numel(outputDatasets)
        h5create(outputFile, outputDatasets{k}, size(outputVars{k}));
        h5write(outputFile, outputDatasets{k}, outputVars{k});
    end

    h5writeatt(outputFile, '/', 'label',   label_str);
    h5writeatt(outputFile, '/', 'N_En',    int32(N_En));
    if ~isempty(seed)
        h5writeatt(outputFile, '/', 'seed', seed);
    end

    fprintf('Input data saved to   %s\n', inputFile);
    fprintf('Output data saved to  %s\n', outputFile);
end



function [perm_for_RTM, poro_for_RTM, ...         
          perm_for_deep, poro_for_deep] = get_perm( ...
                perm_C,  por_C, por_def, ...
                por_RT_top, por_RT_bottom, ...
                LS, Model, ...
                RT_field_top, RT_field_bottom, ...
                RT_top, RT_bottom, ...
                Perm_field_def)

    % 1D geometry interpolants
    x_geo       = linspace(0, Model.Dx, Model.Nx);
    RT_geo_top    = griddedInterpolant(x_geo, RT_top,    'spline');
    RT_geo_bottom = griddedInterpolant(x_geo, RT_bottom, 'spline');

    RT_field_top_deep    = RT_field_top;
    RT_field_bottom_deep = RT_field_bottom;
    LS_deep              = LS;

    % Full 2D fields (deep grid)
    perm_for_deep = mask( ...
        perm_C, ...
        exp(Perm_field_def), ...
        exp(RT_field_top_deep), ...
        exp(RT_field_bottom_deep), ...
        RT_geo_top, RT_geo_bottom, LS_deep);

    poro_for_deep = mask( ...
        por_C, ...
        por_def, ...
        por_RT_top, ...
        por_RT_bottom, ...
        RT_geo_top, RT_geo_bottom, LS_deep);

    % Interpolate to RTM mesh
    x = linspace(0, Model.Dx, Model.Nx);
    y = linspace(0, Model.Dy, Model.Nx);
    [X, Y] = meshgrid(x, y);

    perm_for_RTM = interp2(X, Y, perm_for_deep, ...
        Model.opt.mesh.elem_center(:,1), ...
        Model.opt.mesh.elem_center(:,2), 'nearest');

    poro_for_RTM = interp2(X, Y, poro_for_deep, ...
        Model.opt.mesh.elem_center(:,1), ...
        Model.opt.mesh.elem_center(:,2), 'nearest');

  
end


function mask = mask(Central, Defect, ...
                     RT_field_top_deep, RT_field_bottom_deep, ...
                     RT_geo_top, RT_geo_bottom, LS)

    N  = 120;
    Dx = 0.3;

    x = linspace(0, Dx, N);
    y = linspace(0, Dx, N);
    [X, Y] = meshgrid(x, y);

    LS = reshape(LS, N, N);

    mask = Central * ones(N);

    % Defect region controlled by LS
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

    % Top and bottom RT regions
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

        C_prior(:, i) = sigma^2 * 2^(1-nu) / gamma(nu) ...
                         .* (h.^nu) .* besselk(nu, h);
    end

    % Diagonal fix (h = 0 -> besselk issues)
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
        v = (x(i) - x(:));
        h = sqrt((v.^2) / Lx^2);

        C_prior(:, i) = sigma^2 * 2^(1-nu) / gamma(nu) ...
                         .* (h.^nu) .* besselk(nu, h);
    end

    for i = 1:N
        C_prior(i,i) = sigma^2;
    end

    C = chol(C_prior);
end
