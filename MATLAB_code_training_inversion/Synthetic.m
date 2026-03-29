%% Setup
addpath('Tools')
setenv('MATLAB_LOG_DIR', '/tmp')

% Clean up any existing parallel jobs
myCluster = parcluster('Processes');
delete(myCluster.Jobs)

% Reproducible random seed
rng(989*30*561)

%% Global model / grid settings
Dx = 0.3;
Nx = 120;   % pass this into set_model_RTM

%% Rock / porosity properties
K_nom    = 4.0e-10;
K_top1   = 25e-10;
K_top2   = 40e-10;
K_bottom = 30e-10;
K_def    = 1.2e-10;
K_def2   = 0.4e-10;

poro_nom    = 0.73;
poro_def    = 0.62;
poro_top    = 0.91;
poro_bottom = 0.91;

% Build permeability / porosity fields on a regular N x N grid
x = linspace(0, Dx, Nx);
y = linspace(0, Dx, Nx);
[X, Y] = meshgrid(x, y);

center2 = [0.075, 0.1];
radius  = 0.04;
dist2   = sqrt((X - center2(1)).^2 + (Y - center2(2)).^2);

mask = K_nom * ones(Nx);
mask(dist2 < radius)                       = K_def;
mask(0.975*Dx < Y & X < 0.4*Dx)           = K_top1;
mask(0.95*Dx  < Y & X > 0.55*Dx)          = K_top2;
mask(Y < 0.025*Dx & X > 0.5*Dx)           = K_bottom;
mask(Y > 0.2*Dx & Y < 0.75*Dx & ...
     X < 0.7*Dx & X > 0.6*Dx)             = K_def2;

mask2 = poro_nom * ones(Nx);
mask2(dist2 < radius)                     = poro_def;
mask2(0.975*Dx < Y & X < 0.4*Dx)         = poro_top;
mask2(0.95*Dx  < Y & X > 0.55*Dx)        = poro_top;
mask2(Y < 0.02*Dx & X > 0.5*Dx)          = poro_bottom;
mask2(Y > 0.2*Dx & Y < 0.75*Dx & ...
      X < 0.7*Dx & X > 0.6*Dx)           = poro_def;

%% Forward model physical parameters
mu    = 0.0922;
Pi    = 1.0912e5;
gamma = 1.114;
beta  = 0.42;
frac  = 0.66;

%% Loop over sensor configurations using TAGS (no booleans)
% dataTags:
%   '_for_real' -> lab sensor layout (matches syndata_for_real)
%   '_100'      -> 10x10 synthetic grid (matches syndata_100)
dataTags = {'_for_real', '_100'};

for k = 1:numel(dataTags)
    tag = dataTags{k};

    % Build Model with chosen sensor layout
    Model = set_model_RTM(Dx, Nx, tag);

    % Interpolate permeability / porosity to RTM element centers
    query   = [Model.opt.mesh.elem_center(:,1), ...
               Model.opt.mesh.elem_center(:,2)];
    query_x = query(:,1);
    query_y = query(:,2);

    perm_interp = interp2(X, Y, mask,  query_x, query_y, 'linear');
    poro        = interp2(X, Y, mask2, query_x, query_y, 'linear');

    % Run forward model
    [noise_free_data, ~, ~] = fwd_model(perm_interp, poro, ...
                                        Model, mu, Pi, gamma, beta, frac);

    % Add synthetic noise
    scalar    = 0.025 * abs(noise_free_data);
    Error_std = max(scalar, 100);
    syn_data  = noise_free_data + Error_std .* randn(size(noise_free_data));

    % Choose output filename based on tag
    %   tag = '_for_real' -> syndata_for_real.mat
    %   tag = '_100'      -> syndata_100.mat
    outFile = ['MATLAB_files_for_emulator/syndata' tag '.mat'];

    % Save everything you need
    save(outFile, ...
        'syn_data', 'mask', 'mask2', 'Model', 'Error_std', 'noise_free_data');

    fprintf('Saved synthetic data to %s (sensor_type = %s)\n', ...
            outFile, Model.sensor_type);
end
save('MATLAB_files_for_emulator/Model.mat', 'Model')