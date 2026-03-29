function Inversion(N,tag)

addpath('Tools')
setenv('MATLAB_LOG_DIR', '/tmp')
myCluster = parcluster('Processes');
delete(myCluster.Jobs)




%==================== SAFE PARPOOL INITIALIZER ====================

% Desired number of workers
slurm_cpus = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(slurm_cpus) || slurm_cpus <= 0
    slurm_cpus = 90; % fallback default
end

% Check if a pool already exists
poolobj = gcp('nocreate');
if isempty(poolobj)
    % No pool exists: create one
    parpool('local', slurm_cpus);
else
    % A pool exists: check if it's the right size
    if poolobj.NumWorkers ~= slurm_cpus
        delete(poolobj)
        parpool('local', slurm_cpus);
    end
end

Dx=0.3;
Nx=120;

Model = set_model_RTM(Dx, Nx,tag);
EKI(N, Model, tag, 91882);

%Model = set_model_RTM(Dx, Nx, '_100');
%EKI(N, Model, '_100', 91882);

delete(gcp);
exit;
