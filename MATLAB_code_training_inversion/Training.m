


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

%==================== END SAFE PARPOOL INITIALIZER ====================

Dx=0.3;
N=120;
N_En=40000;

Model = set_model_RTM(Dx, N, '_lab');


Get_Training_Data(N_En, Model, 'batch_seed_1', 3123);


N_En=10000;
Get_Training_Data(N_En, Model, 'batch_seed_2', 4501);


delete(gcp);
exit;





