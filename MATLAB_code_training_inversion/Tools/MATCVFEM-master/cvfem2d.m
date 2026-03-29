function opt = cvfem2d(opt)
%  CVFEM2D 2D control volume finite elment method.
%     opt = CVFEM2D(opt) solves moving boundary problems using the control volume finite
%     element method.
%     
%     See also cvfem_setup

gD = str2func(opt.bndry.gd_filename);
opt.cvfem.u = zeros(opt.mesh.nnode,1);
opt.cvfem.u(opt.bndry.inlet_flag==1) = gD(opt.bndry.inlet_pos);
opt.cvfem.u(opt.cvfem.u==0) = opt.bndry.pvent;


% Initialise activeNode, activeElement vectors, fTime, fFactor,
opt = cvfem2d_init(opt);

if opt.vis.dumpFlag == 1
    fTimes = zeros(opt.mesh.nnode,1);
    cvfem2d_vis_timestep(opt);
    opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
end
Q   = update_flow_rate_tri(opt);

dt  = compute_time_increment(Q,opt.cvfem.fFactor,opt.cvfem.V);
opt.cvfem.fTime = increase_fTime(opt.cvfem.fTime,dt);
[opt.cvfem.fFactor, opt.cvfem.new_filled_volume] ...
    = update_filling_factor(opt.cvfem.fFactor,Q,dt,opt.cvfem.V);


if opt.vis.dumpFlag == 1
    fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
    cvfem2d_vis_timestep(opt);
    opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
end

while ~isFilled(opt.cvfem.fFactor,opt.mesh.nnode,opt.bndry.vent_idx)
    [opt.cvfem, opt.bndry] = still_solver(opt.cvfem,opt.mesh,opt.bndry);

    Q   = update_flow_rate_tri(opt);

    dt  = compute_time_increment(Q,opt.cvfem.fFactor,opt.cvfem.V);
    if dt == 0
        break;
    end
    opt.cvfem.fTime = increase_fTime(opt.cvfem.fTime,dt);
    [opt.cvfem.fFactor, opt.cvfem.new_filled_volume] ...
        = update_filling_factor(opt.cvfem.fFactor,Q,dt,opt.cvfem.V);
    if opt.vis.dumpFlag == 1
        fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
        cvfem2d_vis_timestep(opt);
        opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
    end
end

if opt.vis.dumpFlag == 1
    
    opt.vis.dumpIdx = opt.vis.dumpIdx - 1;
  
    [flag, ME,MID] = mkdir(opt.vis.filename);
    if strcmp(MID,'MATLAB:MKDIR:DirectoryExists')
        system(['rm -rf ' opt.vis.filename]);
        mkdir(opt.vis.filename);
    end
    
    cwd = pwd;
    system(['mv *.vtu ' cwd filesep opt.vis.filename]);
    cvfem2d_vis_global(fTimes(1:opt.vis.dumpIdx),opt.vis.filename)
    
end

