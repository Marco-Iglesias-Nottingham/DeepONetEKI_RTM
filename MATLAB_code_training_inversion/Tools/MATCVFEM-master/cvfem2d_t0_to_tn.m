function opt = cvfem2d_t0_to_tn(opt,tn)

if tn <= opt.cvfem.last_tn
    error('tn must be greater than last_tn')
end
gD = str2func(opt.bndry.gd_filename);
opt.cvfem.u = zeros(opt.mesh.nnode,1);
opt.cvfem.u(opt.bndry.inlet_flag==1) = gD(opt.bndry.inlet_pos);
opt.cvfem.u(opt.cvfem.u==0) = opt.bndry.pvent;
opt.cvfem.voidID = ones(opt.mesh.nnode,1);


% Initialise the activeNode and activeElement vectors
opt = cvfem2d_init(opt);

if opt.vis.dumpFlag == 1 && opt.cvfem.last_tn == 0
    opt.vis.fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
    cvfem2d_vis_timestep(opt);
    opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
end
Q   = update_flow_rate_tri(opt);

dt  = compute_time_increment(Q,opt.cvfem.fFactor,opt.cvfem.V);
opt.cvfem.fTime = increase_fTime(opt.cvfem.fTime,dt);
[opt.cvfem.fFactor, opt.cvfem.new_filled_volume] ...
    = update_filling_factor(opt.cvfem.fFactor,Q,dt,opt.cvfem.V);


if opt.vis.dumpFlag == 1 && opt.cvfem.last_tn < opt.cvfem.fTime
    opt.vis.fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
    cvfem2d_vis_timestep(opt);
    opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
end
u_old = opt.cvfem.u;
Q_old = Q;
while ~isFilled(opt.cvfem.fFactor,opt.mesh.nnode,opt.bndry.vent_idx)
    [opt.cvfem, opt.bndry] = still_solver(opt.cvfem,opt.mesh,opt.bndry);
    
    Q   = update_flow_rate_tri(opt);
    
    dt  = compute_time_increment(Q,opt.cvfem.fFactor,opt.cvfem.V);
    if dt == 0
        break;
    end
    opt.cvfem.fTime = increase_fTime(opt.cvfem.fTime,dt);
    
    
    if opt.cvfem.fTime > tn
        opt.cvfem.u = (tn - opt.cvfem.fTime+dt)/dt*opt.cvfem.u + ...
            (opt.cvfem.fTime-tn)/dt*u_old;
        Q(Q<0) = 0;
        Q_old(Q_old<0) = 0;
        opt.cvfem.Q = (tn-opt.cvfem.fTime+dt)/dt*Q + (opt.cvfem.fTime-tn)/dt*Q_old;
        opt.cvfem.fTime = tn;
        if opt.vis.dumpFlag == 1 && opt.cvfem.last_tn < opt.cvfem.fTime
            opt.vis.fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
            cvfem2d_vis_timestep(opt);
            opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
            opt.cvfem.last_tn = opt.cvfem.fTime;
        end
        break;
        
    end
    if opt.vis.dumpFlag == 1 && opt.cvfem.last_tn < opt.cvfem.fTime
        opt.vis.fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
        cvfem2d_vis_timestep(opt);
        opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
    end
    
    [opt.cvfem.fFactor, opt.cvfem.new_filled_volume] ...
        = update_filling_factor(opt.cvfem.fFactor,Q,dt,opt.cvfem.V);
    u_old = opt.cvfem.u;
    Q_old = Q;
end



