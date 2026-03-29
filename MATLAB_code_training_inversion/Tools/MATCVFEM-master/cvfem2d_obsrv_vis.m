function [opt, pressures, flow_rates, filling_factors] = cvfem2d_obsrv(opt,obsrv_times)

obsrv_times = sort(obsrv_times);
tindex = 1;
pressures = zeros(opt.mesh.nnode,length(obsrv_times));
flow_rates = zeros(opt.mesh.nnode,length(obsrv_times));
filling_factors = zeros(opt.mesh.nnode,length(obsrv_times));
gD = str2func(opt.bndry.gd_filename);
opt.cvfem.u = zeros(opt.mesh.nnode,1);
opt.cvfem.u(opt.bndry.inlet_flag==1) = gD(opt.bndry.inlet_pos);
opt.cvfem.u(opt.cvfem.u==0) = opt.bndry.pvent;
opt.cvfem.voidID = ones(opt.mesh.nnode,1);


% Initialise the activeNode and activeElement vectors
opt = cvfem2d_init(opt);


if opt.vis.dumpFlag == 1
    opt.vis.fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
    cvfem2d_vis_timestep(opt);
    opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
end


Q   = update_flow_rate_tri(opt);

dt  = compute_time_increment(Q,opt.cvfem.fFactor,opt.cvfem.V);
opt.cvfem.fTime = increase_fTime(opt.cvfem.fTime,dt);
[opt.cvfem.fFactor, opt.cvfem.new_filled_volume] ...
    = update_filling_factor(opt.cvfem.fFactor,Q,dt,opt.cvfem.V);

u_old = opt.cvfem.u;
Q_old = Q;
fFactor_old = opt.cvfem.fFactor;
while ~isFilled(opt.cvfem.fFactor,opt.mesh.nnode,opt.bndry.vent_idx)
    [opt.cvfem, opt.bndry] = still_solver(opt.cvfem,opt.mesh,opt.bndry);
    
    Q   = update_flow_rate_tri(opt);
    
    dt  = compute_time_increment(Q,opt.cvfem.fFactor,opt.cvfem.V);
    if dt == 0
        break;
    end
    opt.cvfem.fTime = increase_fTime(opt.cvfem.fTime,dt);
    
    if tindex <= length(obsrv_times)
        while opt.cvfem.fTime >= obsrv_times(tindex)
            Q(Q<0) = 0;
            Q_old(Q_old<0) = 0;
            current_time = obsrv_times(tindex);
            pressures(:,tindex) ...
                = (current_time-opt.cvfem.fTime+dt)/dt*opt.cvfem.u+...
                (opt.cvfem.fTime-current_time)/dt*u_old;
            flow_rates(:,tindex)...
                = (current_time-opt.cvfem.fTime+dt)/dt*Q +...
                (opt.cvfem.fTime-current_time)/dt*Q_old;
            filling_factors(:,tindex) ...
                = (current_time-opt.cvfem.fTime+dt)/dt*opt.cvfem.fFactor +...
                (opt.cvfem.fTime-current_time)/dt*fFactor_old;
            tindex = tindex + 1;
            if opt.vis.dumpFlag == 1
                opt.vis.fTimes(opt.vis.dumpIdx) = opt.cvfem.fTime;
                cvfem2d_vis_timestep(opt);
                opt.vis.dumpIdx = opt.vis.dumpIdx + 1;
                aa=opt.vis.dumpIdx
            end
            
            
            if tindex > length(obsrv_times)
                break;
            end
            
        end
        
    end
    
    [opt.cvfem.fFactor, opt.cvfem.new_filled_volume] ...
        = update_filling_factor(opt.cvfem.fFactor,Q,dt,opt.cvfem.V);
    
    
    u_old = opt.cvfem.u;
    Q_old = Q;
    fFactor_old = opt.cvfem.fFactor;
    
    
end
if tindex <= length(obsrv_times)
    if opt.cvfem.fTime < obsrv_times(tindex)
        for i=tindex:length(obsrv_times)
            pressures(:,i)=opt.cvfem.u;
        end
    end
end





