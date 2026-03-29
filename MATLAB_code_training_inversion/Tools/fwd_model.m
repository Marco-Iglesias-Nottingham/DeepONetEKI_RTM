function [meas, pressures,fill]=fwd_model(U,Phi,Model,mu,Pi,gamma,beta,frac)

opt=Model.opt;
if (Model.opt.mesh.nelem~=length(U))
    throw(MException('fwd_model:wrongInput',...
        'perm must be interpolated'));
end

opt.darcy.mu=mu;
opt.bndry.pinlet =Pi;

opt = setK(opt,U);
opt = setPhi(opt,Phi);
opt = cvfem2d_init(opt);
[opt, pressures, flow,fill]  = cvfem2d_obsrv(opt,Model.ob_times,gamma,beta,frac);
pressures=max(pressures,opt.bndry.pvent);
meas=pressures(Model.nodes_meas,:);

