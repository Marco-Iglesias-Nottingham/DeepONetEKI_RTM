function Get_Posterior_Predictions(N_En,Model,filename)



% Read datasets from the HDF5 file
LS_field             = h5read(filename, '/Input3');
RT_top     = h5read(filename, '/Input4');
RT_bottom  = h5read(filename, '/Input5');
RT_field_top   = h5read(filename, '/Input6');
RT_field_bottom= h5read(filename, '/Input7');
vec            = h5read(filename, '/Input8');
Perm_field_def      = h5read(filename, '/Input9');


perm_C=vec(1,:);
perm_def=vec(2,:);
por_C=vec(3,:);
por_def_v=vec(4,:);
por_RT_top=vec(5,:);
por_RT_bottom=vec(6,:); 
mu=vec(7,:);
P=exp(vec(8,:)); 
gamma=vec(9,:);
beta=vec(10,:);
frac=vec(11,:);


%% Parameters
Nx = Model.Nx;            % grid size in x
Ny = Model.Nx;            % grid size in y



tic 

    parfor en=1:N_En
        [perm_for_RTM,poro_for_RTM]=...'
            get_perm(exp(perm_C),por_C,por_def_v,por_RT_top,por_RT_bottom,...
            LS_field(:,en),Model,en,RT_field_top(:,en),RT_field_bottom(:,en),RT_top(:,en),RT_bottom(:,en),Perm_field_def(:,en));
        meas=fwd_model(perm_for_RTM,poro_for_RTM,Model,mu(en),P(en),gamma(en),beta(en),frac(en));    
        Output(:,en)=meas(:);
    end





save('pos_pred.mat', "Output")






end


function [perm_for_RTM,poro_for_RTM]=...'
    get_perm(perm_C,por_C,por_def,por_RT_top,por_RT_bottom,LS,Model,en,RT_field_top,RT_field_bottom,RT_top,RT_bottom,Perm_field_def)




x = linspace(0,Model.Dx,Model.Nx);
RT_geo_top = griddedInterpolant(x, RT_top, 'spline');
RT_geo_bottom = griddedInterpolant(x, RT_bottom, 'spline');


perm_for_deep = mask(perm_C(en), exp(Perm_field_def),exp(RT_field_top),exp(RT_field_bottom),RT_geo_top,RT_geo_bottom,LS);
poro_for_deep = mask(por_C(en), por_def(en),por_RT_top(en),por_RT_bottom(en),RT_geo_top,RT_geo_bottom,LS);


y = linspace(0, Model.Dy, Model.Nx);   % Y-axis grid points
[X, Y] = meshgrid(x, y);  % Create meshgrid
perm_for_RTM = interp2(X, Y, perm_for_deep, Model.opt.mesh.elem_center(:,1), Model.opt.mesh.elem_center(:,2), 'nearest');  % Method can be 'linear', 'spline', 'cubic', etc.
poro_for_RTM = interp2(X, Y, poro_for_deep, Model.opt.mesh.elem_center(:,1), Model.opt.mesh.elem_center(:,2), 'nearest');  % Method can be 'linear', 'spline', 'cubic', etc.






end




function mask=mask(Central, Defect,RT_field_top_deep,RT_field_bottom_deep,RT_geo_top,RT_geo_bottom,LS)

N = 120;
Dx=0.3;
% Create [0,1] x [0,1] grid
x = linspace(0, Dx, N);
y = linspace(0, Dx, N);
[X, Y] = meshgrid(x, y);
LS=reshape(LS,N,N);

mask = Central * ones(N);  % initialize with K2
if ~isscalar(Defect)
    mask= mask+(reshape(Defect,N,N)-mask).*(LS>1.0);
else
    mask(LS>1.0) = Defect;
end
if ~isscalar(RT_field_top_deep)
    RT_field_top_deep=reshape(RT_field_top_deep,N,N);
    RT_field_bottom_deep=reshape(RT_field_bottom_deep,N,N);
end
mask = mask+(RT_field_top_deep-mask).*(Y <RT_geo_top(X))+(RT_field_bottom_deep-mask).*(Y >Dx-RT_geo_bottom(X) );

end

