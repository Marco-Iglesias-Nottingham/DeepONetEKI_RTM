function opt = set_opt(n,pos)
% SET_OPT(N,NPAD) This m file generates a option file for the control
% volume finite element simulations
if length(n) == 1
    opt.n = [1 1]*n;
else
    opt.n = n;
end

if nargin < 2
    opt.xstart = 0;
    opt.xend = 2;
    opt.ystart = 0;
    opt.yend = 1;
else
    opt.xstart = pos(1);
    opt.xend = pos(2);
    opt.ystart = pos(3);
    opt.yend = pos(4);
end

opt.N = prod(opt.n+1) + prod(opt.n);
opt.NT = prod(opt.n)*4;

[opt.elem, opt.node, opt.nbd_elem, opt.nbd_node, opt.nbd_elem_elem] = kill_build_fem_mesh(opt);

opt.darcy_para.thickness =  .001;     % meter
opt.darcy_para.mu  = .1;               % viscousity (Pa.s)
opt.darcy_para.phi =  .7;              % porosity 

opt.fTime = 0;              % The filling time
opt.fFactor = zeros(opt.N,1);            % the filling factor

opt.activeElement   = zeros(opt.NT,1);  % 1 if the correponding element is active in the 
                                        % computational domain. otherwise 0
                                        
                                        
opt.activeNode      = zeros(opt.N,1);   % 1 if the node is used in the compuational domain.
                                        % Otherwise 0.
                                       
opt.iArray = zeros(9*opt.NT,1);         % row index array
opt.jArray = zeros(9*opt.NT,1);         % column index array
opt.sArray = zeros(9*opt.NT,1);         % value array
opt.arrayIndex = 0;                     % index of the last nonzero element in the arrays.

opt.u = zeros(opt.N,1);                 % pressure vector
opt.V = compute_volumes(opt.node,opt.elem,opt.darcy_para.thickness);           % Volume of the control volume

% inlet location
opt.inlet_idx = 1:opt.n(1)+1:(opt.n(1)+1)*(opt.n(2)+1);
opt.inlet_pos = zeros(length(opt.inlet_idx),2);


% vent location
opt.vent_idx = opt.n(1)+1:opt.n(1)+1:(opt.n(1)+1)*(opt.n(2)+1);
opt.vent_elem = 4*opt.n(1)-1:4*opt.n(1):4*opt.n(1)*opt.n(2);
% Boundary conditions
opt.gD = 'g_D';     % Dirichlet
opt.gN = 'g_N';     % Neumann

opt.pvent = 100000;
opt.pinlet = 600000;

% parameters for the dumper
opt.filename = 'test';
opt.dumpFlag = true;
opt.dumpIdx = 1;

