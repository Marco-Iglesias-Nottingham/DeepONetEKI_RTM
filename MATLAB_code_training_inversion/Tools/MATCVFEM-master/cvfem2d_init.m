function opt = cvfem2d_init(opt)

opt.cvfem.activeNode = zeros(opt.mesh.nnode,1);
opt.cvfem.activeNode(opt.bndry.inlet_flag==1) = 1;

opt.cvfem.activeElement = zeros(opt.mesh.nelem,1);
inlet_idx = find(opt.bndry.inlet_flag);
% At the begining, there are no active elements because no flow moves into
% the domain through the inlet yet. But in the flux calculation, elements
% involving inlet nodes needs to be highlighted somehow. 
% We assigned 0.5 to these elements to distinguish them from finite elements.
candidate = zeros(opt.mesh.nelem,1);
for i = 1 : nnz(opt.bndry.inlet_flag)
    ival = inlet_idx(i);
    for j = 2 : opt.mesh.has_node_i(ival,1)+1
        candidate(opt.mesh.has_node_i(ival,j))= 1;
    end
end
candidate_idx = find(candidate);
for i = 1 : length(candidate_idx)
    if sum(opt.bndry.inlet_flag(opt.mesh.elem(candidate_idx(i),:)))==2
        opt.cvfem.activeElement(candidate_idx(i)) = 0.5;
    end
end

opt.mesh.elem_including_inlet_edge = sparse(opt.cvfem.activeElement==0.5);

% filling time and filling factor
opt.cvfem.fTime = 0;                         % The filling time
opt.cvfem.fFactor = zeros(opt.mesh.nnode,1);          % the filling factor

% allocate a sparse matrix 
opt.cvfem.A = spalloc(opt.mesh.nnode,opt.mesh.nnode,opt.mesh.nnode*10);

opt.cvfem.nvoid = 0; % number of voids
opt.cvfem.void_volume = zeros(opt.mesh.nnode,2);
opt.cvfem.voidID = ones(opt.mesh.nnode,1);
