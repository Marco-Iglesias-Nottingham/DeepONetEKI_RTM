function Q = update_flow_rate_tri(opt)

Q = zeros(opt.mesh.nnode,1);
candidate_elem = zeros(opt.mesh.nelem,1);
u = opt.cvfem.u;
K = opt.cvfem.K;
darcy = opt.darcy;
fFactor = opt.cvfem.fFactor;
normal_vec = opt.mesh.normal_vec;

candidate_node = find(~xor(opt.cvfem.activeNode==1,1-fFactor > 1e-15));

for i = 1 : length(candidate_node)
    candidate = candidate_node(i);
    for j = 2 : opt.mesh.has_node_i(candidate,1)+1
        elem = opt.mesh.has_node_i(candidate,j);
        if opt.cvfem.activeElement(elem) >= 0.5
            candidate_elem(opt.mesh.has_node_i(candidate,j)) = 1;
        end
    end
end

candidate_elem = find(candidate_elem);

inlet_elem = candidate_elem(opt.mesh.elem_including_inlet_edge(candidate_elem)==1);
other_elem = candidate_elem(opt.mesh.elem_including_inlet_edge(candidate_elem)==0);

for i = 1 : length(inlet_elem)
    elem_i = opt.mesh.elem(inlet_elem(i),:);
    node_i = opt.mesh.node(elem_i,:);
    vi = velocity_centre_tri(u(elem_i),node_i,K(inlet_elem(i),:),darcy, darcy.phi(inlet_elem(i)));
    Q(elem_i) = Q(elem_i) + local_flux_tri_inlet(node_i,vi,fFactor(elem_i),opt.bndry.inlet_flag(elem_i),opt.mesh.bndry_nodes(elem_i),darcy);
end

for i = 1 : length(other_elem)
    elem_idx = other_elem(i);
    elem_i = opt.mesh.elem(elem_idx,:);
    node_i = opt.mesh.node(elem_i,:);
    vi = velocity_centre_tri(u(elem_i),node_i,K(elem_idx,:),darcy, darcy.phi(elem_idx));
   
    
    Q(elem_i) = Q(elem_i) + local_flux_tri(...
        normal_vec(elem_idx,:),...
        vi,...
        darcy);
end
