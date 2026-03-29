function neumann_flag = ...
    find_nuemann_points(bnd_nodes,inlet_flag, vent_flag)
nnode = length(bnd_nodes);
nbnd_node = nnz(bnd_nodes);
candidate = find(bnd_nodes);
neumann_flag = zeros(nnode,1);
for i = 1 : nbnd_node
    ci = candidate(i);
    if ~inlet_flag(ci)&& ~vent_flag(ci)
        neumann_flag(ci) = 1;
    end
end
neumann_flag = sparse(neumann_flag);