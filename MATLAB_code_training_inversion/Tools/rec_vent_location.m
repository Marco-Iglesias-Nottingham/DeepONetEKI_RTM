function  [vent_flag, vent_elem] = rec_vent_location(node,bnd_node)
nnode = size(node,1);

nbnd_node = nnz(bnd_node);
vent = zeros(nbnd_node,1);
vent_idx = 0;
candidate = find(bnd_node);
for i = 1 : nbnd_node
    ci = candidate(i);
    if node(ci,1) == 0.300000
        vent_idx = vent_idx+1;
        vent(vent_idx) = ci;
    end
end
vent_flag = sparse(nnode,1);
vent_flag(vent(1:vent_idx)) = 1;

vent_elem = 0;
