function [inlet_flag, inlet_pos, Dirichlet] = rec_inlet_location(node,bnd_node)
nnode = size(node,1);
nbnd_node = nnz(bnd_node);
inlet = zeros(nbnd_node,1);
inlet_idx = 0;
candidate = find(bnd_node);
for i = 1 : nbnd_node
    ci = candidate(i);
    if node(ci,1) == 0.000000 
        inlet_idx = inlet_idx+1;
        inlet(inlet_idx) = ci;
    end
end
inlet_flag = sparse(nnode,1);
inlet_flag(inlet(1:inlet_idx)) = 1;
inlet_pos = node(inlet_flag==1,:);

Dirichlet =zeros(nnode,1);
Dirichlet(inlet_flag==1) = 1;
