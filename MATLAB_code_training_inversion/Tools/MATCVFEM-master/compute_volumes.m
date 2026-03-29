function V  = compute_volumes(node,elem,thickness)
N = size(node,1);
NT = size(elem,1);

V = zeros(N,1);
for i = 1 : NT
    elem_i = elem(i,1:3);
    V(elem_i) = V(elem_i) + compute_volume_i(node(elem_i,:), thickness);
end


