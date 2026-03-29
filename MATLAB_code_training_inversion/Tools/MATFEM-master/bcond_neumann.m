function b = bcond_neumann(b, node, Neumann, gn_filename)
g_N = str2func(gn_filename);
if (~isempty(Neumann))
    Nve = node(Neumann(:,1),:) - node(Neumann(:,2),:);
    edgeLength = sqrt(sum(Nve.^2,2));
    mid = (node(Neumann(:,1),:) + node(Neumann(:,2),:))/2;
    accumarray([Neumann(:),ones(2*size(Neumann,1),1)],repmat(edgeLength.*g_N(mid)/2,2,1),[N,1]);
    b = b + accumarray([Neumann(:),ones(2*size(Neumann,1),1)],repmat(edgeLength.*g_N(mid)/2,2,1),[N,1]);   
end