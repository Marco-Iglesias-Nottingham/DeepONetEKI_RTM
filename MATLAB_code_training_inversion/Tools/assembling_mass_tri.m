function M = assembling_mass_tri(node, elem)
% M = assembling_mass_tri(node, elem)
% Assemble mass matrix on triangular mesh
% node: Nx2 array of coordinates
% elem: Mx3 array of element connectivity

N = size(node,1);
M = spalloc(N,N,N*9);  % preallocate sparse matrix

rows = zeros(9*size(elem,1),1);
cols = zeros(9*size(elem,1),1);
vals = zeros(9*size(elem,1),1);
nnzs = 0;

for t = 1:size(elem,1)
    % nodes of triangle t
    vidx = elem(t,:);
    v1 = node(vidx(1),:);
    v2 = node(vidx(2),:);
    v3 = node(vidx(3),:);
    
    % compute area of triangle
    area = polyarea([v1(1),v2(1),v3(1)], [v1(2),v2(2),v3(2)]);
    
    % local mass matrix
    Mt = area/12 * [2 1 1; 1 2 1; 1 1 2];
    
    % assemble into global matrix
    for i = 1:3
        for j = 1:3
            nnzs = nnzs + 1;
            rows(nnzs) = vidx(i);
            cols(nnzs) = vidx(j);
            vals(nnzs) = Mt(i,j);
        end
    end
end

M = M + sparse(rows,cols,vals,N,N);
