function A = assembling_stiffness_tri(node, elem, K, A, newElement)
% A = assembling_stiffness_tri(N,node,elem,K) For general use
%
% [A, iArray, jArray, sArray, arrayIndex) = assembling_stiffness_tri(N,node,elem,K,iArray, jArray, sArray, newElement, arrayIndex]
% When you need to update the computation domain as time step evoles, keep
% the matrix structure vectors for the next time step. 

% Each row of K consists of three components, Kx, Ky, and Kxy (=Kyx)
% If K has one column, the K is constant multiple of an identity matrix.
% If K has two columns, the K is a diagonal matrix of the form [Kx 0; 0 Ky].
% If K has three columns, then K is a symmetric matrix of the form [Kx Kxy; Kyx Ky]
N = size(node,1);
if size(K,2) == 1
    K = [K K zeros(size(K))];  
elseif size(K,2) == 2
    K = [K zeros(size(K))];
end

if ~(nargin == 3 || nargin == 5)
    error('Missing arguments. help assembling_stiffness_tri')
end

if nargin == 3
    A = spalloc(N,N,N*9);
    newAE = 1:size(elem,1);    
else
    newAE = find(newElement);
end
nNewAE = length(newAE);
rows = zeros(9*nNewAE,1);
cols = zeros(9*nNewAE,1);
vals = zeros(9*nNewAE,1);
nnzs = 0;

for i = 1 : length(newAE)
    t = newAE(i);
    At = local_stiffness_tri(node(elem(t,1:3),:),K(t,:));
    for ti = 1 : 3
        for tj = 1 : 3
            nnzs = nnzs + 1;
            rows(nnzs) = elem(t,ti);
            cols(nnzs) = elem(t,tj);
            vals(nnzs) = At(ti,tj);
        end
    end
end


A = A + sparse(rows,cols,vals,N,N);
