function [u, b, freeNode, bdNode] = bcond_dirichlet(A,Dirichlet,activeNode, node, gd_filename,param,t)
g_D = str2func(gd_filename);
N = size(A,1);
b = zeros(N,1);
isBdNode = false(N,1);
isBdNode(Dirichlet==1) = true;
bdNode = find(isBdNode);
freeNode = find(~isBdNode);
freeNode = freeNode(activeNode(freeNode)==1);
u = zeros(N,1);

u(bdNode) = g_D(node(bdNode,:),param,t);

