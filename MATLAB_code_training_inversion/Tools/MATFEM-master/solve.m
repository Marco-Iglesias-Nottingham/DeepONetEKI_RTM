function u = solve(A,u,b,freeNode)
u(freeNode) = A(freeNode,freeNode)\b(freeNode);
