function b = modify_rhs(A,u,b)
b = b - A*u;