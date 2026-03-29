function A = local_stiffness_tri(x,K)
M = [ones(3,1) x];
Mi = inv(M);
area = .5*det(M);

A = Mi(2:3,:)'*[K(1) K(3); K(3) K(2)]*Mi(2:3,:)*area;


