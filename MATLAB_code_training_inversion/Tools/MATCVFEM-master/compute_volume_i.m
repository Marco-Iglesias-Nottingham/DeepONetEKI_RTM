function v = compute_volume_i(xi, thickness)

A = .5*det([1 xi(1,1) xi(1,2);...
         1 xi(2,1) xi(2,2);...
         1 xi(3,1) xi(3,2)]);
         
v = ones(3,1)*A/3*thickness;

