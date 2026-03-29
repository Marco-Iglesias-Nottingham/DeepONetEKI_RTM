function qn = local_flux_tri(normal_vec,v,darcy_para)
qn = zeros(3,1);


qn(1) = normal_vec(:,[1 2])*v;
qn(2) = normal_vec(:,[3 4])*v;
qn(3) = normal_vec(:,[5 6])*v;

qn = qn*darcy_para.thickness;
