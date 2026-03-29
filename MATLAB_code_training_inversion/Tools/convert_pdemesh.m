function mesh = convert_pdemesh(p,e,t)
mesh.node = p';
mesh.elem = t(1:3,:)';
mesh.elem_type = 5*ones(size(t,2),1); 