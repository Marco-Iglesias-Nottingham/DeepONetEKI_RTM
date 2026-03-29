function [cvfem , bndry]= still_solver2(cvfem,mesh,bndry)

[cvfem.activeElement, cvfem.activeNode, cvfem.newActiveElement, bndry.Dirichlet] = ...
    update_comp_domain2(cvfem.activeElement,cvfem.activeNode,mesh.elem,mesh.nnode,...
    cvfem.new_filled_volume,mesh.has_node_i,bndry.neumann_flag,cvfem.fFactor);

cvfem.A = assembling_stiffness_tri(mesh.node, mesh.elem, cvfem.K, cvfem.A, cvfem.newActiveElement);

[u, b, freeNode] = bcond_dirichlet(cvfem.A,bndry.Dirichlet,cvfem.activeNode,mesh.node,bndry.gd_filename);
b = modify_rhs(cvfem.A,u,b);
cvfem.u = solve(cvfem.A,u,b,freeNode);