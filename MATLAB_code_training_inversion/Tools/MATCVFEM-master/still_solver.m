function [cvfem , bndry]= still_solver(cvfem,mesh,bndry,param,t)

[cvfem.activeElement, cvfem.activeNode, cvfem.newActiveElement, bndry.Dirichlet] = ...
    update_comp_domain(cvfem.activeElement,cvfem.activeNode,mesh.elem,mesh.nnode,...
    cvfem.new_filled_volume,mesh.has_node_i,bndry.neumann_flag,cvfem.fFactor);
%[cvfem.voidID, dryspot_flag] = find_dryspot(cvfem.fFactor, bndry.vent_idx, mesh.nb_nodes);
%if dryspot_flag 
%    [cvfem.void_volume, cvfem.nvoid] = void_partition(cvfem.voidID,...
%        cvfem.void_volume,mesh.Cnode,cvfem.V, cvfem.fFactor, cvfem.nvoid);
%end
cvfem.A = assembling_stiffness_tri(mesh.node, mesh.elem, cvfem.K, cvfem.A, cvfem.newActiveElement);

[u, b, freeNode, bdNode] = bcond_dirichlet(cvfem.A,bndry.Dirichlet,cvfem.activeNode,mesh.node,bndry.gd_filename,param,t);
%u = apply_ideal_gas_law(u,cvfem.void_volume,cvfem.V,cvfem.fFactor,cvfem.nvoid,bndry.vent_flag,bndry.pvent,bdNode);

b = modify_rhs(cvfem.A,u,b);
cvfem.u = solve(cvfem.A,u,b,freeNode);
%[cvfem.vx, cvfem.vy]=GetFlux(mesh,cvfem.u, cvfem.K);