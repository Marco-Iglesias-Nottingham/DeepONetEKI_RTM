function normalvec = compute_normals(elem,node)
    
% Find the mid points and centre point
o = (node(elem(:,1),:) + node(elem(:,2),:) ...
    + node(elem(:,1),:))/3;
a = (node(elem(:,1),:) + node(elem(:,2),:))/2;
b = (node(elem(:,2),:) + node(elem(:,3),:))/2;
c = (node(elem(:,3),:) + node(elem(:,1),:))/2;

% Compute normal vectors
n1 = [o(:,2)-a(:,2) a(:,1)-o(:,1)];
n2 = [o(:,2)-b(:,2) b(:,1)-o(:,1)];
n3 = [o(:,2)-c(:,2) c(:,1)-o(:,1)];

n31 = n3 - n1;
n12 = n1 - n2;
n23 = n2 - n3;
normalvec = [n31 n12 n23];