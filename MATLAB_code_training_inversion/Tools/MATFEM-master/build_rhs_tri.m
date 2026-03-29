function b  = build_rhs_tri(N,node,elem,area,fname)
f = str2func(fname);
mid1 = (node(elem(:,2),:)+node(elem(:,3),:))/2;
mid2 = (node(elem(:,3),:)+node(elem(:,1),:))/2;
mid3 = (node(elem(:,1),:)+node(elem(:,2),:))/2;

bt1 = area.*(f(mid2)+f(mid3))/6;
bt2 = area.*(f(mid3)+f(mid1))/6;
bt3 = area.*(f(mid1)+f(mid2))/6;
b = accumarray(elem(:),[bt1;bt2;bt3],[N 1]);

