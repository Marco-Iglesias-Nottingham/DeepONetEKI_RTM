function u = apply_ideal_gas_law(u,void_volume,V,fFactor,nvoid,vent_flag,pvent,bdNode)
% void_volume contains the initial volume of void when it occurs. Vn stors
% the volume of void at current time step.
Vn = zeros(size(V));

for i = 1 : nvoid
    void_i = find(void_volume(:,2)==i);
    temp = 0;
    for j = 1 : length(void_i)
        void_ij = void_i(j);
        temp = temp + (1-fFactor(void_ij))*V(void_ij); % compute unsaturated volumes, which is the volume of void.
    end
    Vn(void_i) = temp;
end

% apply the ideal gas law 
for i = 1 : length(bdNode)
    bdn_i = bdNode(i);
    if void_volume(bdn_i) > 0 && ~vent_flag(bdn_i) && Vn(bdn_i)>0
        u(bdn_i) = pvent*void_volume(bdn_i)/Vn(bdn_i);
    end
end