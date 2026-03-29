function [fFactor, new_filled_volume] = update_filling_factor(fFactor,Q,dt,V)
pos_q_idx = find(Q>0);


fFactor_new = min(ones(length(pos_q_idx),1),fFactor(pos_q_idx)+ dt*Q(pos_q_idx)./V(pos_q_idx));

fFactor_new(1-fFactor_new<1e+3*eps) = 1;

new_filled_volume = pos_q_idx(fFactor_new>fFactor(pos_q_idx)& fFactor_new == 1);

fFactor(pos_q_idx) = fFactor_new;
% 
% new_filled_volume = zeros(size(fFactor));
% for i = 1 : length(pos_q_idx)
%     qi = pos_q_idx(i);
%     ff_old = fFactor(qi);
%     fFactor(qi) = min(1,fFactor(qi) + dt*Q(qi)/V(qi));
%     
%     if 1-fFactor(qi) < eps
%         fFactor(qi) = 1;
%     end
%     if f
% end