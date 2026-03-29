function qn = local_flux_tri_inlet(x,v,fFactor,inlet_flag, bnd_flag,darcy_para)
if sum(bnd_flag)==3
    error('This code does not support an triangular element with three boundary nodes.');
end
qn = zeros(3,1);

o = sum(x)/3;
a = sum(x([1 2],:))/2;
b = sum(x([2 3],:))/2;
c = sum(x([1 3],:))/2;

n1 = [o(2) - a(2), a(1)-o(1)];
n2 = [o(2) - b(2), b(1)-o(1)];
n3 = [o(2) - c(2), c(1)-o(1)];


if sum(inlet_flag) == 2 % The element contains an edge lying on the inlet
    if inlet_flag(1) == 0
        n4 = fliplr(.5*x(2,:)-.5*x(3,:)).*[1 -1];
        temp1 = zeros(1,2);
        temp2 = n4-n2;
        temp3 = n4+n2;
        if sum(1-fFactor < eps)>= 1
            temp1 = temp1-n1+n3;
            temp2 = temp2 + n1;
            temp3 = temp3 - n3;
        end
    elseif inlet_flag(2) == 0
        n4 = fliplr(.5*x(3,:)-.5*x(1,:)).*[1 -1];
        temp1 = n4+n3;
        temp2 = zeros(1,2);
        temp3 = n4-n3;
        if sum(1-fFactor < eps)>= 1
            temp1 = temp1-n1;
            temp2 = temp2 + n1 - n2;
            temp3 = temp3 - n2;
        end
    elseif inlet_flag(3) == 0
        n4 = fliplr(.5*x(1,:)-.5*x(2,:)).*[1 -1];
        temp1 = n4-n1;
        temp2 = n4+n1;
        temp3 = zeros(1,2);
        if sum(1-fFactor < eps)>= 1
            temp1 = temp1+n3;
            temp2 = temp2-n2;
            temp3 = temp3 +n2- n3;
        end
    end
    
    qn(1) = temp1*v;
    qn(2) = temp2*v;
    qn(3) = temp3*v;
      
elseif sum(inlet_flag) == 1 % At this moment, we ignore elements having single inlet node.
    if sum(1-fFactor < eps)>= 1
        qn(1) = -(n1-n3)*v;
        qn(2) = -(n2-n1)*v;
        qn(3) = -(n3-n2)*v;
%     else
%         if inlet_flag(1) == 1
%             if sum(bnd_flag) == 1
%                 n4 = fliplr(.5*x(1,:)-.5*x(2,:)).*[1 -1];
%                 n5 = fliplr(.5*x(3,:)-.5*x(1,:)).*[1 -1];
%                 qn(1) = max((n4+n5)*v,0);
%             else % sum(bnd_flag) == 2
%                 if bnd_flag(2)
%                     n5 = fliplr(.5*x(3,:)-.5*x(1,:)).*[1 -1];
%                     qn(1) = max(n5*v,0);
%                 else %bnd_flag(3) is 1 
%                     n4 = fliplr(.5*x(1,:)-.5*x(2,:)).*[1 -1];
%                     qn(1) = max((n4)*v,0);
%                 end
%            
%             end
%         elseif inlet_flag(2) == 1
%             if sum(bnd_flag) == 1
%                 n4 = fliplr(.5*x(1,:)-.5*x(2,:)).*[1 -1];
%                 n5 = fliplr(.5*x(2,:)-.5*x(3,:)).*[1 -1];
%                 qn(2) = max((n4+n5)*v,0);
%             else % sum(bnd_flag) == 2
%                 if bnd_flag(1)
%                     n5 = fliplr(.5*x(2,:)-.5*x(3,:)).*[1 -1];
%                     qn(2) = max(n5*v,0);
%                 else %bnd_flag(1) is 1 
%                     n4 = fliplr(.5*x(1,:)-.5*x(2,:)).*[1 -1];
%                     qn(2) = max((n4)*v,0);
%                 end
%             end
%         elseif inlet_flag(3) == 1
%             if sum(bnd_flag) == 1
%                 n4 = fliplr(.5*x(2,:)-.5*x(3,:)).*[1 -1];
%                 n5 = fliplr(.5*x(3,:)-.5*x(1,:)).*[1 -1];
%                 qn(3) = max((n4+n5)*v,0);
%             else %sum(bnd_flag) == 2
%                 if bnd_flag(2)
%                     n5 = fliplr(.5*x(3,:)-.5*x(1,:)).*[1 -1];
%                     qn(3) = max(n5*v,0);
%                 else %bnd_flag(1) is 1 
%                     n4 = fliplr(.5*x(2,:)-.5*x(3,:)).*[1 -1];
%                     qn(3) = max((n4)*v,0);
%                 end
%             end
%         end
    end
else
    error('The triangular element must have at most two nodes on the inlet boundary.');
end

qn = qn*darcy_para.thickness;


