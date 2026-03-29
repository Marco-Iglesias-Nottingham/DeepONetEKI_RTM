function y = g_D(x)
y = ones(size(x,1),1)*0.000000;
for i = 1 : length(y)
   if x(i,2) == 0.000000
       y(i) = 76085.863636;
   end
end
