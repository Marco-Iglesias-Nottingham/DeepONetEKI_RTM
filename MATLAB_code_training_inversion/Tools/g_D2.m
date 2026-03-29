function y = g_D2(x,param,t)
y = ones(size(x,1),1)*0.000000;

A=param(1);
frac=param(2);
lambda=param(3);
beta=param(4);


weibull_cdf = @(A,frac,lambda,beta ,x) frac*A + (A - frac*A) .* (1 - exp(-(x ./lambda).^beta));

for i = 1 : length(y)
   if x(i,1) == 0.000000
       %y(i) = p_I;
       y(i)=weibull_cdf(A,frac,lambda,beta,t);% p_I*(1-0.4*exp(-gamma*t));
   end
end
