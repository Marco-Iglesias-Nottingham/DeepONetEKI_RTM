function opt = setK(opt,K)
if size(K,2) == 1
    opt.cvfem.K = [K K zeros(size(K,1),1)];
    opt.cvfem.ktype = 1;
elseif size(K,2) == 2
    opt.cvfem.K = [K zeros(size(K,1),1)];
    opt.cvfem.ktype = 2;
elseif size(K,2) == 3
    opt.cvfem.K = K;
    opt.cvfem.ktype = 3;
else
    error('unsupported dimension of K!')
end