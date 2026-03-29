function dt = compute_time_increment(Q,fFactor,V)
nfFactor = 1 - fFactor;

candidates = find((Q>eps) & (nfFactor > eps));
if isempty(candidates)
    warning('No positive flux found.')
    dt = 0;
else
    dt = min(nfFactor(candidates).*V(candidates)./Q(candidates));
end