function flag = isFilled(fFactor,N,vent_idx)
flag = all(abs(fFactor(vent_idx)-1) < 4*eps);