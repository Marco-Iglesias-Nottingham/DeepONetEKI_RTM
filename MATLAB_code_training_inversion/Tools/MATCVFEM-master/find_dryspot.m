function [voidID, dryspot_flag] = find_dryspot(fFactor, vent_idx, nb_nodes)
nnode = length(fFactor);

queue   = zeros(nnode,1);
isInQueue = false(nnode,1);
lastIdx = 0;
voidID  = zeros(nnode,1);

voidID(fFactor == 1) = 0.5; % Filled volume

for i = 1 : length(vent_idx)
    vi = vent_idx(i);
    if fFactor(vi) < 1
        lastIdx = lastIdx + 1;
        queue(lastIdx) = vi;
        isInQueue(vi) = true;
    end
end

if lastIdx > 0
    startIdx = 1;
    while 1
        qs = queue(startIdx);
        
        nbqs =nb_nodes{qs};
        for i = 1 : length(nbqs)
            nbi = nbqs(i);
            if fFactor(nbi) < 1 && ~isInQueue(nbi)
                lastIdx = lastIdx+1;
                queue(lastIdx) = nbi;
                isInQueue(nbi) = true;
            end
        end
        startIdx = startIdx + 1;
        if startIdx > lastIdx
            break;
        end
        
        
    end
    voidID(queue(1:lastIdx)) = 1;
end

if isempty(find(voidID==0))
    dryspot_flag = false;
else
    dryspot_flag = true;
end


