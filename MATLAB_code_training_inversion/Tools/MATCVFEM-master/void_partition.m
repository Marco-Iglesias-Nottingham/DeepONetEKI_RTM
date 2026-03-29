function [void_volume, nvoid] = void_partition(voidID, void_volume, Cnode, V, fFactor, nvoid)
candidate = find(voidID == 0);
biggestIdx = max(candidate);
n = length(candidate);
idx = 1;

while 1
    ci = candidate(idx);
    % new void occurs
    if void_volume(ci,1) == 0
        nvoid = nvoid + 1;
        queue = zeros(n,1);
        isInQueue = false(biggestIdx,1);
        qIdx = 1;
        volume = V(ci)*(1-fFactor(ci));
        queue(qIdx) = ci;
        isInQueue(ci) = true;
        nb_nodes = find(Cnode(ci,:));
        for j = 1 : length(nb_nodes)
            nb =nb_nodes(j);
            if voidID(nb) == 0 && ~isInQueue(nb)
                qIdx = qIdx + 1;
                queue(qIdx) = nb;
                isInQueue(nb) = true;
                volume = volume + V(nb)*(1-fFactor(nb));
            end
        end
        if qIdx > 1
            startIdx = 2;
            while 1
                qs = queue(startIdx);
                nb_nodes = find(Cnode(qs,:));
                for i = 1 : length(nb_nodes)
                    nb = nb_nodes(i);
                    if voidID(nb) == 0 && ~isInQueue(nb)
                        qIdx = qIdx + 1;
                        queue(qIdx) = nb;
                        isInQueue(nb) = 1;
                        volume = volume + V(nb)*(1-fFactor(nb));
                    end
                end
                startIdx = startIdx + 1;
                if startIdx > qIdx
                    break;
                end
                    
            end
        end
        void_volume(queue(1:qIdx),1) = volume;
        void_volume(queue(1:qIdx),2) = nvoid;
    end
    idx = idx + 1;
    if idx > n
        break
    end
end