function[ndim] = caculatePerVar(variance, cutoff)
% choose num of dims to keep predefined percentage of variance
% if cutoff >= 1, then nmax = ndim_max;
ndim = 0;
ndim_max = length(variance);
varTotal = sum(variance);
if cutoff >= 1
    ndim = ndim_max;
else
    for i=1:ndim_max
        tmp2 = variance(1:i);
        percent(i) = sum(tmp2) / varTotal;
        if ndim == 0 && percent(i)>=cutoff
            ndim = i;
            continue;
        end
    end
end

if ndim == 0
    ndim = ndim_max-1;
end


