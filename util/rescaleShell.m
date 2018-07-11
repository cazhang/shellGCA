function [FV_out,ratio_out] = rescaleShell(FV1, FV2, Topology, Ev, Ef, Eo, boundaryedges, ratio, options)
% rescale shell via ratio
% ratio < 1: shortening
% ratio > 1: prolongation
K = floor(ratio);
e = ratio - K;
if ~isfield(options, 'useMem')
    options.useMem = false;
end
if ratio <= 1
    disp('shortening');
    pcw(1) = 1-ratio;
    pcw(2) = ratio;
    options.datweights = pcw;
    FV_out = MultiResElasticAvOfTwo( FV1,FV2,Topology,Ev,Ef,Eo,boundaryedges,options );

elseif ratio > 1 && ratio < 2
    disp('short shooting');
    pcw(1) = ratio-1;
    pcw(2) = 1;
    options.expweights = pcw;
    options.step = 1;
    FV_out = TestShellExp2( FV1, FV2,Topology,Ev,Ef,Eo,boundaryedges,options );
elseif ratio >= 2 && e == 0
    disp('long intergral shooting');
    pcw(1)=1;
    pcw(2)=1;
    options.expweights = pcw;
    options.step = K-1;
    FV_out = TestShellExp2( FV1, FV2,Topology,Ev,Ef,Eo,boundaryedges,options );
elseif ratio > 2 && e > 0
    % get intergal shooting last shape(FV_exp), and second last shape(FV_pre)
    disp('long decimal shooting');
    pcw(1)=1;
    pcw(2)=1;
    options.expweights = pcw;
    options.step = K-1;
    [FV_exp, FV_pre] = TestShellExp2( FV1, FV2,Topology,Ev,Ef,Eo,boundaryedges,options );
    pcw(1)=e;
    pcw(2)=1;
    options.expweights = pcw;
    options.step = 1;
    FV_out = TestShellExp2( FV_pre, FV_exp,Topology,Ev,Ef,Eo,boundaryedges,options );
end
if nargout > 1
    % report real ratio
    if options.useMem
        [dis_ref] = geomDSD(FV1, FV2,Topology,Ev,Ef,options.eta,1);
        [dis_out] = geomDSD(FV1, FV_out,Topology,Ev,Ef,options.eta,1);
    else
        [dis_ref] = DSD(FV1, FV2,Topology,Ev,Ef,options.eta,1);
        [dis_out] = DSD(FV1, FV_out,Topology,Ev,Ef,options.eta,1);
    end
    
    len_ref = sqrt(dis_ref);
    len_out = sqrt(dis_out);
    ratio_out = len_out / len_ref;
end

    
end