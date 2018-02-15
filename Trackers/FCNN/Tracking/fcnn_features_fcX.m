function [ feat ] = fcnn_features_fcX(net, ims, opts)
% Get Scores 
% Compute CNN scores from input features.
%
% modified from fmdnet_features_fcX() in MDNet Library.
% 

n = size(ims,4);
nBatches = ceil(n/opts.batchSize);
% 
net_var = dagnn.DagNN();
net_var.layers = [net.layers(1:6) net.layers(9:end)];
net_var.vars = net.vars(1:end-3);
net_var.params = net.params;
net_var = dagnn.DagNN().loadobj(net_var);
net_var = net_var.saveobj;
net_var = dagnn.DagNN().loadobj(net_var);
for i=1:nBatches
    batch = ims(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    net_var.mode = 'test';
    net_var.eval({'input', batch}) ;
    res = net_var.vars(strcmp({net_var.vars.name},'prob')).value;
    
    f = gather(res) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i)) = f;
end