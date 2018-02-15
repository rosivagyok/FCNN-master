function [ feat ] = fcnn_features_convX(net, img, boxes, opts)
% Get Features
% Extract CNN features from bounding box regions of an input image.
%
% modified from mdnet_features_convX() in MDNet Library.
% 

n = size(boxes,1);

ims = mdnet_extract_regions(img, boxes, opts);
nBatches = ceil(n/opts.batchSize_test);

for i=1:nBatches
%     fprintf('extract batch %d/%d...\n',i,nBatches);
    
    batch = ims(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    net.eval({'input', batch}) ;
    res = net.vars(strcmp({net.vars.name},'relu')).value;
    
    f = gather(res) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i)) = f;
end