function [result_fusion] = run_fusion(frames, gt_init)
%% Run Fusion
% This function is used to load the trained model, run the network and do
% online updates and return the list of bounding boxes for the sequence
% Inputs
%   frames - frame list of a particular sequence
%   gt_init - first bounding box of the ground truth. 

t = load('./Models/louise.mat');
net = t.net;

opts.bbreg = 1;
img = imread(char(frames(1)));
nFrames = length(frames);
if(size(img,3)==1), img = cat(3,img,img,img); end
targetLoc = gt_init;
result_fusion = zeros(nFrames, 4); result_fusion(1,:) = targetLoc;

%Get the fully connected layers
net_fc = dagnn.DagNN();
net_fc.layers = [net.layers(21:end)];
net_fc.vars = [net.vars(22:end)];
count_f = 1;

for l=21:length(net.layers)
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        net_fc.params(count_f) = net.params(f_ind);
        net_fc.params(count_f + 1) = net.params(b_ind);
        count_f = count_f + 2;
    elseif(strcmp(class(net.layers(l).block), 'dagnn.BatchNorm'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        v_ind = net.layers(l).paramIndexes(3);
        net_fc.params(count_f) = net.params(f_ind);
        net_fc.params(count_f + 1) = net.params(b_ind);
        net_fc.params(count_f + 2) = net.params(v_ind);
        count_f = count_f + 3;
    end
end

addVar(net_fc, 'input');
net_fc.vars(net_fc.getVarIndex('input')).fanout = net_fc.vars(net_fc.getVarIndex('input')).fanout + 1 ;
net_fc.layers(1).inputs = {'input'};
net_fc.layers(1).inputIndexes = net_fc.getVarIndex('input');

net_fc = net_fc.saveobj;
net_fc = dagnn.DagNN().loadobj(net_fc);

%Get the conv layers
net_conv = dagnn.DagNN();
net_conv.layers = [net.layers(1:20)];
net_conv.vars = [net.vars(1:21)];
count_f = 1;

for l=1:19
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        net_conv.params(count_f) = net.params(f_ind);
        net_conv.params(count_f + 1) = net.params(b_ind);
        count_f = count_f + 2;
    elseif(strcmp(class(net.layers(l).block), 'dagnn.BatchNorm'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        v_ind = net.layers(l).paramIndexes(3);
        net_conv.params(count_f) = net.params(f_ind);
        net_conv.params(count_f + 1) = net.params(b_ind);
        net_conv.params(count_f + 2) = net.params(v_ind);
        count_f = count_f + 3;
    end
end

net_conv = net_conv.saveobj;
net_conv = dagnn.DagNN().loadobj(net_conv);

% for l = 2:length(net_fc.layers) - 1
%     var = net.layers(l + 19).inputs;
%     net_fc.layers(l).inputIndexes = net_fc.getVarIndex(var);
%     var_out = net.layers(l + 19).inputs; 
% end


opts.bbreg_nSamples = 1000;
% 
%% Extract training examples
fprintf('  extract features...\n');
opts.learningRate_init = 0.0001; 
opts.maxiter_init = 30;

% % draw positive/negative samples
opts.nPos_init = 500;
opts.nNeg_init = 5000;
opts.posThr_init = 0.7;
opts.negThr_init = 0.5;

% data gathering policy
opts.nFrames_long = 100; % long-term period
opts.nFrames_short = 20; % short-term period

% cropping policy
opts.input_size = 107;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;

% scaling policy
opts.scale_factor = 1.05;

% sampling policy
opts.nSamples = 256;
opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

% set image size
opts.imgSize = size(img);
opts.useGpu = false;
opts.batchSize_test = 256;

% learning policy
opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;


% update policy
opts.learningRate_update = 0.0003; 
opts.maxiter_update = 10;

opts.nPos_update = 50;
opts.nNeg_update = 200;
opts.posThr_update = 0.7;
opts.negThr_update = 0.3;

opts.update_interval = 10; % interval for long-term update

%% Train a bbox regressor
if(opts.bbreg)
    pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    feat_conv = fcnn_features_convX(net_conv, img, pos_examples, opts);
    
    X = permute(gather(feat_conv),[4,3,1,2]);
    X = X(:,:);
    bbox = pos_examples;
    bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
end

pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
r = overlap_ratio(pos_examples,targetLoc);
pos_examples = pos_examples(r>opts.posThr_init,:);
pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
    gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

feat_conv = fcnn_features_convX(net_conv, img, examples, opts);
pos_data = feat_conv(:,:,:,pos_idx);
neg_data = feat_conv(:,:,:,neg_idx);

%% Learning CNN
fprintf('  training cnn...\n');
net_fc = fcnn_finetune_hnm(net_fc,pos_data,neg_data,opts,...
    'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init,opts);

%% Prepare training data for online update
total_pos_data = cell(1,1,1,nFrames);
total_neg_data = cell(1,1,1,nFrames);

neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

feat_conv = fcnn_features_convX(net_conv, img, examples, opts);
total_pos_data{1} = feat_conv(:,:,:,pos_idx);
total_neg_data{1} = feat_conv(:,:,:,neg_idx);

success_frames = 1;
trans_f = opts.trans_f;
scale_f = opts.scale_f;

for To = 2:length(frames)
    img = imread(frames{To});
 
    spf = tic;
    
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    
    samples = gen_samples('gaussian', targetLoc, opts.nSamples, opts, trans_f, scale_f);
    feat_conv = fcnn_features_convX(net_conv, img, samples, opts);
    
    % evaluate the candidates
    feat_fc = fcnn_features_fcX(net_fc, feat_conv, opts);
    feat_fc = squeeze(feat_fc)';
    [scores,idx] = sort(feat_fc(:,2),'descend');
    target_score = mean(scores(1:5));
    targetLoc = round(mean(samples(idx(1:5),:)));
    
    % final target
    result_fusion(To,:) = targetLoc;
    
    % extend search space in case of failure
    if(target_score<0)
        trans_f = min(1.5, 1.1*trans_f);
    else
        trans_f = opts.trans_f;
    end
    
    % bbox regression
    if(opts.bbreg && target_score>0)
        X_ = permute(gather(feat_conv(:,:,:,idx(1:5))),[4,3,1,2]);
        X_ = X_(:,:);
        bbox_ = samples(idx(1:5),:);
        pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
        result_fusion(To,:) = round(mean(pred_boxes,1));
    end
    
    %% Prepare training data
    if(target_score>0)
        pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_update*2, opts, 0.1, 5);
        r = overlap_ratio(pos_examples,targetLoc);
        pos_examples = pos_examples(r>opts.posThr_update,:);
        pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);
        
        neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
        r = overlap_ratio(neg_examples,targetLoc);
        neg_examples = neg_examples(r<opts.negThr_update,:);
        neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);
        
        examples = [pos_examples; neg_examples];
        pos_idx = 1:size(pos_examples,1);
        neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
        
        feat_conv = fcnn_features_convX(net_conv, img, examples, opts);
        total_pos_data{To} = feat_conv(:,:,:,pos_idx);
        total_neg_data{To} = feat_conv(:,:,:,neg_idx);
        
        success_frames = [success_frames, To];
        if(numel(success_frames)>opts.nFrames_long)
            total_pos_data{success_frames(end-opts.nFrames_long)} = single([]);
        end
        if(numel(success_frames)>opts.nFrames_short)
            total_neg_data{success_frames(end-opts.nFrames_short)} = single([]);
        end
    else
        total_pos_data{To} = single([]);
        total_neg_data{To} = single([]);
    end
    
    %% Network update
    if((mod(To,opts.update_interval)==0 || target_score<0) && To~=nFrames)
        if (target_score<0) % short-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
        else % long-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_long+1):end)));
        end
        neg_data = cell2mat(total_neg_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
        
        [net_fc] = fcnn_finetune_hnm(net_fc,pos_data,neg_data,opts,...
            'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update);
    end
    
    spf = toc(spf);
    fprintf('%d, %f seconds\n',To,spf);
end

