function [net, poss, hardnegs] = fcnn_finetune_hnm(net,pos_data,neg_data,opts,varargin)
% Finetune the network  
% Train a CNN by SGD, with hard minibatch mining.
%
% modified from mdnet_finetune_hnm() in MDNet Library.
%

opts.conserveMemory = true ;
opts.sync = true ;
opts.nesterovUpdate = false ;
opts.solver = [];

opts.maxiter = 30;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts.batchSize_hnm = 256;
opts.batchAcc_hnm = 4;

opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;

opts = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
for l=1:length(net.layers)
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        switch (net.layers(l).name)
            case {'fusion_concat','fusion_concat2'}
                net.params(f_ind).learningRate = 1;
                net.params(b_ind).learningRate = 2;
            case {'fusion_concat3'}
                net.params(f_ind).learningRate = 10;
                net.params(b_ind).learningRate = 20;
        end
    end
end

%% initilizing
if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
res = [] ;

n_pos = size(pos_data,4);
n_neg = size(neg_data,4);
train_pos_cnt = 0;
train_neg_cnt = 0;

% extract positive batches
train_pos = [];
remain = opts.batch_pos*opts.maxiter;
while(remain>0)
    if(train_pos_cnt==0)
        train_pos_list = randperm(n_pos)';
    end
    train_pos = cat(1,train_pos,...
        train_pos_list(train_pos_cnt+1:min(end,train_pos_cnt+remain)));
    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
    remain = opts.batch_pos*opts.maxiter-length(train_pos);
end

% extract negative batches
train_neg = [];
remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter;
while(remain>0)
    if(train_neg_cnt==0)
        train_neg_list = randperm(n_neg)';
    end
    train_neg = cat(1,train_neg,...
        train_neg_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
    remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter-length(train_neg);
end

% learning rate
lr = opts.learningRate ;

% for saving positives
poss = [];

% for saving hard negatives
hardnegs = [];

% objective fuction
objective = zeros(1,opts.maxiter);
state = [] ;
% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
  state.solverState = cell(1, numel(net.params)) ;
  state.solverState(:) = {0} ;
end

parserv = [];


%% training on training set
% fprintf('\n');
for t=1:opts.maxiter
    fprintf('\ttraining batch %3d of %3d ...\n ', t, opts.maxiter) ;
    iter_time = tic ;
    
    % ----------------------------------------------------------------------
    % hard negative mining
    % ----------------------------------------------------------------------
    score_hneg = zeros(opts.batchSize_hnm*opts.batchAcc_hnm,1);
    hneg_start = opts.batchSize_hnm*opts.batchAcc_hnm*(t-1);
    for h=1:opts.batchAcc_hnm
        batch = neg_data(:,:,:,...
            train_neg(hneg_start+(h-1)*opts.batchSize_hnm+1:hneg_start+h*opts.batchSize_hnm));
        if opts.useGpu
            batch = gpuArray(batch) ;
        end
        
        % backprop  
        labels = ones(opts.batchSize_hnm,1,'single') ;
        inputs = {'input', batch, 'label', labels};
        net.mode = 'test' ;
        net.eval(inputs);
        res = net.vars(strcmp({net.vars.name},'fusion_concat3')).value;
%         net.layers{end}.class = ones(opts.batchSize_hnm,1,'single') ;
%         res = vl_simplenn(net, batch, [], res, ...
%             'disableDropout', true, ...
%             'conserveMemory', opts.conserveMemory, ...
%             'sync', opts.sync) ;
        score_hneg((h-1)*opts.batchSize_hnm+1:h*opts.batchSize_hnm) = ...
            squeeze(gather(res(1,1,2,:)));
    end
    [~,ord] = sort(score_hneg,'descend');
    hnegs = train_neg(hneg_start+ord(1:opts.batch_neg));
    im_hneg = neg_data(:,:,:,hnegs);
%     fprintf('hnm: %d/%d, ', opts.batch_neg, opts.batchSize_hnm*opts.batchAcc_hnm) ;
    hardnegs = [hardnegs; hnegs];
    opts.derOutputs = {'objective', 1} ;
    % ----------------------------------------------------------------------
    % get next image batch and labels
    % ----------------------------------------------------------------------
    poss = [poss; train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)];
    
    batch = cat(4,pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)),...
        im_hneg);
    labels = [2*ones(opts.batch_pos,1,'single');ones(opts.batch_neg,1,'single')];
    if opts.useGpu
        batch = gpuArray(batch) ;
    end
    
    opts.numSubBatches = 1 ;
    inputs = {'input', batch, 'label', labels};
    net.mode = 'normal' ;
    net.accumulateParamDers = (t ~= 1) ;
    net.eval(inputs, opts.derOutputs, 'holdOn', t < opts.numSubBatches);
    state = accumulateGradients(net, state, opts, opts.batchSize, parserv) ;    
end % next batch

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
% numGpus = numel(params.gpus) ;
% otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)

  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = net.params(p).der ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/batchSize/net.params(p).fanout),  parDer) ;

    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.params(p).value) ;

        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{p} = vl_taccum(...
            params.momentum, state.solverState{p}, ...
            -1, parDer) ;

          % Nesterov update (aka one step ahead).
          if params.nesterovUpdate
            delta = params.momentum * state.solverState{p} - parDer ;
          else
            delta = state.solverState{p} ;
          end

          % Update parameters.
          net.params(p).value = vl_taccum(...
            1,  net.params(p).value, thisLR, delta) ;

        else
          % call solver function to update weights
          [net.params(p).value, state.solverState{p}] = ...
            params.solver(net.params(p).value, state.solverState{p}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

