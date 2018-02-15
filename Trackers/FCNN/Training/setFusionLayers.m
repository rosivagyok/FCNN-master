function [ net ] = setFusionLayers()
%% Set Layers
net_MD = load(fullfile('models','MDNet','mdnet_VOT2016_new.mat'));
net_MD.layers = net_MD.layers(1:9);
net_MD = vl_simplenn_tidy(net_MD) ;
net_MD = dagnn.DagNN.fromSimpleNN(net_MD, 'canonicalNames', true) ;

net_TCNN = load(fullfile('models','TCNN','imagenet-vgg-m_conv3_512-512-2.mat'));
net_TCNN.layers = net_TCNN.layers(1:9);
net_TCNN = vl_simplenn_tidy(net_TCNN) ;
net_TCNN = dagnn.DagNN.fromSimpleNN(net_TCNN, 'canonicalNames', true) ;

for x=1:numel(net_MD.layers)
  if isfield(net_MD.layers(x), 'name'), net_MD.layers(x).name = [net_MD.layers(x).name '_MDNet'] ;  end
end

for i = 1:numel(net_MD.vars)
    if~strcmp(net_MD.vars(i).name,'label')
        net_MD.renameVar(net_MD.vars(i).name, [net_MD.vars(i).name '_MDNet']); 
    end 
end 

for i = 1:numel(net_MD.params)
    net_MD.renameParam(net_MD.params(i).name, [net_MD.params(i).name '_MDNet']); 
end

for x=1:numel(net_TCNN.layers)
  if isfield(net_TCNN.layers(x), 'name'), net_TCNN.layers(x).name = [net_TCNN.layers(x).name '_TCNN'] ;  end
end

for i = 1:numel(net_TCNN.vars)
    if~strcmp(net_TCNN.vars(i).name,'label')
        net_TCNN.renameVar(net_TCNN.vars(i).name, [net_TCNN.vars(i).name '_TCNN']); 
    end 
end 

for i = 1:numel(net_TCNN.params)
    net_TCNN.renameParam(net_TCNN.params(i).name, [net_TCNN.params(i).name '_TCNN']); 
end

net_MD = dagnn.DagNN().loadobj(net_MD);
net_MD = net_MD.saveobj;

net_TCNN = dagnn.DagNN().loadobj(net_TCNN);
net_TCNN = net_TCNN.saveobj;

net.layers = [net_MD.layers net_TCNN.layers];
net.params = [net_MD.params net_TCNN.params];
net.vars = [net_MD.vars net_TCNN.vars];
net.meta = net_MD.meta;

net = dagnn.DagNN().loadobj(net);

% f=1/100 ;
% net.layers = {} ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,3,8, 'single'), zeros(1, 8, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'softmaxloss') ;
% 
% % Meta parameters
% % net.meta.inputSize = [28 28 1] ;
% net.meta.trainOpts.learningRate = 0.001 ;
% net.meta.trainOpts.numEpochs = 20 ;
% net.meta.trainOpts.batchSize = 1 ;
% % 
% % % Fill in defaul values
% net = vl_simplenn_tidy(net) ;
% net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% net.addLayer('classifier', dagnn.Conv('hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'x17_MDNet'}, {'classifier'},  {'conv8f'  'conv8b'});

block = dagnn.Concat() ;
net.addLayer('fusion', block, ...
             {'x9_MDNet', 'x9_TCNN'},...
             'fusion');   
%%
net.addLayer('relu', dagnn.ReLU(), {'fusion'}, 'relu', {}); %i think we dont need this, because at this point we are collecting everything from the two conv layers  
block_conv = dagnn.Conv('size', [3, 3, 1024, 512], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]);
net.addLayer('fusion_concat', block_conv, {'relu'},{'fusion_concat'},  {'conv8f'  'conv8b'});
net.addLayer('relu_fusion', dagnn.ReLU(), {'fusion_concat'}, 'relu_fusion', {});
% net.addLayer('dropout_fusion', dagnn.DropOut('rate', 0.5), {'relu_fusion'}, 'dropout_fusion', {}); % instead of this we use batch normalization

%%
block_conv = dagnn.Conv('size', [1, 1, 512, 512], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]);
net.addLayer('fusion_concat2', block_conv, {'relu_fusion'},{'fusion_concat2'},  {'conv9f'  'conv9b'});
net.addLayer('relu_fusion2', dagnn.ReLU(), {'fusion_concat2'}, 'relu_fusion2', {});
% net.addLayer('dropout_fusion2', dagnn.DropOut('rate', 0.5), {'relu_fusion2'}, 'dropout_fusion2', {}); %instead of this we use batch normalization

%%
block_conv = dagnn.Conv('size', [1, 1, 512, 2], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]); % this fully connected layer is probably not needed
net.addLayer('fusion_concat3', block_conv, {'relu_fusion2'},{'fusion_concat3'},  {'conv10f'  'conv10b'});

%%
net.addLayer('prob',dagnn.SoftMax(),'fusion_concat3','prob');
net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prob', 'label'}, {'objective'}, {});
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), {'prob', 'label'}, 'error') ;
addVar(net, 'input');

%%
net.vars(net.getVarIndex('input')).fanout = net.vars(net.getVarIndex('input')).fanout + 1 ;
net.layers(1).inputs = {'input'};
net.layers(1).inputIndexes = net.getVarIndex('input');
net.layers(10).inputs = {'input'};
net.layers(10).inputIndexes = net.getVarIndex('input');

f = 1/100;
init_bias = 0.1;
net.params(net.layers(net.getLayerIndex('fusion_concat')).paramIndexes(1)).value = f*randn(3, 3, 1024, 512, 'single');
net.params(net.layers(net.getLayerIndex('fusion_concat')).paramIndexes(2)).value = init_bias*ones(1,512,'single');%zeros(1, 512, 'single');

net.params(net.layers(net.getLayerIndex('fusion_concat2')).paramIndexes(1)).value = f*randn(1, 1, 512, 512, 'single');
net.params(net.layers(net.getLayerIndex('fusion_concat2')).paramIndexes(2)).value = init_bias*ones(1,512,'single'); %zeros(1, 512, 'single');

net.params(net.layers(net.getLayerIndex('fusion_concat3')).paramIndexes(1)).value = f*randn(1, 1, 512, 2, 'single');
net.params(net.layers(net.getLayerIndex('fusion_concat3')).paramIndexes(2)).value = zeros(1, 2, 'single');

net.params(net.layers(net.getLayerIndex('fusion_concat')).paramIndexes(1)).learningRate = 10;
net.params(net.layers(net.getLayerIndex('fusion_concat')).paramIndexes(1)).weightDecay = 1;
net.params(net.layers(net.getLayerIndex('fusion_concat')).paramIndexes(2)).learningRate = 20;
net.params(net.layers(net.getLayerIndex('fusion_concat')).paramIndexes(2)).weightDecay = 0;

net.params(net.layers(net.getLayerIndex('fusion_concat2')).paramIndexes(1)).learningRate = 10;
net.params(net.layers(net.getLayerIndex('fusion_concat2')).paramIndexes(1)).weightDecay = 1;
net.params(net.layers(net.getLayerIndex('fusion_concat2')).paramIndexes(2)).learningRate = 20;
net.params(net.layers(net.getLayerIndex('fusion_concat2')).paramIndexes(2)).weightDecay = 0;

net.params(net.layers(net.getLayerIndex('fusion_concat3')).paramIndexes(1)).learningRate = 10;
net.params(net.layers(net.getLayerIndex('fusion_concat3')).paramIndexes(1)).weightDecay = 1;
net.params(net.layers(net.getLayerIndex('fusion_concat3')).paramIndexes(2)).learningRate = 20;
net.params(net.layers(net.getLayerIndex('fusion_concat3')).paramIndexes(2)).weightDecay = 0;
% for l=1:length(net.layers)
%     if(isa(net.layers(l).block, 'dagnn.Conv'))
%         f_ind = net.layers(l).paramIndexes(1);
%         b_ind = net.layers(l).paramIndexes(2);
%         
%         fusion.params(f_ind).value = {f*randn(5,5,3,8, 'single')};
%        % fusion.params(f_ind).der = 0.01 * randn(1,1,1, 256,'single'); %randn(1,1,2, roidb_length/2,'single');
%        % fusion.params(f_ind).value = 0.01 * randn(1,1,2, roidb_length/2,'single');
% 
%         fusion.params(f_ind).learningRate = 1;
%         fusion.params(f_ind).weightDecay = 1;
% 
%         fusion.params(b_ind).value = zeros(1, 8, 'single');
%         %fusion.params(b_ind).der = zeros(1, 256, 'single');
%         fusion.params(b_ind).learningRate = 2;
%         fusion.params(b_ind).weightDecay = 0;
% 
%     end
% end

ind = 1;
for l=1:length(net.layers)-4
    % is a convolution layer?
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);

      if l<= length(net.layers)-10  %freeze the pararamiters
        net.params(f_ind).learningRate = 0;
        net.params(f_ind).weightDecay = 1;
        net.params(b_ind).learningRate = 0;
        net.params(b_ind).weightDecay = 1;

      else  %let paramiters to change in last layers
        net.params(f_ind).learningRate = 1;
        net.params(f_ind).weightDecay = 1;
        net.params(b_ind).learningRate = 1;
        net.params(b_ind).weightDecay = 1;   
      end
    end
end

for l=length(net.layers)-9:length(net.layers)
    % is a convolution layer?
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        
        net.params(f_ind).learningRate = 1;
        net.params(f_ind).weightDecay = 1;

        net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
        net.params(b_ind).learningRate = 0.5;
        net.params(b_ind).weightDecay = 1;

    end
end

net.meta.inputSize = [107 107 3] ;
% net.meta.trainOpts.learningRate = [0.05*ones(1,70) 0.005*ones(1,20) 0.0005*ones(1,10)];
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numEpochs = 100;


% net.vars(net.getVarIndex('stop_mdnet')).precious = 1;
% net.vars(net.getVarIndex('stop_tcnn')).precious = 1;

net.vars(net.getVarIndex('fusion')).precious = 1;
%net.vars(net.getVarIndex('fusion_concat')).precious = 1;
net.vars(net.getVarIndex('fusion_concat3')).precious = 1;
% net.vars(net.getVarIndex('prob')).precious = 1;
% fusion.vars(fusion.getVarIndex('error')).precious = 1;
% 
% fusion.rebuild() ;


net.conserveMemory = 1 ;

