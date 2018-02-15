function [ roidb ] = setup_data(seqList, opts)
% -------------------------------------------------------------------------
roidb = {};    
roidb_ = cell(1,length(seqList));

for i = 1:length(seqList)
    seq = seqList{i};
    fprintf('sampling %s:%s ...\n', 'VOT2016', seq);

    config = genConfig('VOT2016', seq);
    roidb_{i} = seq2roidb(config, opts);

end
roidb = [roidb,roidb_];