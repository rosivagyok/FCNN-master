clear mex
run matconvnet\matlab\vl_setupnn ;
cd matconvnet;
vl_compilenn('verbose', 1);
cd ..\;

conf = genConfig('VOT2016','bag');
s_frames = conf.imgList;
seq.init_rect = conf.gt(1,:);
results.res = TCNNtrack(s_frames, seq.init_rect);