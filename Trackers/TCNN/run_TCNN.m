function results=run_TCNN(seq,  res_path, bSaveImage)

close all

s_frames = seq.s_frames;

% clear mex
% run matconvnet\matlab\vl_setupnn ;
% cd matconvnet;
% vl_compilenn('verbose', 1);
% cd ..\;
  
results.type = 'rect';
results.res = TCNNtrack(s_frames, seq.init_rect);