function [results] = run_FCNN(seq, res_path, bSaveImage)
%% Run FCNN
%  This function is used by the benchmark to get the results and plot the 
%  sucess and precision plots
s_frames = seq.s_frames;

results.type = 'rect';
results.res = run_fusion(s_frames, seq.init_rect);
end

