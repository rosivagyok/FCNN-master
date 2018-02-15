%% Setup FCNN
% Runs MatConvnet to be possible to Train and Track

clear mex
run matconvnet/matlab/vl_setupnn ;
cd matconvnet;

vl_compilenn('verbose', 1);
cd ../;