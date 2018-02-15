seqName = 'Deer'
conf = genConfig('otb',seqName);

result_fusion = run_fusion(conf.imgList, conf.gt(1,:));

v = VideoWriter([seqName '.avi']);
filename = char(conf.imgList(1));
I = imread(filename);
frame = im2frame(I);

open(v);
writeVideo(v,frame);
fid = fopen( [seqName '.txt'], 'wt' );

for test = 1:length(result_fusion)
  [a1] = result_fusion(test,:);
  fprintf( fid, '%f,%f,%f,%f\n', a1(1), a1(2), a1(3), a1(4));
end
fclose(fid);

for test = 2:length(result_fusion)
    filename = char(conf.imgList(test));
    I = imread(filename);
    
    RGB = insertShape(I ,'rectangle', result_fusion(test,:),  'Color', 'red' ,'LineWidth',3);
    RGB = insertText(RGB, [10,10], test,'TextColor','black', 'FontSize', 30);
    
    frame = im2frame(RGB);
    
    writeVideo(v,frame);
end
close(v);