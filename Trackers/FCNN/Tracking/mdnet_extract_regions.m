function ims = mdnet_extract_regions(im, boxes, opts)
% MDNET_EXTRACT_REGIONS
% Extract the bounding box regions from an input image. 
%
% extracted from MDnet Library - Hyeonseob Nam, 2015
% 

num_boxes = size(boxes, 1);

crop_mode = opts.crop_mode;
crop_size = opts.input_size;
crop_padding = opts.crop_padding;

ims = zeros(crop_size, crop_size, 3, num_boxes, 'single');
% mean_rgb = mean(mean(single(im)));

if(opts.useGpu > 0)
    parfor i = 1:num_boxes
    bbox = boxes(i,:);
    crop = im_crop(im, bbox, crop_mode, crop_size, crop_padding);
    ims(:,:,:,i) = crop;
    end
else
    for i = 1:num_boxes
    bbox = boxes(i,:);
    crop = im_crop(im, bbox, crop_mode, crop_size, crop_padding);
    ims(:,:,:,i) = crop;
    end
end
