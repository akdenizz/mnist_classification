
load mnist_net.mat

test_img = imread("test.png");
one_img = test_img(:,:,1);
imshow(one_img)

y = classify(net, one_img);
imshow(one_img)
title(y)