
process = preprocess();

process.unzipgz;

pyrunfile("extract_mnist.py");

N_train = 60000; % number of data in train
N_test = 10000; % number of data in test

train_labels = process.load_labels("train");
test_labels = process.load_labels("test");

[XTrain, YTrain] = process.create_inputs(N_train, "train", train_labels);
[XTest, YTest] = process.create_inputs(N_test, "test", test_labels);

YTrain = categorical(YTrain);
YTest = categorical(YTest);

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
imageSize = [28 28 1];
augimds = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);

layers=[imageInputLayer([28 28 1],'Name','input')
        convolution2dLayer(3,6,'Padding','same')
        reluLayer
        batchNormalizationLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,16)
        reluLayer
        batchNormalizationLayer
        maxPooling2dLayer(2,'Stride',2)
        fullyConnectedLayer(120,'name','f1')
        reluLayer
        fullyConnectedLayer(84,'name','f2')
        reluLayer
        fullyConnectedLayer(10,'name','f3')
        softmaxLayer
        classificationLayer];

options = trainingOptions('adam','MaxEpochs',25,"Plots","training-progress",'LearnRateSchedule' ,'piecewise','LearnRateDropPeriod',15,'LearnRateDropFactor' ,0.1);

net = trainNetwork(augimds,layers,options);

save mnist_net.mat net

YPred = classify(net,XTest);
accuracy = sum(YTest==YPred)/numel(YTest);

disp("Accuracy of the training net is: " + accuracy)