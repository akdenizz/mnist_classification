classdef preprocess
    methods(Static)
        function unzipgz
            gunzip dataset_raw\train-images-idx3-ubyte.gz dataset\
            gunzip dataset_raw\train-labels-idx1-ubyte.gz dataset\
            gunzip dataset_raw\t10k-images-idx3-ubyte.gz dataset\
            gunzip dataset_raw\t10k-labels-idx1-ubyte.gz dataset\
        end

        function labels = load_labels(name)
            filename = "dataset\" + name + "_data\labels\" + name + "_labels.txt";
            f = fopen(filename);
            data = textscan(f,'%s');
            fclose(f);
            labels = str2double(data{1}(1:1:end));
        end

        function [X, Y] = create_inputs(N_sample, name, labels)
            X = zeros(28,28,1,N_sample);
            Y = zeros(N_sample,1);
            for k=1:N_sample
                images_file = "dataset\" + name + "_data\images\" + num2str(k) + ".png";
                image=imread(images_file);  
                X(:,:,1,k)=image;
                Y(k)=labels(k);
            end
        end
    end
end
