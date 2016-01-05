houses1 = [102 110 113 115 116 118 120];
houses2 = [102 110 113 115 116 118 120];
for d = 1:length(houses1)
    for u = 1:length(houses2)
        if(houses1(d) ~= houses2(u))
            current_input_directory = sprintf('datasets/hh%d/hh%d_hh%d/', houses1(1,d),houses1(1,d),houses2(1,u));
            fileNameSource = sprintf('%sStrat_source.csv',current_input_directory);
            fileNameTarget = sprintf('%sStrat_target.csv',current_input_directory);      


            X1 = load(fileNameSource);
            X2 = load(fileNameTarget);

            % Run K-Means algorithm on this data
            K = 16; 
            max_iters = 10;

            % When using K-Means, it is important the initialize the centroids
            % randomly. 
            initial_centroids = kMeansInitCentroids(X1, K);

            % Run K-Means
            [centroids, idx] = runkMeans(X1, initial_centroids, max_iters);

            % Find closest cluster members
            idx = findClosestCentroids(X1, centroids);
            K_Means_matrix = zeros(length(idx),length(idx));
            for i = 1:size(idx,1)
                K_Means_matrix(i,:) = (idx == idx(i,1))';
            end
            fileNameKmeans = sprintf('%sKMeansMatrix.csv',current_input_directory);
            csvwrite(fileNameKmeans,K_Means_matrix);
            
            X_recovered = centroids(idx,:);
            
            fileNameKmeans_output = sprintf('%sKMeansClassified.csv',current_input_directory);
            csvwrite(fileNameKmeans_output,X_recovered);
        end
    end
end