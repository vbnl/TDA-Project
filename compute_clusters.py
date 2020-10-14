def compute_clusters(diagrams):
    num_dgms = len(diagrams)
    distance_matrix = np.empty([num_dgms,num_dgms])
    for i in range(num_dgms):
        for j in range(num_dgms):
            if i == j:
                distance_matrix[i][i] = 0
            else:
                distance_matrix[i][j] = persim.bottleneck(diagrams[i],diagrams[j])
    embedding = sklearn.manifold.MDS(n_components = 2, random_state = 0, dissimilarity = 'precomputed')        
    coords = embedding.fit_transform(distance_matrix)
    x = coords[:,0]
    y = coords[:,1]
    album = np.concatenate(np.full((first_album_length,1), 0), np.full((second_album_length,1), 1). np.full((third_album_length,1), 2))
    kmeans = sklearn.cluster.KMeans(n_clusters = 3).fit(coords)
    cluster_labels = kmeans.labels_
    print("I should have "+str(num_dgms)+"labels and I really have "+str(len(cluster_labels)))
    #plt.scatter(x,y,c=cluster_labels)
    
    # This is not a particularly elegant solution, but the simplest way I could get it to plot both different colors and different markers
    x_first_album = x[:first_album_length]
    y_first_album = y[:first_album_length]
    labels_first_album = cluster_labels[:first_album_length]
    
    x_second_album = x[first_album_length:first_album_length+second_album_length]
    y_second_album = y[first_album_length:first_album_length+second_album_length]
    labels_second_album = cluster_labels[first_album_length:first_album_length+second_album_length]
    x_third_album = x[first_album_length + second_album_length:first_album_length+second_album_length + third_album_length]
    y_third_album = y[first_album_length + second_album_length:first_album_length+second_album_length + third_album_length]
    labels_third_album = cluster_labels[first_album_length + second_album_length:first_album_length+second_album_length + third_album_length]
    
    plt.scatter(x_first_album, y_first_album, c = labels_first_album, marker = 'o')
    plt.scatter(x_second_album, y_second_album, c = labels_second_album, marker = 'x')
    plt.scatter(x_third_album, y_third_album, c = labels_third_album, marker = '+')
