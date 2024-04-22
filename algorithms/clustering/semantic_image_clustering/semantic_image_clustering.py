#Semantic Clustering by Adopting Nearest neighbors (SCAN) algorithm.

#This example demonstrates how to apply the Semantic Clustering by Adopting Nearest neighbors (SCAN) algorithm (Van Gansbeke et al., 2020) 
#on the CIFAR-10 dataset. The algorithm consists of two phases:
#Self-supervised visual representation learning of images, in which we use the simCLR technique.
#Clustering of the learned visual representation vectors to maximize the agreement between the cluster assignments of neighboring vectors.
#https://keras.io/examples/vision/semantic_image_clustering/