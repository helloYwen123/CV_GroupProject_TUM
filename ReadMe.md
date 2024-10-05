# 3D Motion Reconstruction Using the Incremental Method

## Class Information

Class: TUM Computer Vision SS23

Group: 11

Members: Jincheng Pan, Wenjie Xie, Yao Ling, Kecheng Zhou, Shixin Li

## Principal of Algorithm

This software uses an incremental algorithm for multi-view 3D structure reconstruction. The point (track) concept is introduced in the reconstruction process. After traversing the images in the dataset and performing one-to-one image matching, a set of feature points (track point sets) in the 3-dimensional structure displayed by the image set is obtained. The track point describes a 3-dimensional space point in the corresponding 2-dimensional image coordinates and images. Further, the order of reconstructed pictures is selected to ensure the accuracy of the required camera external parameters (R, t) and the stability of the reconstructed object structure.

## Pipeline

Input: (images set, camera parameters)

Process:

Loop for selecting two pictures with the most feature matching points.

Calculate the track (Tracks).

Calculate the connection graph G.

Select the edge e in the connection graph G.

Robustly estimate the intrinsic matrix E.

Decompose E to obtain the pose (R, t).

Triangulate the points in the intersection of the track and the edge, initialize.

For other edges (use PnP to get external parameters).

Triangulate new edges.

Execute bundle adjustment.

Loop until all images are reconstructed.

## Toolbox Functions

In Reconstruction

rgb2gray: Convert RGB images to grayscale images.

detectSIFTFeatures: Use SIFT feature detector to extract feature points.

Input: image, specified as an M-by-N matrix.

Output: SIFTPoints object.

Arguments: Contrast threshold, Edge threshold, Number of layers in each octave.

extractFeatures: Extracted feature vectors, also known as descriptors, and their corresponding locations from a binary or intensity image.

matchFeatures: Returns indices of the matching features in the two input feature sets.

Input: two binaryFeatures objects, an M1-by-N matrix, or as one of the point feature objects.

Output: Indices of corresponding features between the two input feature sets.

addConnection, addView: Add views to view set and add connection between views in view set.

adaptConnection, adaptView: Update view in view set and update connection between views in view set.

findTracks: Find matched points across multiple views.

Input: Image view set, specified as an imageviewset object, View identifiers.

Output: Point tracks across multiple views.

Argument: Minimum length of the tracks, specified as a positive integer equal to or greater than 2.

triangulateMultiview: Triangulation algorithm to compute the 3-D locations of world points.

Input: Matched points across multiple images, specified as an N-element array of pointTrack objects.

Output: 3-D world points.

estimateEssentialMatrix: Estimate essential matrix from corresponding points in a pair of images.

estworldpose: Use PnP to estimate camera pose from 3-D to 2-D point correspondences.

bundleAdjustment: Adjust collection of 3-D points and camera poses to optimize the quality of reconstructed 3D points.

Point Cloud Generation and Color Matching

pointCloud: Object for storing 3-D point cloud.

Input: xyz-Points Location.

Output: Returned as a pointCloud object.

Argument: 'Color' (Point cloud color), specified as an RGB value.

pcshow: Plot 3-D point cloud.

pcdenoise: Remove noise from 3-D point cloud.

## GUI

Select Picture: Choose the picture to process.

Choose camera.txt file: User-defined value is also allowed.

Scene Type: For some monotonous scene reconstructions (images look almost the same), choose "small scene" to use a different function for estimating the essential matrix. For most outdoor big scenes, we recommend choosing "big scene."

Wait Bar: Displays the progress of processing. It will take some time to finish the process. The goal is to minimize the time users spend selecting and arranging pictures. The program will filter the uploaded image set and find the best arrangement for reconstruction.

Results: In the graph area, users can find the reconstruction of sparse point cloud, dense point cloud, dense point cloud with rendering, clustering, and bounding box.

Distance Measurement: In the clustering graph, each box is numbered. To check the size of a box or distance between two boxes, type in the number of the boxes you want to check.

## Parameters

In this section, we offer some parameters in the main function. If errors occur, it might mean that some parameters (thresholds) are incorrectly set. You can modify these parameters to achieve better results (use Ctrl+F to find the value below).

### Connecting Graph

numel(pairsIdx) >= Value: Ensure most edges threshold.

### Dense Reconstruction

error: Filter value for noisy points (recommended: 15-1000).

reprojectionErrors: Find good points for dense reconstruction (recommended: 5).

### Clustering

minDistance: Smallest distance between two different clusters (recommended: 0.45 for delivery_area, 2 for post_hall).

minPoints: Least points in each cluster (recommended: 100).

maxPoints: Most points in each cluster (recommended: 100000).

### helperEstimateRelativePose.m

MaxNumTrials: (recommended: 1000).

Confidence: (recommended: 60).

MaxDistance: (recommended: 1000).
