Readme

Project Name: 3D motion reconstruction using the incremental method
Class:        TUM Computer Vision SS23
Group:        11
Member:       Jincheng Pan, Wenjie Xie, Yao Ling, Kecheng Zhou, Shixin Li


1. Principal of Algorithm:

The software uses an incremental algorithm for multi-view 3D structure reconstruction. Introduce the point(track) concept in the reconstruction process. After traversing the images in the data set and performing one-to-one image matching, a set of feature points(track point sets) in the 3-dimensional structure displayed by the image set is obtained. The track point describes: a 3-dimensional space point in the corresponding 2-dimensional Image coordinates and images. Further select the order of reconstructed pictures to ensure the accuracy of the required camera external parameters (R,t) and the stability of the reconstructed object structure.


2. Pipline:

Input (images set, camera parameters) ->  Loop for selecting two pictures with the most feature matching points -> Calculate the track(Tracks) → calculate the connection graph G → select the edge e in the connection graph G → robustly estimate the intrinsic matrix E → decompose E to obtain the pose (R,t) → triangulate the points in the intersection of the track and the edge, initialize → for other edges (use PnP to get external parameters) → triangulate new edges → execute bundle adjustment -> Loop until all images are reconstructed.


3. Toolbox Functions:
BASE: Computer Vision Toolbox

(In Reconstruction):

  3.1. rgb2gray： Convert RGB images to grayscale images.

  3.2. detectSIFTFeatures: Use SIFT feature detector to extract feature points. 
    [Input: image, specified as an M-by-N matrix; Output: SIFTPoints object; Argument: Contrast threshold, Edge threshold,Number of layers in each octave]

  3.3. extractFeatures: extracted feature vectors, also known as descriptors, and their corresponding locations, from a binary or intensity image.

  3.4. returns indices of the matching features in the two input feature sets.
    [Input: two binaryFeatures objects, an M1-by-N matrix, or as one of the point feature objects; Output: Indices of corresponding features between the two input feature sets]

  3.5. addConnection, addView：Add views to view set and Add connection between views in view set.

  3.6. adaptConnection, adaptView：Update view in view set and Update connection between views in view set.

  3.7. findTracks: Find matched points across multiple views.
       [Input:Image view set, specified as an imageviewset object,View identifiers; Output:Point tracks across multiple views.
       Argument: Minimum length of the tracks, specified as a positive integer equal to or greater than 2]

  3.8. triangulateMultiview: Triangulation Algorithm to compute the 3-D locations of world points
       [Input: Matched points across multiple images, specified as an N-element array of pointTrack objects; Output: 3-D world points]

  3.9. estimateEssentialMatrix: Estimate essential matrix from corresponding points in a pair of images

  3.10. estworldpose: using PnP to estimate camera pose from 3-D to 2-D point correspondences

  3.11. bundleAdjustment: adjust collection of 3-D points and camera poses to optimize the quality of reconstructed 3D points.

(Point cloud generation and color matching):

  3.1. pointCloud: Object for storing 3-D point cloud
       [Input: xyz-Points Location; Output: returned as a pointCloud object; Argument:'Color'(Point cloud color), specified as an RGB value]

  3.2. pcshow: Plot 3-D point cloud

  3.3. pcdenoise: Remove noise from 3-D point cloud


4. GUI:

  4.1. Select Picture
  
  4.2. Choose camera.txt file, user defined value is also allowed.

  4.3. For some monotonous scene reconstructions (images looked almost same): choose small scene so that you can use another function to estimate the essential matrix. But for most outdoor big scenes, we recommed you to choose big scene.

  4.4. Wait bar: this will show you the progress of processing. It will take slightly a bit time to finish the process. What we want to guarantee is that the user can spent less time selecting and arranging pictures. The program will filter the image set uploaded and find the most proper arrange to reconstruct.
  
  4.5. Results: in graph area user can find the reconstruction of sparse point cloud, dense point cloud, dense point cloud with rendering, clustering and bounding box. 

  4.6. Distance: in clustering graph each box is numbered. if user want to check the size of box or distance between two boxes, type in the number of boxes that you want to check.

5. Parameter:
  In this section, we will offer you some parameters in main function, if there shows some error, it maybe means that some parameters(thresholds) are't correctly set. you can try to modify these parameters to get better results. (Ctrl+F input value below)
  
  5.1. Connecting graph:
     numel(pairsIdx) >= Value:     to make sure most edges threshold

  5.2. Dense reconstruction:
     error:                    filter value for noisy points.(recommend: 15-1000)
     reprojectionErrors:       find good points for dense reconstruction, try it out(recommend:5)

  5.3. Clustering:
     minDistance:              smallest distance between two different clusters.(recommend: 0.45 for delivery_area, 2 for post_hall)
     minPoints:                least points in each cluster.(recommend: 100)
     maxPoints:                most points in each cluster.(recommend: 100000)

  5.4. helperEstimateRalativePose.m:
     MaxNumTrials:             (recommend: 1000)
     Confidence:               (recommend: 60)
     MaxDistance:              (recommend: 1000)

  
