// PCL lib Functions for processing point clouds

#ifndef PROCESSPOINTCLOUDS_H_
#define PROCESSPOINTCLOUDS_H_

#include "kdtree.h"
#include "render/box.h"
#include <boost/filesystem.hpp>
#include <chrono>
#include <ctime>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <string>
#include <unordered_set>
#include <vector>

template <typename PointT> class ProcessPointClouds {
public:
  // constructor
  ProcessPointClouds();
  // deconstructor
  ~ProcessPointClouds();

  void numPoints(typename pcl::PointCloud<PointT>::Ptr cloud);

  typename pcl::PointCloud<PointT>::Ptr
  FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes,
              Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint);

  std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>
  SeparateClouds(pcl::PointIndices::Ptr inliers,
                 typename pcl::PointCloud<PointT>::Ptr cloud);

  std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>
  SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations,
               float distanceThreshold);

  std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>
  SegmentPlane2(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations,
                float distanceThreshold);

  std::vector<typename pcl::PointCloud<PointT>::Ptr>
  Clustering(typename pcl::PointCloud<PointT>::Ptr cloud,
             float clusterTolerance, int minSize, int maxSize);
  std::vector<typename pcl::PointCloud<PointT>::Ptr>
  Clustering2(typename pcl::PointCloud<PointT>::Ptr cloud,
              float clusterTolerance, int minSize, int maxSize);

  Box BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster);

  void savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file);

  typename pcl::PointCloud<PointT>::Ptr loadPcd(std::string file);

  std::vector<boost::filesystem::path> streamPcd(std::string dataPath);
};

// constructor:
template <typename PointT> ProcessPointClouds<PointT>::ProcessPointClouds() {}

// de-constructor:
template <typename PointT> ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template <typename PointT>
void ProcessPointClouds<PointT>::numPoints(
    typename pcl::PointCloud<PointT>::Ptr cloud) {
  std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(
    typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes,
    Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint) {

  // Time segmentation process
  auto startTime = std::chrono::steady_clock::now();

  typename pcl::PointCloud<PointT>::Ptr cloud_cropped(
      new pcl::PointCloud<PointT>);
  typename pcl::PointCloud<PointT>::Ptr cloud_filtered(
      new pcl::PointCloud<PointT>);
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(filterRes, filterRes, filterRes);
  sor.filter(*cloud_filtered);

  pcl::CropBox<PointT> cropBox(true);
  cropBox.setMin(minPoint);
  cropBox.setMax(maxPoint);
  cropBox.setInputCloud(cloud_filtered);
  cropBox.filter(*cloud_cropped);

  std::vector<int> roof_indices;
  pcl::CropBox<PointT> roof(true);
  cropBox.setMin(Eigen::Vector4f(-1.5f, -1.7f, -1.0, 1));
  cropBox.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1));
  cropBox.setInputCloud(cloud_cropped);
  cropBox.filter(roof_indices);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  for (int i : roof_indices) {
    inliers->indices.push_back(i);
  }

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(cloud_cropped);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cloud_cropped);

  auto endTime = std::chrono::steady_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  std::cout << "filtering took " << elapsedTime.count() << " milliseconds"
            << std::endl;

  return cloud_cropped;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
          typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SeparateClouds(
    pcl::PointIndices::Ptr inliers,
    typename pcl::PointCloud<PointT>::Ptr cloud) {
  // TODO: Create two new point clouds, one cloud with obstacles and other with
  // segmented plane Extract the inliers

  typename pcl::PointCloud<PointT>::Ptr plane(new pcl::PointCloud<PointT>);
  typename pcl::PointCloud<PointT>::Ptr obstacles(new pcl::PointCloud<PointT>);

  pcl::ExtractIndices<PointT> extract = pcl::ExtractIndices<PointT>();
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*plane);

  extract.setNegative(true);
  extract.filter(*obstacles);

  std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>
      segResult(obstacles, plane);
  return segResult;
}

// Ransac algorithm as presented in the course.
template <typename PointT>
std::unordered_set<int>
RansacSegmentation(typename pcl::PointCloud<PointT>::Ptr cloud,
                   int maxIterations, float distanceTol) {
  std::unordered_set<int> inliersResult;
  auto nrPoints = cloud->size();
  srand(time(NULL));

  std::unordered_set<int> inliersAcc;
  for (int i = 0; i < maxIterations; i++) {
    inliersAcc.clear();
    while (inliersAcc.size() < 3) {
      inliersAcc.insert(rand() % nrPoints);
    }
    auto itr = inliersAcc.begin();
    int ix1 = *itr;
    itr++;
    int ix2 = *(itr);
    itr++;
    int ix3 = *(itr);
    PointT p1 = cloud->points[ix1];
    PointT p2 = cloud->points[ix2];
    PointT p3 = cloud->points[ix3];
    float A = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
    float B = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
    float C = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    float D = -1 * (A * p1.x + B * p1.y + C * p1.z);
    float r = sqrt(pow(A, 2) + pow(B, 2) + pow(C, 2));

    for (int i = 0; i < nrPoints; i++) {
      auto p = cloud->points[i];
      float pX = p.x;
      float pY = p.y;
      float pZ = p.z;
      float d = fabs(A * pX + B * pY + C * pZ + D) / r;
      if (d < distanceTol) {
        inliersAcc.insert(i);
      }
    }

    if (inliersAcc.size() > inliersResult.size()) {
      inliersResult = inliersAcc;
    }
  }
  return inliersResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
          typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SegmentPlane2(
    typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations,
    float distanceThreshold) {
  // Time segmentation process
  auto startTime = std::chrono::steady_clock::now();

  std::unordered_set<int> ixs =
      RansacSegmentation<PointT>(cloud, maxIterations, distanceThreshold);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  for (auto ix : ixs) {
    inliers->indices.push_back(ix);
  }

  if (inliers->indices.size() == 0) {
    std::cout << "Could not estimate a planar model for the given dataset."
              << std::endl;
  }

  auto endTime = std::chrono::steady_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  std::cout << "plane segmentation took " << elapsedTime.count()
            << " milliseconds" << std::endl;

  std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>
      segResult = SeparateClouds(inliers, cloud);
  return segResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
          typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SegmentPlane(
    typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations,
    float distanceThreshold) {
  // Time segmentation process
  auto startTime = std::chrono::steady_clock::now();

  pcl::SACSegmentation<PointT> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(maxIterations);
  seg.setDistanceThreshold(distanceThreshold);
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.size() == 0) {
    std::cout << "Could not estimate a planar model for the given dataset."
              << std::endl;
  }

  auto endTime = std::chrono::steady_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  std::cout << "plane segmentation took " << elapsedTime.count()
            << " milliseconds" << std::endl;

  std::pair<typename pcl::PointCloud<PointT>::Ptr,
            typename pcl::PointCloud<PointT>::Ptr>
      segResult = SeparateClouds(inliers, cloud);
  return segResult;
}

void clusterHelper(std::vector<int> &cluster,
                   std::unordered_set<int> &processed,
                   const std::vector<std::vector<float>> &points,
                   const KdTree *tree, const int pointId,
                   const float distanceTol) {
  processed.insert(pointId);
  cluster.push_back(pointId);
  auto neighbors = tree->search(points[pointId], distanceTol);
  for (int neighbor : neighbors) {
    if (processed.find(neighbor) == processed.end())
      clusterHelper(cluster, processed, points, tree, neighbor, distanceTol);
  }
}

std::vector<std::vector<int>>
euclideanCluster(const std::vector<std::vector<float>> &points,
                 const KdTree *tree, const float distanceTol) {
  std::vector<std::vector<int>> clusters;
  std::unordered_set<int> processed;

  for (int i = 0; i < points.size(); i++) {
    if (processed.find(i) == processed.end()) {
      std::vector<int> cluster;
      clusterHelper(cluster, processed, points, tree, i, distanceTol);
      clusters.push_back(cluster);
    }
  }

  return clusters;
}

// Clustering based on kd trees. The algorithm follows closely the solution
// presented in the course.
template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::Clustering2(
    typename pcl::PointCloud<PointT>::Ptr cloud, const float clusterTolerance,
    const int minSize, const int maxSize) {

  // Time clustering process
  auto startTime = std::chrono::steady_clock::now();

  std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

  // build point vector and kd tree from cloud
  KdTree *tree(new KdTree);
  std::vector<std::vector<float>> points;
  for (int i = 0; i < cloud->size(); i++) {
    auto point = cloud->points[i];
    std::vector<float> v = {point.x, point.y, point.z};
    tree->insert(v, i);
    points.push_back(v);
  }

  std::vector<std::vector<int>> cluster_indices =
      euclideanCluster(points, tree, clusterTolerance);

  for (const auto &cluster : cluster_indices) {
    typename pcl::PointCloud<PointT>::Ptr cloud_cluster(
        new pcl::PointCloud<PointT>);
    for (const auto &idx : cluster) {
      cloud_cluster->push_back((*cloud)[idx]);
    } //*
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    clusters.push_back(cloud_cluster);
  }

  auto endTime = std::chrono::steady_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  std::cout << "clustering took " << elapsedTime.count()
            << " milliseconds and found " << clusters.size() << " clusters"
            << std::endl;

  return clusters;
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::Clustering(
    typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance,
    int minSize, int maxSize) {

  // Time clustering process
  auto startTime = std::chrono::steady_clock::now();

  std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

  typename pcl::search::KdTree<PointT>::Ptr tree(
      new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(clusterTolerance); // 2cm
  ec.setMinClusterSize(minSize);
  ec.setMaxClusterSize(maxSize);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  for (const auto &cluster : cluster_indices) {
    typename pcl::PointCloud<PointT>::Ptr cloud_cluster(
        new pcl::PointCloud<PointT>);
    for (const auto &idx : cluster.indices) {
      cloud_cluster->push_back((*cloud)[idx]);
    } //*
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    clusters.push_back(cloud_cluster);
  }

  auto endTime = std::chrono::steady_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  std::cout << "clustering took " << elapsedTime.count()
            << " milliseconds and found " << clusters.size() << " clusters"
            << std::endl;

  return clusters;
}

template <typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(
    typename pcl::PointCloud<PointT>::Ptr cluster) {

  // Find bounding box for one of the clusters
  PointT minPoint, maxPoint;
  pcl::getMinMax3D(*cluster, minPoint, maxPoint);

  Box box;
  box.x_min = minPoint.x;
  box.y_min = minPoint.y;
  box.z_min = minPoint.z;
  box.x_max = maxPoint.x;
  box.y_max = maxPoint.y;
  box.z_max = maxPoint.z;

  return box;
}

template <typename PointT>
void ProcessPointClouds<PointT>::savePcd(
    typename pcl::PointCloud<PointT>::Ptr cloud, std::string file) {
  pcl::io::savePCDFileASCII(file, *cloud);
  std::cerr << "Saved " << cloud->points.size() << " data points to " + file
            << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
ProcessPointClouds<PointT>::loadPcd(std::string file) {

  typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

  if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) //* load the file
  {
    PCL_ERROR("Couldn't read file \n");
  }
  std::cerr << "Loaded " << cloud->points.size() << " data points from " + file
            << std::endl;

  return cloud;
}

template <typename PointT>
std::vector<boost::filesystem::path>
ProcessPointClouds<PointT>::streamPcd(std::string dataPath) {

  std::vector<boost::filesystem::path> paths(
      boost::filesystem::directory_iterator{dataPath},
      boost::filesystem::directory_iterator{});

  // sort files in accending order so playback is chronological
  sort(paths.begin(), paths.end());

  return paths;
}

#endif /* PROCESSPOINTCLOUDS_H_ */
