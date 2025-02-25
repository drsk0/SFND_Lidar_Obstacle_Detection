/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

#include "processPointClouds.h"
#include "render/render.h"
#include "sensors/lidar.h"
#include <pcl/impl/point_types.hpp>

std::vector<Car> initHighway(bool renderScene,
                             pcl::visualization::PCLVisualizer::Ptr &viewer) {

  Car egoCar(Vect3(0, 0, 0), Vect3(4, 2, 2), Color(0, 1, 0), "egoCar");
  Car car1(Vect3(15, 0, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car1");
  Car car2(Vect3(8, -4, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car2");
  Car car3(Vect3(-12, 4, 0), Vect3(4, 2, 2), Color(0, 0, 1), "car3");

  std::vector<Car> cars;
  cars.push_back(egoCar);
  cars.push_back(car1);
  cars.push_back(car2);
  cars.push_back(car3);

  if (renderScene) {
    renderHighway(viewer);
    egoCar.render(viewer);
    car1.render(viewer);
    car2.render(viewer);
    car3.render(viewer);
  }

  return cars;
}

void simpleHighway(pcl::visualization::PCLVisualizer::Ptr &viewer) {
  // ----------------------------------------------------
  // -----Open 3D viewer and display simple highway -----
  // ----------------------------------------------------

  // RENDER OPTIONS
  // bool renderScene = true;
  bool renderScene = false;
  std::vector<Car> cars = initHighway(renderScene, viewer);

  Lidar *lidar = new Lidar(cars, 0.0);

  ProcessPointClouds<pcl::PointXYZ> processor =
      ProcessPointClouds<pcl::PointXYZ>();

  pcl::PointCloud<pcl::PointXYZ>::Ptr pc = lidar->scan();

  std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,
            pcl::PointCloud<pcl::PointXYZ>::Ptr>
      segmentCloud = processor.SegmentPlane(pc, 100, 0.2);
  // renderPointCloud(viewer,segmentCloud.first,"obstCloud",Color(1,0,0));
  // renderPointCloud(viewer,segmentCloud.second,"planeCloud",Color(0,1,0));

  // renderPointCloud(viewer, pc, "pc");
  // renderRays(viewer, lidar->position, pc);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters =
      processor.Clustering(segmentCloud.first, 1.0, 3, 30);

  int clusterId = 0;
  std::vector<Color> colors = {Color(1, 0, 0), Color(0, 1, 0), Color(0, 0, 1)};

  for (pcl::PointCloud<pcl::PointXYZ>::Ptr cluster : cloudClusters) {
    std::cout << "cluster size ";
    processor.numPoints(cluster);
    renderPointCloud(viewer, cluster, "obstCloud" + std::to_string(clusterId),
                     colors[clusterId]);

    Box box = processor.BoundingBox(cluster);
    renderBox(viewer, box, clusterId);

    ++clusterId;
  }
}

// setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle,
                pcl::visualization::PCLVisualizer::Ptr &viewer) {

  viewer->setBackgroundColor(0, 0, 0);

  // set camera position and angle
  viewer->initCameraParameters();
  // distance away in meters
  int distance = 16;

  switch (setAngle) {
  case XY:
    viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0);
    break;
  case TopDown:
    viewer->setCameraPosition(0, 0, distance, 1, 0, 1);
    break;
  case Side:
    viewer->setCameraPosition(0, -distance, 0, 0, 0, 1);
    break;
  case FPS:
    viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
  }

  if (setAngle != FPS)
    viewer->addCoordinateSystem(1.0);
}

void cityBlock(pcl::visualization::PCLVisualizer::Ptr &viewer,
               ProcessPointClouds<pcl::PointXYZI> *pointProcessorI,
               const pcl::PointCloud<pcl::PointXYZI>::Ptr &inputCloud) {
  // ----------------------------------------------------
  // -----Open 3D viewer and display City Block     -----
  // ----------------------------------------------------

  // std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr,
  // pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentCloud =
  // pointProcessorI->SegmentPlane2(inputCloud, 100, 0.1);
  // renderPointCloud(viewer, segmentCloud.second, "plane", Color(0,1,0));
  // pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud =
  // processor->loadPcd("../src/sensors/data/pcd/data_1/0000000000.pcd");
  pcl::PointCloud<pcl::PointXYZI>::Ptr filterCloud =
      pointProcessorI->FilterCloud(inputCloud, 0.2f,
                                   Eigen::Vector4f(-20.0f, -6.0f, -3.0f, 1.0f),
                                   Eigen::Vector4f(30.0f, 7.0f, 5.0f, 1.0f));
  std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr,
            pcl::PointCloud<pcl::PointXYZI>::Ptr>
      segmentCloud = pointProcessorI->SegmentPlane2(filterCloud, 50, 0.2);
  renderPointCloud(viewer, segmentCloud.second, "plane", Color(0, 1, 0));
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters =
      pointProcessorI->Clustering2(segmentCloud.first, 0.42f, 18, 1500);

  // rendering
  int clusterId = 0;
  std::vector<Color> colors = {Color(1, 0, 0), Color(1, 1, 0), Color(0, 0, 1),
                               Color(1, 1, 0), Color(1, 0, 1), Color(0, 1, 1)};
  for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : cloudClusters) {
    std::cout << "cluster size ";
    pointProcessorI->numPoints(cluster);
    renderPointCloud(viewer, cluster, "obstCloud" + std::to_string(clusterId),
                     colors[clusterId % 6]);
    Box box = pointProcessorI->BoundingBox(cluster);
    renderBox(viewer, box, clusterId);
    ++clusterId;
  }
  // renderPointCloud(viewer,filterCloud,"inputCloud");
}

int main(int argc, char **argv) {
  std::cout << "starting enviroment" << std::endl;

  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("3D Viewer"));
  CameraAngle setAngle = XY;
  initCamera(setAngle, viewer);

  ProcessPointClouds<pcl::PointXYZI> *pointProcessorI =
      new ProcessPointClouds<pcl::PointXYZI>();
  std::vector<boost::filesystem::path> stream =
      pointProcessorI->streamPcd("../src/sensors/data/pcd/data_1");
  auto streamIterator = stream.begin();
  pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI;
  // simpleHighway(viewer);

  // cityBlock(viewer, pointProcessorI, inputCloudI);

  while (!viewer->wasStopped()) {

    // Clear viewer
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    // Load pcd and run obstacle detection process
    inputCloudI = pointProcessorI->loadPcd((*streamIterator).string());
    cityBlock(viewer, pointProcessorI, inputCloudI);

    streamIterator++;
    if (streamIterator == stream.end())
      streamIterator = stream.begin();

    viewer->spinOnce();
  }
}
