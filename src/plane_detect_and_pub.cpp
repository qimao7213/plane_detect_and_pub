#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unistd.h>
#include <omp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h> 
#include <pcl/surface/concave_hull.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "plane_fitter_pcl.hpp"
#include "AHCPlaneSeg.hpp"
#include "lshaped_fitting.h"
const cv::Size imgSize(640, 480);
const int width = 640;
const int height = 480;
typedef pcl::PointCloud<pcl::PointXYZRGB> myPointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZ> myPointCloud;
//--------TUM3---------------
// const float cx = 320.1;
// const float cy = 247.6;
// const float fx = 535.4;
// const float fy = 539.2;
// const float depthScale = 5000.0;
//--------Bonn---------------
// //-----------
// const float cx = 315.59;
// const float cy = 237.76;
// const float fx = 542.82;
// const float fy = 542.58; 
// const float depthScale = 5000.0;

//--------i515-------------
const float cx = 325.44140625;
const float cy = 236.3984375;
const float fx = 460.2265625;
const float fy = 460.2265625;
const float depthScale = 1000.0;

//--------D455-------------
// const float cx = 328.57140625;
// const float cy = 240.3284375;
// const float fx = 390.2265625;
// const float fy = 390.2265625;
// const float depthScale = 1000.0;

void replaceZeroDepth_and_generatePointCloud(cv::Mat& depthImage, const myPointCloud::Ptr Cloud) 
{
    int radius = 2;  // 5x5 neighborhood, so radius is 2
    cv::Mat tempImage;
    depthImage.copyTo(tempImage);
    // omp_set_num_threads(8);
    // #pragma omp parallel for 
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            if( i >= radius && i < height - radius && j >= radius && j < width - radius)
            {
                if (depthImage.at<uint16_t>(i, j) == 0) 
                {
                    int sum = 0;
                    int count = 0;
                    for (int x = -radius; x <= radius; ++x) {
                        for (int y = -radius; y <= radius; ++y) {
                            uint16_t depth = tempImage.at<uint16_t>(i + x, j + y);
                            if (depth != 0) {
                                sum += depth;
                                count++;
                            }
                        }
                    }
                    if (count > 5) {
                        uint16_t averageDepth = static_cast<uint16_t>(sum / count);
                        depthImage.at<uint16_t>(i, j) = averageDepth;
                    }
                }
            }
            float d = (float)depthImage.ptr<uint16_t>(i)[j]; // 深度值
            if (d < 1e-3f) 
            {
                pcl::PointXYZ p;
                p.x = std::numeric_limits<float>::quiet_NaN ();
                p.y = std::numeric_limits<float>::quiet_NaN ();
                p.z = std::numeric_limits<float>::quiet_NaN ();
                Cloud->points.push_back(p);
                continue;
            } // 为0表示没有测量到
            Eigen::Vector3f point;
            point[2] = (d) * (1.0f / depthScale);
            point[0] = (j - cx) * point[2] / fx;
            point[1] = (i - cy) * point[2] / fy;
            Eigen::Vector3f pointWorld = point;
            pcl::PointXYZ p;
            p.x = pointWorld[0];
            p.y = pointWorld[1];
            p.z = pointWorld[2];
            Cloud->points.push_back(p);
        }
    }
}

vector<cv::Scalar> get_color(int n){
	vector<cv::Scalar> colors;
	int k = n/7 + 1;
    colors.push_back(cv::Scalar(0,0,0));
	for(int i = 0; i < k; i++)
	{
		colors.push_back(cv::Scalar(0,0,255)/(i+1.0));
		colors.push_back(cv::Scalar(0,255,0)/(i+1.0));
		colors.push_back(cv::Scalar(255,0,0)/(i+1.0));
		colors.push_back(cv::Scalar(0,255,255)/(i+1.0));
		colors.push_back(cv::Scalar(255,0,255)/(i+1.0));
		colors.push_back(cv::Scalar(255,255,0)/(i+1.0));
		colors.push_back(cv::Scalar(255,255,255)/(i+1.0));
	}
	return colors;
}

void getPlanePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr planepoints, Eigen::Vector3d& normal, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ& center)
{
  planepoints->reserve(cloud->size());
  // std::cout<<"1"<<std::endl;
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero(3,3);
  // std::cout<<"1"<<std::endl;
  // pcl::PointCloud<pcl::PointXYZ>::iterator index = cloud->begin();
  pcl::PointCloud<pcl::PointXYZ>::iterator iter_point;
  // std::cout<<"1"<<std::endl;
  for (iter_point = cloud->begin(); iter_point != cloud->end(); iter_point++)
  {   
    // std::cout<<"2"<<std::endl;
    Eigen::Vector3d ve((*iter_point).x - center.x, (*iter_point).y - center.y, (*iter_point).z - center.z);
    M += ve*ve.transpose();
  }
  // std::cout<<"get M matrix"<<std::endl;
  Eigen::EigenSolver<Eigen::Matrix3d> es(M);
  Eigen::Matrix3d::Index b;
  auto minEigenValue = es.eigenvalues().real().minCoeff(&b);
  double eigenValuesSum = es.eigenvalues().real().sum();
  normal = es.eigenvectors().real().col(b);
  // std::cout<<"get normal"<<std::endl;
  Eigen::Vector3d center_(center.x, center.y, center.z);
  double d = -(normal.dot(center_));
  
  for (iter_point = cloud->begin(); iter_point != cloud->end(); iter_point++)
  {
    Eigen::Vector3d point((*iter_point).x, (*iter_point).y, (*iter_point).z);
    double dis = normal.dot(point) + d;
    Eigen::Vector3d pointShape = point - dis*normal;
    pcl::PointXYZ p(pointShape(0), pointShape(1), pointShape(2));
    planepoints->emplace_back(p);
  }
  // std::cout<<"get plane points"<<std::endl;
}

vector<Eigen::Vector3d> fitRect(pcl::PointCloud<pcl::PointXYZ>& cloud_hull, Eigen::Vector3d & normal, Eigen::Vector3d & center_eigen)
{
  Eigen::Vector3d z_axid = normal;
  Eigen::Vector3d x_point;
  for (auto & iter:(cloud_hull))
  {
    Eigen::Vector3d tmppoint(iter.x, iter.y, iter.z);
    if ((tmppoint - center_eigen).norm() > 0.2)
    {
        x_point = tmppoint;
        break;
    }
  }
//   cout<<"the cor point is "<<x_point(0)<<" "<<x_point(1)<<" "<<x_point(2)<<endl;
  Eigen::Vector3d x_axid = (x_point - center_eigen).normalized();
  Eigen::Vector3d y_axid = (normal.cross(x_axid)).normalized();

//   cout<<"x : "<<x_axid.transpose()<<endl;
//   cout<<"y : "<<y_axid.transpose()<<endl;
//   cout<<"z : "<<z_axid.transpose()<<endl;
  // 从定义的平面坐标系到世界坐标系
  Eigen::Matrix3d rotation2W;

  rotation2W<<x_axid.dot(Eigen::Vector3d::UnitX()), y_axid.dot(Eigen::Vector3d::UnitX()), 
              z_axid.dot(Eigen::Vector3d::UnitX()), x_axid.dot(Eigen::Vector3d::UnitY()),
              y_axid.dot(Eigen::Vector3d::UnitY()), z_axid.dot(Eigen::Vector3d::UnitY()),
              x_axid.dot(Eigen::Vector3d::UnitZ()), y_axid.dot(Eigen::Vector3d::UnitZ()),
              z_axid.dot(Eigen::Vector3d::UnitZ());
  Eigen::Isometry3d T1=Eigen::Isometry3d::Identity();
  T1.rotate (rotation2W);
  T1.pretranslate (center_eigen);
  std::vector<cv::Point2f> hull;
  for (auto & iter:(cloud_hull))
  {
      Eigen::Vector3d new_p = T1.inverse()*Eigen::Vector3d(iter.x, iter.y, iter.z);
      hull.emplace_back(cv::Point2f(new_p(0), new_p(1)));
  }
  LShapedFIT lshaped;
  cv::RotatedRect rr = lshaped.FitBox(&hull);
  std::vector<cv::Point2f> vertices = lshaped.getRectVertex();
  vector<Eigen::Vector3d> edgePoints;
  for (auto & iter: vertices)
  {
    Eigen::Vector3d point(iter.x, iter.y, 0.0);
    edgePoints.emplace_back(T1*point);
  }
  return edgePoints;
}

class PlaneDetectAndPub
{
private:
    ros::NodeHandle nh;
    ros::Publisher pcl_pub1;
    ros::Publisher pcl_pub2;
    ros::Publisher markarry_pub;
    ros::Subscriber imgDepth_sub;
public:
    PlaneDetectAndPub(): nh("~")
    {
        pcl_pub1 = nh.advertise<sensor_msgs::PointCloud2>("plane_points", 10);
        pcl_pub2 = nh.advertise<sensor_msgs::PointCloud2>("edge_points", 10);
        markarry_pub = nh.advertise<visualization_msgs::MarkerArray>("plane_edge",10);
        imgDepth_sub = nh.subscribe<sensor_msgs::Image>("/camera/depth/image_rect_raw", 10, &PlaneDetectAndPub::imagCallback, this);
    }


private:
    void imagCallback(const sensor_msgs::ImageConstPtr& depthImageMsg)
    {
        std::chrono::steady_clock::time_point t0, t1, t2, t3;
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(depthImageMsg, sensor_msgs::image_encodings::TYPE_16UC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat imgDepth = cv_ptr->image;
        myPointCloud::Ptr Cloud = pcl::make_shared<myPointCloud>();
        Cloud->points.reserve(imgDepth.rows * imgDepth.cols);
        Cloud->is_dense = false;
        replaceZeroDepth_and_generatePointCloud(imgDepth, Cloud);
        Cloud->width = imgDepth.cols;
        Cloud->height = imgDepth.rows;  
        cv::Mat imgEdgeByPlane(imgSize, CV_8UC1, cv::Scalar(0));
        cv::Mat imgSeg(imgSize, CV_8UC1, cv::Scalar(0));
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudColored = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        std::vector<ahc::PlaneSeg::shared_ptr> extractedPlanes;
        PlanarContourExtraction pce(*Cloud);
        t0 = std::chrono::steady_clock::now();
        pce.run(imgSeg);
        t1 = std::chrono::steady_clock::now();
        double maxLabel;
        cv::minMaxLoc(imgSeg, 0, &maxLabel);
        std::vector<myPointCloud::Ptr> allPlanes;// 每个簇都单独生成点云
        std::vector<pcl::PointXYZ> allCenterPoint(maxLabel+1);
        // std::vector<cv::Mat> vecImgOcclusion(12);
        for(int i = 0; i <= maxLabel; i++)
        {
            myPointCloud::Ptr tmp1 = pcl::make_shared<myPointCloud>();
            tmp1->points.reserve(imgDepth.cols * imgDepth.rows);
            tmp1->is_dense = false;
            allPlanes.push_back(tmp1);
        }
        vector<cv::Scalar> colorTab = get_color(maxLabel);
        
        for(int row = 0; row < height; row++)
        {
            for(int col = 0; col < width; col++)
            {
                uchar label = (uchar)imgSeg.ptr<uchar>(row)[col];
                if(label == 0)
                {
                    continue;
                }
                float d = (float)imgDepth.ptr<uint16_t>(row)[col]; // 深度值
                pcl::PointXYZRGB pointColord;
                pcl::PointXYZ pointXYZ;
                Eigen::Vector3f point;
                point[2] = (d) * (1.0f / depthScale);
                point[0] = (col - cx) * point[2] / fx;
                point[1] = (row - cy) * point[2] / fy;
                pointColord.x = point[0];
                pointColord.y = point[1];
                pointColord.z = point[2];
                pointXYZ.x = point[0];
                pointXYZ.y = point[1];
                pointXYZ.z = point[2];           
                // uint32_t rgbValue = encodeLabelToColor((int)label);
                // pointColord.rgb = *reinterpret_cast<float*>(&rgbValue);
                pointColord.r = colorTab[label][0];
                pointColord.g = colorTab[label][1];
                pointColord.b = colorTab[label][2];
                
                CloudColored->push_back(pointColord);
                allPlanes[label]->push_back(pointXYZ);
                allCenterPoint[label].x += pointColord.x;
                allCenterPoint[label].y += pointColord.y;
                allCenterPoint[label].z += pointColord.z;
            }
        }
        
        myPointCloudRGB::Ptr CloudColoredOutput  = pcl::make_shared<myPointCloudRGB>();
        myPointCloudRGB::Ptr CloudEdgeOutput  = pcl::make_shared<myPointCloudRGB>();
        CloudColoredOutput->reserve(imgDepth.cols * imgDepth.rows);


        visualization_msgs::MarkerArray ma;
        t2 = std::chrono::steady_clock::now();
        //遍历每一个平面
        //0是没有被检测出来的区域
        for(int i = 1; i <= maxLabel; i ++)
        {
            int numPoint = allPlanes[i]->size();
            if(numPoint < 1000)
            {
                cout<<"the number in this plan is less than 1000, discard the plane."<<endl;
                continue;
            }

            myPointCloud::Ptr cloud_filtered = pcl::make_shared<myPointCloud>();
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(allPlanes[i]);//给滤波对象设置需过滤的点云
            sor.setLeafSize(0.01f, 0.01f, 0.01f);//设置滤波时创建的体素大小为1cm*1cm*1cm的立方体
            sor.filter(*cloud_filtered);//执行滤波处理，存储输出为cloud_filtered

            pcl::PointXYZ tmpCenterPoint;
            tmpCenterPoint.x = allCenterPoint[i].x/(float)numPoint;
            tmpCenterPoint.y = allCenterPoint[i].y/(float)numPoint;
            tmpCenterPoint.z = allCenterPoint[i].z/(float)numPoint;
            myPointCloud::Ptr plane_points = pcl::make_shared<myPointCloud>();
            Eigen::Vector3d normal;
            getPlanePoints(plane_points, normal, cloud_filtered, tmpCenterPoint);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull = pcl::make_shared<myPointCloud>();
            pcl::ConcaveHull<pcl::PointXYZ> chull;
            chull.setInputCloud(plane_points);
            chull.setAlpha(10);
            chull.reconstruct(*cloud_hull);
            vector<Eigen::Vector3d> poly;
            poly.reserve(cloud_hull->size());

            Eigen::Vector3d center_eigen(tmpCenterPoint.x, tmpCenterPoint.y, tmpCenterPoint.z);
            vector<Eigen::Vector3d> rectPoint = fitRect(*cloud_hull, normal, center_eigen);
            double length0 = (rectPoint[0] - rectPoint[1]).norm();
            double length1 = (rectPoint[1] - rectPoint[2]).norm();
            double length2 = (rectPoint[2] - rectPoint[3]).norm();
            double length3 = (rectPoint[3] - rectPoint[0]).norm();
            std::cout << "----------对于平面 " << to_string(i) << ": ----------" << std::endl;
            std::cout << "Point0: " <<  rectPoint[0].transpose() << std::endl;
            std::cout << "Point1: " <<  rectPoint[1].transpose() << std::endl;
            std::cout << "Point2: " <<  rectPoint[2].transpose() << std::endl;
            std::cout << "Point3: " <<  rectPoint[3].transpose() << std::endl;
            std::cout << "4 Edges' length: " << length0 << ", " << length1 << ", " << length2 << ", " << length3 << std::endl;
            std::cout << "Center Point is: " << center_eigen.transpose() << std::endl;
            int k = 0;
            for(int j = 0; j < plane_points->size(); j ++)
            {
                pcl::PointXYZRGB tmpPoint;
                tmpPoint.x = plane_points->points[j].x;
                tmpPoint.y = plane_points->points[j].y;
                tmpPoint.z = plane_points->points[j].z;
                tmpPoint.r = colorTab[i][0];
                tmpPoint.g = colorTab[i][1];
                tmpPoint.b = colorTab[i][2];
                CloudColoredOutput->push_back(tmpPoint);
            }
            for(int j = 0; j < cloud_hull->size(); j ++)
            {
                pcl::PointXYZRGB tmpPoint;
                tmpPoint.x = cloud_hull->points[j].x;
                tmpPoint.y = cloud_hull->points[j].y;
                tmpPoint.z = cloud_hull->points[j].z;
                tmpPoint.r = 255;
                tmpPoint.g = 255;
                tmpPoint.b = 255;
                CloudEdgeOutput->push_back(tmpPoint);
            }
            for(int j = 0; j < rectPoint.size(); j ++)
            {
                pcl::PointXYZRGB tmpPoint;
                tmpPoint.x = rectPoint[j].x();
                tmpPoint.y = rectPoint[j].y();
                tmpPoint.z = rectPoint[j].z();
                tmpPoint.r = colorTab[i][0] * 0.5;
                tmpPoint.g = colorTab[i][1] * 0.5;
                tmpPoint.b = colorTab[i][2] * 0.5;
                CloudEdgeOutput->push_back(tmpPoint);
            }
                //-----------------------------------------
            visualization_msgs::Marker marker;
            marker.header.frame_id = "camera";
            marker.header.stamp = ros::Time::now();
            marker.ns = "hull_" + std::to_string(i);
            marker.id = i;
            marker.type = visualization_msgs::Marker::LINE_LIST;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = 0;
            marker.pose.position.y = 0;
            marker.pose.position.z = 0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.03;
            marker.scale.y = 0.03;
            marker.scale.z = 0.03;
            marker.color.a = 1.0;
            const double r = colorTab[i][0] * 1.0/255.0;
            const double g = colorTab[i][1] * 1.0/255.0;
            const double b = colorTab[i][2] * 1.0/255.0;
            marker.points.reserve(8);
            marker.colors.reserve(8);
            for (size_t j = 0; j < cloud_hull->size(); j++)
            {
                geometry_msgs::Point point;
                std_msgs::ColorRGBA point_color;
                point.x = cloud_hull->points[j].x;
                point.y = cloud_hull->points[j].y;
                point.z = cloud_hull->points[j].z;
                point_color.r = (float)r;
                point_color.g = (float)g;
                point_color.b = (float)b;
                point_color.a = 1.0;
                marker.colors.push_back(point_color);
                marker.points.push_back(point);
                point.x = cloud_hull->points[(j+1) % cloud_hull->size()].x;
                point.y = cloud_hull->points[(j+1) % cloud_hull->size()].y;
                point.z = cloud_hull->points[(j+1) % cloud_hull->size()].z;
                point_color.r = (float)r;
                point_color.g = (float)g;
                point_color.b = (float)b;
                point_color.a = 1.0;
                marker.colors.push_back(point_color);
                marker.points.push_back(point);
                marker.frame_locked = true;
            }
            ma.markers.push_back(marker);
        }
        t3 = std::chrono::steady_clock::now();
        sensor_msgs::PointCloud2 output1, output2;
        pcl::toROSMsg(*CloudColoredOutput, output1);
        pcl::toROSMsg(*CloudEdgeOutput, output2);
        output1.header.frame_id = "camera";
        output2.header.frame_id = "camera";
        output1.header.stamp = ros::Time::now();
        output2.header.stamp = ros::Time::now();
        {
            pcl_pub1.publish(output1);
            pcl_pub2.publish(output2);
            markarry_pub.publish(ma);
        }

        cout << "平面分割: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t1 - t0 ).count()*1000 << " ms ." <<endl;
        cout << "处理每个平面: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t3 - t2 ).count()*1000 << " ms ." <<endl;

    }

};


int main(int argc, char** argv)
{
    // Eigen::Isometry3d M;
    // double theta = 152.5/57.3;
    // M.matrix() << 0, cos(theta), sin(theta), 0.1412,
    //            -1, 0, 0, 0,
    //            0, -sin(theta), cos(theta), -0.2474,
    //            0, 0, 0, 1;
    // std::cout << M.matrix() << std::endl;
    // std::cout << "----------------" << std::endl;
    // std::cout << M.inverse().matrix() << std::endl;
    //     std::cout << "----------------" << std::endl;
    // Eigen::Vector3d euler_angles = M.inverse().rotation().eulerAngles(0, 2, 1); // ZYX顺序
    // std::cout << "----------------" << std::endl;
    // std::cout << "Translation (m): " << M.inverse().translation().transpose() << std::endl;
    // std::cout << "Euler angles (rad): " << euler_angles.transpose() << std::endl;

    // // 将弧度转换为角度
    // Eigen::Vector3d euler_angles_deg = euler_angles * 180.0 / M_PI;
    // std::cout << "Euler angles (deg): " << euler_angles_deg.transpose() << std::endl;
    // Eigen::Quaterniond quaternion(M.inverse().matrix().block<3, 3>(0, 0));
    // std::cout << "Quaternion: " << quaternion.coeffs().transpose() << std::endl;
    // Eigen::Quaterniond quaternion0(M.matrix().block<3, 3>(0, 0));
    // std::cout << "Quaternion0: " << quaternion0.coeffs().transpose() << std::endl;

    ros::init(argc, argv, "color_pointcloud_publisher");
    PlaneDetectAndPub PDAP;
    ros::spin();

    int k = 0;
    return 0;
}