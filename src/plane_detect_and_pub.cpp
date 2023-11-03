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
#include <pcl/filters/passthrough.h>
#include "plane_fitter_pcl.hpp"
#include "AHCPlaneSeg.hpp"
#include "lshaped_fitting.h"
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
const cv::Size imgSize(640, 480);
const int width = 640;
const int height = 480;
typedef pcl::PointCloud<pcl::PointXYZRGB> myPointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZ> myPointCloud;
typedef Eigen::Matrix<double,6,1> Vector6d;
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
// const float cx = 325.44140625;
// const float cy = 236.3984375;
// const float fx = 460.2265625;
// const float fy = 460.2265625;
// const float depthScale = 1000.0;

//--------D455-------------
const float cx = 328.57140625;
const float cy = 240.3284375;
const float fx = 390.2265625;
const float fy = 390.2265625;
const float depthScale = 1000.0;

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

int findMinIndex(const std::vector<float>& zValues) {
    if (zValues.empty()) {
        std::cerr << "Vector is empty." << std::endl;
        return -1;  // Return -1 to indicate an error or empty vector
    }

    float minVal = std::numeric_limits<float>::max();  // Initialize with a large value
    int minIndex = -1;  // Initialize with an invalid index

    for (size_t i = 0; i < zValues.size(); ++i) {
        if (zValues[i] < minVal) {
            minVal = zValues[i];
            minIndex = static_cast<int>(i);
        }
    }

    return minIndex;
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

// double calDistance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
// {
//     p1.no
//     double disdis = 
// }
class PlaneDetectAndPub
{
private:
    ros::NodeHandle nh;
    ros::Publisher pcl_pub1;
    ros::Publisher pcl_pub2;
    ros::Publisher pcl_pub3;
    ros::Publisher markarry_pub;
    ros::Subscriber imgDepth_sub;
    tf::TransformListener listener_;
    tf::StampedTransform transform_;
    tf::TransformListener listener2_;
    tf::StampedTransform transform2_;
    geometry_msgs::PoseStamped T_w_c;
    geometry_msgs::PoseStamped T_w_l;
    // Eigen::Isometry3d Twc;
public:
    PlaneDetectAndPub(): nh("~")
    {
        
        pcl_pub1 = nh.advertise<sensor_msgs::PointCloud2>("plane_points", 10);
        pcl_pub2 = nh.advertise<sensor_msgs::PointCloud2>("edge_points", 10);
        pcl_pub3 = nh.advertise<sensor_msgs::PointCloud2>("no_ground_points", 10);
        markarry_pub = nh.advertise<visualization_msgs::MarkerArray>("plane_edge",10);
        imgDepth_sub = nh.subscribe<sensor_msgs::Image>("/camera/depth/image_rect_raw", 1, &PlaneDetectAndPub::imagCallback, this);
    }


private:
    void imagCallback(const sensor_msgs::ImageConstPtr& depthImageMsg)
    {
        try
        {
            listener2_.waitForTransform("3dmap", "velodyne", ros::Time::now(), ros::Duration(0.5));
            std::cout << "--------------------------------" << std::endl;
            listener2_.lookupTransform ("3dmap", "velodyne", ros::Time::now(), transform2_);
        }
        catch(tf::TransformException &ex)
        {
            ROS_ERROR("%s", ex.what());
            return; 
        }
        std::cout << transform2_.getRotation().x() << ", " << transform2_.getRotation().y()  << ", " << transform2_.getRotation().z()  << ", " <<  transform2_.getRotation().w() << std::endl;
        double qx = transform2_.getRotation().x();
        double qy = transform2_.getRotation().y();
        double qz = transform2_.getRotation().z();
        double qw = transform2_.getRotation().w();
        double R[3][3] = {
        {1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)},
        {2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)},
        {2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)}};

        // 计算欧拉角（XYZ顺序）
        double roll = atan2(R[2][1], R[2][2]);
        double pitch = atan2(-R[2][0], sqrt(R[2][1]*R[2][1] + R[2][2]*R[2][2]));
        double yaw = atan2(R[1][0], R[0][0]);

        // 将弧度转化为度
        roll = roll * 180.0 / M_PI;
        pitch = pitch * 180.0 / M_PI;
        yaw = yaw * 180.0 / M_PI;

        // 输出欧拉角
        std::cout << "Roll: " << roll << " degrees" << std::endl;
        std::cout << "Pitch: " << pitch << " degrees" << std::endl;
        std::cout << "Yaw: " << yaw << " degrees" << std::endl;
        T_w_l.pose.orientation.x = transform2_.getRotation().x();
        T_w_l.pose.orientation.y = transform2_.getRotation().y();
        T_w_l.pose.orientation.z = transform2_.getRotation().z();
        T_w_l.pose.orientation.w = transform2_.getRotation().w();
        T_w_l.pose.position.x = transform2_.getOrigin().x();
        T_w_l.pose.position.y = transform2_.getOrigin().y();
        T_w_l.pose.position.z = transform2_.getOrigin().z();
        Eigen::Vector3d t_w_l(T_w_l.pose.position.x, T_w_l.pose.position.y, T_w_l.pose.position.z);
        Eigen::Quaterniond q_w_l(T_w_l.pose.orientation.w, T_w_l.pose.orientation.x, T_w_l.pose.orientation.y, T_w_l.pose.orientation.z);
        Eigen::Isometry3d Twl(q_w_l);
        Twl.pretranslate(t_w_l);
        // double q0 = w;
        // double q1 = x;
        // double q2 = y;
        // double q3 = z;
        // double roll_deg = 1 * std::atan2(2*q2*q3 + 2*q0*q1, -2 * q1 * q1 -2 * q2 * q2 + 1) * 57.3;
        // double pitch_deg = 1 * std::asin(2*q0*q2 - 2*q3*q1) * 57.3;
        // double yaw_deg = 1 * std::atan2(2*q1*q2 + 2*q0*q3, -2 * q2 * q2 -2 * q3 * q3 + 1) * 57.3;
        // std::cout << "Roll: " << roll_deg << " degrees" << std::endl;
        // std::cout << "Pitch: " << pitch_deg << " degrees" << std::endl;
        // std::cout << "Yaw: " << yaw_deg << " degrees" << std::endl;

        try
        {
            listener_.waitForTransform("3dmap", "camera", ros::Time::now(), ros::Duration(0.5));
            // std::cout << "--------------------------------" << std::endl;
            listener_.lookupTransform ("3dmap", "camera", ros::Time::now(), transform_);
        }
        catch(tf::TransformException &ex)
        {
            ROS_ERROR("%s", ex.what());
            return;
        }
        T_w_c.pose.orientation.x = transform_.getRotation().x();
        T_w_c.pose.orientation.y = transform_.getRotation().y();
        T_w_c.pose.orientation.z = transform_.getRotation().z();
        T_w_c.pose.orientation.w = transform_.getRotation().w();
        T_w_c.pose.position.x = transform_.getOrigin().x();
        T_w_c.pose.position.y = transform_.getOrigin().y();
        T_w_c.pose.position.z = transform_.getOrigin().z();
        Eigen::Vector3f t(T_w_c.pose.position.x, T_w_c.pose.position.y, T_w_c.pose.position.z);
        Eigen::Quaternionf q(T_w_c.pose.orientation.w, T_w_c.pose.orientation.x, T_w_c.pose.orientation.y, T_w_c.pose.orientation.z);
        Eigen::Isometry3f Twc(q);
        Twc.pretranslate(t);
        // std::cout << Twc.matrix() << std::endl;
        // std::cout << T_w_c.pose.orientation.
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
        double rateValid = cv::countNonZero(imgDepth) / (double)(width * height); 
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
        std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> planeNormalAndCenter;
        PlanarContourExtraction pce(*Cloud);
        t0 = std::chrono::steady_clock::now();
        pce.run(imgSeg, planeNormalAndCenter);
        cv::imshow("imgSeg", imgSeg);
        cv::waitKey(1);
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
        myPointCloud::Ptr CloudNoGround  = pcl::make_shared<myPointCloud>();
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
                Eigen::Vector3f point, pointTransed;
                point[2] = (d) * (1.0f / depthScale);
                point[0] = (col - cx) * point[2] / fx;
                point[1] = (row - cy) * point[2] / fy;
                // point[3] = 0.0f;
                pointTransed = Twc * point;
                pointColord.x = pointTransed[0];
                pointColord.y = pointTransed[1];
                pointColord.z = pointTransed[2];
                pointXYZ.x = pointTransed[0];
                pointXYZ.y = pointTransed[1];
                pointXYZ.z = pointTransed[2];           
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

                // if(row > 0.5 * height)
                // {
                //     CloudNoGround->points.push_back(pointXYZ);
                // }
            }
        }
        
        myPointCloudRGB::Ptr CloudColoredOutput  = pcl::make_shared<myPointCloudRGB>();
        myPointCloudRGB::Ptr CloudEdgeOutput  = pcl::make_shared<myPointCloudRGB>();

        // CloudNoGround->clear();
        CloudColoredOutput->reserve(imgDepth.cols * imgDepth.rows);
        //通过中心点z值来判断地面
        vector<float> zValues;
        // zValues.push_back(100);
        for(int i = 1; i <= maxLabel; i++)
        {
            zValues.push_back(allCenterPoint[i].z/(float)allPlanes[i]->size());
        }
        int minIndex = findMinIndex(zValues) + 1;
        std::cout << "图像的空洞率为：  " << (100.0 - rateValid*100) << "%" << std::endl;
        std::cout << "地面平面编号为：  " << minIndex << std::endl;
        Eigen::Vector3d refPoint;
        refPoint = Eigen::Vector3d(T_w_l.pose.position.x, T_w_l.pose.position.y, T_w_l.pose.position.z) - Twl * Eigen::Vector3d(0.00938, 0, 0);
        double height_ground = zValues[minIndex - 1];
        refPoint[2] = height_ground;
        
        // pcl::PassThrough<pcl::PointXYZ> passfilter;
        // passfilter.setInputCloud(CloudNoGround);
        // passfilter.setFilterFieldName("z");   // 设置过滤的轴，可以是"x"、"y"、"z"等
        // passfilter.setFilterLimits(height_ground, height_ground + 1.0); 
        // passfilter.filter(*CloudNoGround);
        for(int row = 0; row < height; row++)
        {
            for(int col = 0; col < width; col++)
            {
                uchar label = (uchar)imgSeg.ptr<uchar>(row)[col];
                if(label == minIndex || row > 0.5 * height)
                {
                    continue;
                }
                float d = (float)imgDepth.ptr<uint16_t>(row)[col]; // 深度值
                pcl::PointXYZ pointXYZ;
                Eigen::Vector3f point, pointTransed;
                point[2] = (d) * (1.0f / depthScale);
                point[0] = (col - cx) * point[2] / fx;
                point[1] = (row - cy) * point[2] / fy;
                pointTransed = Twc * point;
                pointXYZ.x = pointTransed[0];
                pointXYZ.y = pointTransed[1];
                pointXYZ.z = pointTransed[2];  
                if(label != 0)
                {
                    CloudNoGround->points.push_back(pointXYZ);
                }
                else if(pointXYZ.z > (height_ground + 0.1))
                    CloudNoGround->points.push_back(pointXYZ);
            }
        }

        visualization_msgs::MarkerArray ma;
        ma.markers.clear();
        t2 = std::chrono::steady_clock::now();
        Eigen::Matrix3d rotationMatrixRefine;
        

        //检验地面法向量
        {
            int numPoint = allPlanes[minIndex]->size();
            myPointCloud::Ptr cloud_filtered = pcl::make_shared<myPointCloud>();
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(allPlanes[minIndex]);//给滤波对象设置需过滤的点云
            sor.setLeafSize(0.01f, 0.01f, 0.01f);//设置滤波时创建的体素大小为1cm*1cm*1cm的立方体
            sor.filter(*cloud_filtered);//执行滤波处理，存储输出为cloud_filtered

            pcl::PointXYZ tmpCenterPoint;
            tmpCenterPoint.x = allCenterPoint[minIndex].x/(float)numPoint;
            tmpCenterPoint.y = allCenterPoint[minIndex].y/(float)numPoint;
            tmpCenterPoint.z = allCenterPoint[minIndex].z/(float)numPoint;
            myPointCloud::Ptr plane_points = pcl::make_shared<myPointCloud>();
            Eigen::Vector3d normal;
            getPlanePoints(plane_points, normal, cloud_filtered, tmpCenterPoint);
            std::cout << "地面的旧法向量为：" << normal.transpose() << std::endl;
            Eigen::Vector3d targetNormal(0, 0, 1);

            rotationMatrixRefine = Eigen::Quaterniond::FromTwoVectors(normal, targetNormal).toRotationMatrix();
            // 应用旋转矩阵到当前法向量
            Eigen::Vector3d newNormal = rotationMatrixRefine * normal;
            std::cout << "地面的新法向量为：" << newNormal.transpose() << std::endl;
        }
        Eigen::Matrix3f rotationMatrixRefineFloat = rotationMatrixRefine.inverse().cast<float>();
        Eigen::Matrix4f transformMatrixRefine = Eigen::Matrix4f::Identity();
        transformMatrixRefine.block<3, 3>(0, 0) = rotationMatrixRefineFloat; // 将旋转矩阵插入左上角
        transformMatrixRefine.block<3, 1>(0, 3) = Eigen::Vector3f::Zero();
        // std::cout << transformMatrixRefine << std::endl; 
        // for(int i = 1; i <= maxLabel; i++)
        // {
        //     allCenterPoint[i] = allCenterPoint[i] * Twc.inverse().matrix() * transformMatrixRefine * Twc.matrix();
        // }
        Eigen::Vector3d ground_normal(0, 0, 1);
        //遍历每一个平面
        //0是没有被检测出来的区域
        for(int i = 1; i <= maxLabel; i ++)
        {

            int numPoint = allPlanes[i]->size();
            if(numPoint < 200)
            {
                // cout<<"the number in this plan is less than 1000, discard the plane."<<endl;
                continue;
            }
            Eigen::Matrix4f kkk = Twc.matrix() * transformMatrixRefine * Twc.inverse().matrix();
            // std::cout << kkk << std::endl; 
            Eigen::Matrix4d kkk_double = kkk.cast<double>();
            Eigen::Isometry3d kkk_trans;
            kkk_trans.matrix() = kkk_double;
            myPointCloud::Ptr cloud_filtered = pcl::make_shared<myPointCloud>();
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            // pcl::transformPointCloud(*allPlanes[i], *allPlanes[i], Twc.inverse().matrix());//先转
            // pcl::transformPointCloud(*allPlanes[i], *allPlanes[i], transformMatrixRefine);//先转
            // pcl::transformPointCloud(*allPlanes[i], *allPlanes[i], Twc.matrix());//先转
            for(int j = 0; j < numPoint; ++j)
            {
                Eigen::Vector3d tmpPoint(allPlanes[i]->points[j].x, allPlanes[i]->points[j].y, allPlanes[i]->points[j].z);
                Eigen::Vector3d tmpPointTransed = kkk_trans * tmpPoint;
                allPlanes[i]->points[j].x = tmpPointTransed[0];
                allPlanes[i]->points[j].y = tmpPointTransed[1];
                allPlanes[i]->points[j].z = tmpPointTransed[2];
                // if(j == 100)
                // {
                //     std::cout << "转换前的点：" << tmpPoint.transpose() << std::endl;
                //     std::cout << "转换后的点：" << tmpPointTransed.transpose() << std::endl;
                // }
                
            }
            sor.setInputCloud(allPlanes[i]);//给滤波对象设置需过滤的点云
            sor.setLeafSize(0.01f, 0.01f, 0.01f);//设置滤波时创建的体素大小为1cm*1cm*1cm的立方体
            sor.filter(*cloud_filtered);//执行滤波处理，存储输出为cloud_filtered

            pcl::PointXYZ tmpCenterPoint;
            tmpCenterPoint.x = allCenterPoint[i].x/(float)numPoint;
            tmpCenterPoint.y = allCenterPoint[i].y/(float)numPoint;
            tmpCenterPoint.z = allCenterPoint[i].z/(float)numPoint;
            Eigen::Vector3d center_eigen(tmpCenterPoint.x, tmpCenterPoint.y, tmpCenterPoint.z);

            center_eigen = kkk_trans * center_eigen;
            tmpCenterPoint.x = (float)center_eigen[0];
            tmpCenterPoint.y = (float)center_eigen[1];
            tmpCenterPoint.y = (float)center_eigen[2];
            myPointCloud::Ptr plane_points = pcl::make_shared<myPointCloud>();
            Eigen::Vector3d normal;
            getPlanePoints(plane_points, normal, cloud_filtered, tmpCenterPoint);
            std::cout << "平面的法向量为：" << normal.transpose() << std::endl;
            // if(normal[2] < 0.7 && normal[2] > -0.7)
            //     continue;
            // *CloudNoGround += *plane_points;
            if(i == minIndex)//地面
            {
                ground_normal = normal;
                continue;
            }
                
            int classTerr; //0为台阶，1为斜坡，2为其他
            if(normal[2] < 0.6 && normal[2] > -0.6)
            {
                classTerr = 2;
                continue;
            }
            else if(normal[2] < 0.75 && normal[2] > -0.75)
            {
                classTerr = 1;
            }
            else
            {
                classTerr = 0;
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull = pcl::make_shared<myPointCloud>();
            pcl::ConcaveHull<pcl::PointXYZ> chull;
            chull.setInputCloud(plane_points);
            chull.setAlpha(10);
            chull.reconstruct(*cloud_hull);
            vector<Eigen::Vector3d> poly;
            poly.reserve(cloud_hull->size());

            vector<Eigen::Vector3d> rectPoint = fitRect(*cloud_hull, normal, center_eigen);
            double length0 = (rectPoint[0] - rectPoint[1]).norm();
            double length1 = (rectPoint[1] - rectPoint[2]).norm();
            double length2 = (rectPoint[2] - rectPoint[3]).norm();
            double length3 = (rectPoint[3] - rectPoint[0]).norm();
            
            double dot_product = normal.dot(ground_normal);
            // 计算向量A的模长
            double norm_normal = normal.norm();
            // 计算向量B的模长
            double norm_ground_normal = ground_normal.norm();
            // 计算角度（弧度）
            double angle_radians = std::acos(dot_product / (norm_normal * norm_ground_normal));
            // 将弧度转换为度
            double angle_degrees = angle_radians * (180.0 / M_PI);  
            if(classTerr == 0)
            {
                // std::cout << "----------对于平面 " << to_string(i) << ": ----------" << std::endl;
                std::cout << "台阶角点0坐标: " <<  rectPoint[0].transpose() << ", 相对参考点距离为：" << (rectPoint[0]-refPoint).norm() << std::endl;
                std::cout << refPoint.transpose() << std::endl;
                std::cout << "台阶角点1坐标: " <<  rectPoint[1].transpose() << ", 相对参考点距离为：" << (rectPoint[1]-refPoint).norm() << std::endl;
                std::cout << "台阶角点2坐标: " <<  rectPoint[2].transpose() << ", 相对参考点距离为：" << (rectPoint[2]-refPoint).norm() << std::endl;
                std::cout << "台阶角点3坐标: " <<  rectPoint[3].transpose() << ", 相对参考点距离为：" << (rectPoint[3]-refPoint).norm() << std::endl;
                std::cout << "台阶长度为： " << std::max(length0, length1) << "m." << std::endl;
                std::cout << "台阶宽度为： " << std::min(length0, length1) << "m." << std::endl;
                // std::cout << "Center Point is: " << center_eigen.transpose() << std::endl;
                // std::cout << "Norm vector is " << normal.transpose() << std::endl;
                std::cout << "台阶高度为:  " << center_eigen[2] - height_ground << "m."<< std::endl;
                // 输出角度
                std::cout << "台阶角度为:  " << angle_degrees << "°." <<  std::endl;
            }          
            else if(classTerr == 1)
            {
                std::cout << "----------对于台阶 " << to_string(i) << ": ----------" << std::endl;
                std::cout << "斜坡角点0坐标: " <<  rectPoint[0].transpose() << std::endl;
                std::cout << "斜坡角点1坐标: " <<  rectPoint[1].transpose() << std::endl;
                std::cout << "斜坡角点2坐标: " <<  rectPoint[2].transpose() << std::endl;
                std::cout << "斜坡角点3坐标: " <<  rectPoint[3].transpose() << std::endl;
                std::cout << "斜坡长度为： " << std::max(length0, length1) << "m." << std::endl;
                std::cout << "斜坡宽度为： " << std::min(length0, length1) << "m." << std::endl;
                // std::cout << "Center Point is: " << center_eigen.transpose() << std::endl;
                // std::cout << "Norm vector is " << normal.transpose() << std::endl;
                std::cout << "斜坡高度为:  " << center_eigen[2] - height_ground << "m."<< std::endl;
                // 输出角度
                std::cout << "斜坡角度为:  " << angle_degrees << "°." <<  std::endl;
            }

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
            marker.header.frame_id = "3dmap";
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
        sensor_msgs::PointCloud2 output1, output2, output3;
        pcl::toROSMsg(*CloudColoredOutput, output1);
        pcl::toROSMsg(*CloudEdgeOutput, output2);
        pcl::toROSMsg(*CloudNoGround, output3);
        output1.header.frame_id = "3dmap";
        output2.header.frame_id = "3dmap";
        output3.header.frame_id = "3dmap";
        output1.header.stamp = ros::Time::now();
        output2.header.stamp = ros::Time::now();
        output3.header.stamp = ros::Time::now();
        {
            pcl_pub1.publish(output1);
            pcl_pub2.publish(output2);
            pcl_pub3.publish(output3);
            markarry_pub.publish(ma);
        }

        cout << "平面分割用时: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t1 - t0 ).count()*1000 << " ms." <<endl;
        cout << "平面处理用时: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t3 - t2 ).count()*1000 << " ms." <<endl;

    }

};


int main(int argc, char** argv)
{
    Eigen::Isometry3d M;
    double theta = 151/57.3;
    M.matrix() << 0, cos(theta), sin(theta), 0.1412,
               -1, 0, 0, 0,
               0, -sin(theta), cos(theta), -0.2474,
               0, 0, 0, 1;
    std::cout << M.matrix() << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << M.inverse().matrix() << std::endl;
        std::cout << "----------------" << std::endl;
    Eigen::Vector3d euler_angles = M.inverse().rotation().eulerAngles(0, 2, 1); // ZYX顺序
    std::cout << "----------------" << std::endl;
    std::cout << "Translation (m): " << M.inverse().translation().transpose() << std::endl;
    std::cout << "Euler angles (rad): " << euler_angles.transpose() << std::endl;

    // 将弧度转换为角度
    Eigen::Vector3d euler_angles_deg = euler_angles * 180.0 / M_PI;
    std::cout << "Euler angles (deg): " << euler_angles_deg.transpose() << std::endl;
    Eigen::Quaterniond quaternion(M.inverse().matrix().block<3, 3>(0, 0));
    std::cout << "Quaternion: " << quaternion.coeffs().transpose() << std::endl;
    Eigen::Quaterniond quaternion0(M.matrix().block<3, 3>(0, 0));
    std::cout << "Quaternion0: " << quaternion0.coeffs().transpose() << std::endl;
    std::cout << "??" << std::endl;
    ros::init(argc, argv, "color_pointcloud_publisher");
    PlaneDetectAndPub PDAP;
    ros::spin();

    int k = 0;
    return 0;
}