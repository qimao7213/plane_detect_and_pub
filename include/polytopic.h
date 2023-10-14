#ifndef POLYTOPIC_H_
#define POLYTOPIC_H_
#include <iostream>
#include <Eigen/Core>
#include <vector>
using namespace std;

class polytopic
{
private:
    Eigen::Vector3f normal;
    // 需要考虑有没有孔的平面, 外轮廓的方向和polygon的表达方式一致,一个多边形只有一个外轮廓和多个内轮廓
    // 没有把终点和起始点连接起来，即最后一个点是终点不是起始点
    // 3D下的points位于相机坐标系下
    vector<Eigen::Vector3f> outer_points;
    vector<vector<Eigen::Vector3f>> inners_points;
    size_t N;
    
    Eigen::Vector3f center;
    double MSE;
public:
    polytopic()
    {

    }
    // 一定注意外圈顺时针，内圈逆时针
    // 使用内外轮廓来初始化多边形
    polytopic(const Eigen::Vector3f & normal_, const vector<vector<Eigen::Vector3f>>  & contours, const size_t N_, const Eigen::Vector3f & center_, const double MSE_)
    {
        normal = normal_;
        outer_points = *contours.begin();
        inners_points = vector<vector<Eigen::Vector3f>>(contours.begin() + 1, contours.end());
        N = N_;
        center = center_;
        MSE = MSE_;
    }
    ~polytopic()
    {
        outer_points.clear();
        inners_points.clear();
    }
};

#endif
