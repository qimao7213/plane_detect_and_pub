// Copyright (C) 2014,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <vector>
#include <set>
#include <queue>
#include <map>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
//#define DEBUG_CLUSTER
//#define DEBUG_CALC
//#define DEBUG_INIT
//#define EVAL_SPEED

#include "AHCTypes.hpp"
#include "AHCPlaneSeg.hpp"
#include "AHCParamSet.hpp"
#include "AHCUtils.hpp"

// #include "matplotlib/matplotlibcpp.h"
// namespace plt = matplotlibcpp;

// #define TEST_IDEAL_POINTCLOUD
// #define DEBUG_CONTOURS
typedef Eigen::Matrix<double,6,1> Vector6d;
namespace ahc {
	using ahc::utils::Timer;
	using ahc::utils::pseudocolor;

	/**
	 *  \brief An example of Image3D struct as an adaptor for any kind of point cloud to be used by our ahc::PlaneFitter
	 *
	 *  \details A valid Image3D struct should implements the following three member functions:
	 *  1. int width()
	 *     return the #pixels for each row of the point cloud
	 *  2. int height()
	 *     return the #pixels for each column of the point cloud
	 *  3. bool get(const int i, const int j, double &x, double &y, double &z) const
	 *     access the xyz coordinate of the point at i-th-row j-th-column, return true if success and false otherwise (due to NaN depth or any other reasons)
	 */
	struct NullImage3D {
		int width() { return 0; }
		int height() { return 0; }
		//get point at row i, column j
		bool get(const int i, const int j, double &x, double &y, double &z) const { return false; }
	};

	//three types of erode operation for segmentation refinement
	enum ErodeType {
		ERODE_NONE=0,		//no erode
		ERODE_SEG_BORDER=1,	//erode only borders between two segments
		ERODE_ALL_BORDER=2	//erode all borders, either between two segments or between segment and "black"
	};

	/**
	 *  \brief ahc::PlaneFitter implements the Agglomerative Hierarchical Clustering based fast plane extraction
	 *
	 *  \details note: default parameters assume point's unit is mm
	 */
	template <class Image3D>
	struct PlaneFitter {
		/************************************************************************/
		/* Internal Classes                                                     */
		/************************************************************************/
		//for sorting PlaneSeg by size-decreasing order
		struct PlaneSegSizeCmp {
			bool operator()(const PlaneSeg::shared_ptr& a,
				const PlaneSeg::shared_ptr& b) const {
					return b->N < a->N;
			}
		};

		//for maintaining the Min MSE heap of PlaneSeg
		struct PlaneSegMinMSECmp {
			bool operator()(const PlaneSeg::shared_ptr& a,
				const PlaneSeg::shared_ptr& b) const {
				return b->mse < a->mse;
			}
		};
		typedef std::priority_queue<PlaneSeg::shared_ptr,
			std::vector<PlaneSeg::shared_ptr>,
			PlaneSegMinMSECmp> PlaneSegMinMSEQueue;

		/************************************************************************/
		/* Public Class Members                                                 */
		/************************************************************************/
		//input
		const Image3D *points;	//dim=<heightxwidthx3>, no ownership
		int width, height;		//witdth=#cols, height=#rows (size of the input point cloud)

		int maxStep;			//max number of steps for merging clusters
		int minSupport;			//min number of supporting point
		int windowWidth;		//make sure width is divisible by windowWidth
		int windowHeight;		//similarly for height and windowHeight
		bool doRefine;			//perform refinement of details or not
		ErodeType erodeType;

		ParamSet params;		//sets of parameters controlling dynamic thresholds T_mse, T_ang, T_dz

		//output
		ahc::shared_ptr<DisjointSet> ds;//with ownership, this disjoint set maintains membership of initial window/blocks during AHC merging
		std::vector<PlaneSeg::shared_ptr> extractedPlanes;//a set of extracted planes
		cv::Mat membershipImg;//segmentation map of the input pointcloud, membershipImg(i,j) records which plane (plid, i.e. plane id) this pixel/point (i,j) belongs to

		//intermediate
		std::map<int,int> rid2plid;		//extractedPlanes[rid2plid[rootid]].rid==rootid, i.e. rid2plid[rid] gives the idx of a plane in extractedPlanes
		std::vector<int> blkMap;	//(i,j) block belong to extractedPlanes[blkMap[i*Nh+j]]
		//std::vector<std::vector<int>> blkMembership; //blkMembership[i] contains all block id for extractedPlanes[i]
		bool dirtyBlkMbship;
		std::vector<cv::Vec3b> colors;
		std::vector<std::pair<int,int>> rfQueue;//for region grow/floodfill, p.first=pixidx, p.second=plid
		bool drawCoarseBorder;
		//std::vector<PlaneSeg::Stats> blkStats;
#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
		std::string saveDir;
#endif
#ifdef DEBUG_CALC
		std::vector<int>	numNodes;
		std::vector<int>	numEdges;
		std::vector<int>	mseNodeDegree;
		int maxIndvidualNodeDegree;
#endif

		/************************************************************************/
		/* Public Class Functions                                               */
		/************************************************************************/
		PlaneFitter() : points(0), width(0), height(0),
			maxStep(100000), minSupport(3000),
			windowWidth(10), windowHeight(10),
			doRefine(true), erodeType(ERODE_ALL_BORDER),
			dirtyBlkMbship(true), drawCoarseBorder(false)
		{
			static const unsigned char default_colors[10][3] =
			{
				{255, 0, 0},
				{255, 255, 0},
				{100, 20, 50},
				{0, 30, 255},
				{10, 255, 60},
				{80, 10, 100},
				{0, 255, 200},
				{10, 60, 60},
				{255, 0, 128},
				{60, 128, 128}
			};
			for(int i=0; i<10; ++i) {
				colors.push_back(cv::Vec3b(default_colors[i]));
			}
		}

		~PlaneFitter() {}

		/**
		 *  \brief clear/reset for next run
		 */
		void clear() {
			this->points=0;
			this->extractedPlanes.clear();
			ds.reset();
			rid2plid.clear();
			blkMap.clear();
			rfQueue.clear();
			//blkStats.clear();
			dirtyBlkMbship=true;
		}

		/**
		 *  \brief run AHC plane fitting on one frame of point cloud pointsIn
		 *
		 *  \param [in] pointsIn a frame of point cloud
		 *  \param [out] pMembership pointer to segmentation membership vector, each pMembership->at(i) is a vector of pixel indices that belong to the i-th extracted plane
		 *  \param [out] pSeg a 3-channel RGB image as another form of output of segmentation
		 *  \param [in] pIdxMap usually not needed (reserved for KinectSLAM to input pixel index map)
		 *  \param [in] verbose print out cluster steps and #planes or not
		 *  \return when compiled without EVAL_SPEED: 0 if pointsIn==0 and 1 otherwise; when compiled with EVAL_SPEED: total running time for this frame
		 *
		 *  \details this function corresponds to Algorithm 1 in our paper
		 */
		double run(const Image3D* pointsIn,
			cv::Mat &imgSeg,
			std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>>& planeNormalAndCenter,
			std::vector<std::vector<int>>* pMembership=0,
			cv::Mat* pSeg=0, 
			const std::vector<int> * const pIdxMap=0, bool verbose=0)
		{
			if(!pointsIn) return 0;
#ifdef EVAL_SPEED
			Timer timer(1000), timer2(1000);
			timer.tic(); timer2.tic();
#endif
			clear();
			this->points = pointsIn;
			this->height = points->height();
			this->width  = points->width();
			this->ds.reset(new DisjointSet((height/windowHeight)*(width/windowWidth)));
			// std::cout<<"PlaneSegMinMSEQueue minQ "<<std::endl;
			PlaneSegMinMSEQueue minQ;
			this->initGraph(minQ);
#ifdef EVAL_SPEED
			timer.toctic("init time");
#endif
			int step=this->ahCluster(minQ);
#ifdef EVAL_SPEED
			timer.toctic("cluster time");
#endif
			if(doRefine) {
				this->refineDetails(pMembership, pIdxMap, pSeg, imgSeg, planeNormalAndCenter);
#ifdef EVAL_SPEED
				timer.toctic("refine time");
#endif
			} else {
				if(pMembership) {
					this->findMembership(*pMembership, pIdxMap);
				}
				if(pSeg) {
					this->plotSegmentImage(pSeg, minSupport);
				}
#ifdef EVAL_SPEED
				timer.toctic("return time");
#endif
			}
			if(verbose) {
				std::cout<<"#step="<<step<<", #extractedPlanes="
					<<this->extractedPlanes.size()<<std::endl;
			}
#ifdef EVAL_SPEED
			return timer2.toc();
#endif
			return 1;
		}

		/**
		 *  \brief print out the current parameters
		 */
		void logParams() const {
#define TMP_LOG_VAR(var) << #var "="<<(var)<<"\n"
			std::cout<<"[PlaneFitter] Parameters:\n"
			TMP_LOG_VAR(width)
			TMP_LOG_VAR(height)
			// TMP_LOG_VAR(mergeMSETolerance)
			// TMP_LOG_VAR(initMSETolerance)
			// TMP_LOG_VAR(depthSigmaFactor)
			// TMP_LOG_VAR(similarityTh)
			// TMP_LOG_VAR(finalMergeSimilarityTh)
			// TMP_LOG_VAR(simTh_znear)
			// TMP_LOG_VAR(simTh_zfar)
			// TMP_LOG_VAR(simTh_angleMin)
			// TMP_LOG_VAR(simTh_angleMax)
			// TMP_LOG_VAR(depthChangeFactor)
			TMP_LOG_VAR(maxStep)
			TMP_LOG_VAR(minSupport)
			TMP_LOG_VAR(windowWidth)
			TMP_LOG_VAR(windowHeight)
			TMP_LOG_VAR(erodeType)
			TMP_LOG_VAR(doRefine)<<std::endl;
#undef TMP_LOG_VAR
		}

		/************************************************************************/
		/* Protected Class Functions                                            */
		/************************************************************************/
	protected:
		/**
		 *  \brief refine the coarse segmentation
		 *
		 *  \details this function corresponds to Algorithm 4 in our paper; note: plane parameters of each extractedPlanes in the PlaneSeg is NOT updated after this call since the new points added from region grow and points removed from block erosion are not properly reflected in the PlaneSeg
		 */
		void refineDetails(std::vector<std::vector<int>> *pMembership, //pMembership->size()==nPlanes
			const std::vector<int> * const pIdxMap, //if pIdxMap!=0 pMembership->at(i).at(j)=pIdxMap(pixIdx)
			cv::Mat* pSeg, 
			cv::Mat &imgSeg,
			std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>>& planeNormalAndCenter)
		{
			Vector6d tmpVec6d;
			tmpVec6d << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			planeNormalAndCenter.push_back(tmpVec6d);
			if(pMembership==0 && pSeg==0) return;
			std::vector<bool> isValidExtractedPlane; //some planes might be eroded completely
			this->findBlockMembership(isValidExtractedPlane);

			//save the border regions
			std::vector<int> border;
			if(drawCoarseBorder && pSeg) {
				border.resize(rfQueue.size());
				for(int i=0; i<(int)this->rfQueue.size(); ++i) {
					border[i]=rfQueue[i].first;
				}
			}

			this->floodFill();

			//try to merge one last time
			std::vector<PlaneSeg::shared_ptr> oldExtractedPlanes;
			this->extractedPlanes.swap(oldExtractedPlanes);
			PlaneSegMinMSEQueue minQ;
			for(int i=0; i<(int)oldExtractedPlanes.size(); ++i) {
				if(isValidExtractedPlane[i])
					minQ.push(oldExtractedPlanes[i]);
			}
			this->ahCluster(minQ, false);

			//find plane idx maping from oldExtractedPlanes to final extractedPlanes
			std::vector<int> plidmap(oldExtractedPlanes.size(),-1);
			int nFinalPlanes=0;
			for(int i=0; i<(int)oldExtractedPlanes.size(); ++i) {
				const PlaneSeg& op=*oldExtractedPlanes[i];
				if(!isValidExtractedPlane[i]) {
					plidmap[i]=-1;//this plane was eroded
					continue;
				}
				int np_rid;
				if((np_rid=ds->Find(op.rid))==op.rid) {//op is not changed
					if(plidmap[i]<0) //if not updated, otherwise was updated before
						plidmap[i]=nFinalPlanes++;
				} else {//op is merged to np
					const int npid=this->rid2plid[np_rid];
					if(plidmap[npid]<0) //if np's idmap not updated
						plidmap[i]=plidmap[npid]=nFinalPlanes++;
					else
						plidmap[i]=plidmap[npid];
				}
			}
			assert(nFinalPlanes==(int)this->extractedPlanes.size());

			//scan membershipImg
			if(nFinalPlanes>colors.size()) {
				std::vector<cv::Vec3b> tmpColors=pseudocolor(nFinalPlanes-(int)colors.size());
				colors.insert(colors.end(), tmpColors.begin(), tmpColors.end());
			}
			if(pMembership) {
				pMembership->resize(nFinalPlanes, std::vector<int>());
				for(int i=0; i<nFinalPlanes; ++i) {
					pMembership->at(i).reserve(
						(int)(this->extractedPlanes[i]->N*1.2f));
				}
			}
			std::cout<<"提取的平面数量：   "<<extractedPlanes.size()<<std::endl;
			std::vector<cv::Mat> planes(extractedPlanes.size()); 
			for (size_t i = 0; i < extractedPlanes.size(); i++)
			{
				planes.at(i) = cv::Mat(this->height, this->width, CV_8UC1, cv::Scalar(0));
				// std::cout<<"cv::mat "<<i<<" rows: "<<planes.at(i).rows<<" cols: "<<planes.at(i).cols<<std::endl;
			}
			// for (size_t i = 0; i < planes.size(); i++)
			// {
			// 	// cv::imshow("plane extract result", planes.at(i));
			// 	cv::imwrite("reslut" + std::to_string(i) + ".png", planes.at(i));
			// }
			cv::Mat pSeg2(pSeg->size(), CV_8UC1);
			pSeg2.setTo(0);
			static const cv::Vec3b blackColor(0,0,0);
			const int nPixels=this->width*this->height;
			for(int i=0; i<nPixels; ++i) {
				int& plid=membershipImg.at<int>(i);
				if(plid>=0 && plidmap[plid]>=0) {
					plid=plidmap[plid];
					// std::cout << plid << std::endl;
					if(pSeg) 
					{
						pSeg2.at<uchar>(i%640, i - (i%640) * 640) = (uchar)(plid + 1);
						pSeg->at<cv::Vec3b>(i)=this->colors[plid];
						planes.at(plid).at<uchar>(i) = 255;
					}
					if(pMembership) 
						pMembership->at(plid).push_back(pIdxMap?pIdxMap->at(i):i);
				} else {
					if(pSeg) 
						pSeg->at<cv::Vec3b>(i)=blackColor;
				}
			}
			imgSeg = pSeg2.clone();
			// for (size_t i = 0; i < planes.size(); i++)
			// {
			// 	// cv::imshow("plane extract result", planes.at(i));
			// 	cv::imwrite("reslut" + std::to_string(i) + ".png", planes.at(i));
			// }
			// 由像素来计算轮廓
			for (size_t i = 0; i < planes.size(); i++)
			{
				// 先保存一个高斯过滤，后序如果点云质量好再修正
				// cv::Mat Gaussian_filter_image;
      			// cv::GaussianBlur(planes.at(i), Gaussian_filter_image, cv::Size(9, 9), 1, 1);
#ifdef DEBUG_CONTOURS
				// std::cout<<"filted image."<<std::endl;
				// cv::imshow("contours", Gaussian_filter_image);
				// cv::waitKey(500);
#endif
				cv::Mat onePlane = planes.at(i);
				Eigen::Vector3d normal(extractedPlanes.at(i)->normal[0], extractedPlanes.at(i)->normal[1], extractedPlanes.at(i)->normal[2]);
				Eigen::Vector3d center(extractedPlanes.at(i)->center[0], extractedPlanes.at(i)->center[1], extractedPlanes.at(i)->center[2]);
				Vector6d tmpVec6d;
				tmpVec6d << normal[0], normal[1], normal[2], center[0], center[1], center[2];
				planeNormalAndCenter.push_back(tmpVec6d);
				
				// if(extractedPlanes.at(i)->normal[2] > 0.7 || extractedPlanes.at(i)->normal[2] < -0.7)
				// {
				// 	continue;
				// }
				// 需不需要对内部小的轮廓空洞进行填充，需根据点云成像质量决定
				// cv::Mat onePlane = planes.at(i);

				// cv::Mat element3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
				// cv::morphologyEx(onePlane, onePlane, cv::MORPH_CLOSE, element3);
				// std::vector<std::vector<cv::Point>> contours;
				// std::vector<cv::Vec4i> hierarchy;
				// cv::findContours(onePlane, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );
				// cv::drawContours(planeCountous, contours, -1, cv::Scalar(255), 2);

				// cv::drawContours(seg_outputColor, contours, -1, cv::Scalar(0, 255, 0), 1);

// 				std::vector<std::vector<cv::Point>> contours;
//       			std::vector<cv::Vec4i> hierarchy;
// 				// 外圈逆时针，内圈顺时针
// 				// cv::CHAIN_APPROX_NONE 保存所有轮廓点 cv::CHAIN_APPROX_SIMPLE 仅保存拐点， 仅保存拐点的前提是点云质量好，如果不行，需要加高斯平滑
// 				cv::findContours(Gaussian_filter_image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point());
// #ifdef DEBUG_CONTOURS
// 				cv::Mat drawing = cv::Mat::zeros(Gaussian_filter_image.size(), CV_8UC1);
// 				cv::drawContours(drawing, contours, -1, cv::Scalar(255), 2, 8);
// 				std::cout<<"contours image."<<std::endl;
// 				cv::imshow("contours", drawing);
// 				cv::waitKey(500);
// #endif
// 				// 轮廓近似，也可以根据相机成像质量决定，如果近保存拐点，近似是无用的
// 				// std::vector<std::vector<cv::Point>> approx_contours(contours.size());
// 				// std::vector<std::vector<cv::Point>>::iterator approx_contour_iter = approx_contours.begin();
// 				// for (auto & coutour : contours)
// 				// {
// 				// 	cv::approxPolyDP(coutour, *approx_contour_iter, 9, true);
// 				// 	approx_contour_iter++;
// 				// }
				
// 				// 根据像素值计算实际轮廓值  调换轮廓方向，方便加入boost::polygen
// 				bool is_outer = true;
//				Eigen::Vector3f normal(extractedPlanes.at(i)->normal[0], extractedPlanes.at(i)->normal[1], extractedPlanes.at(i)->normal[2]);
// 				Eigen::Vector3f center(extractedPlanes.at(i)->center[0], extractedPlanes.at(i)->center[1], extractedPlanes.at(i)->center[2]);
// 				/*************
// 				 * 两种轮廓计算方式，一种是针对点云质量好，没有nan的点，且不知道相机成像的内参，可以使用点投影到平面的方式
// 				 * 另一种是知道相机内参的情况，
// 				*/
// #ifdef TEST_IDEAL_POINTCLOUD

// 				extractedPlanes.at(i)->contours.resize(contours.size());
// 				std::vector<std::vector<Eigen::Vector3f>>::iterator contours_iter = extractedPlanes.at(i)->contours.begin();
// 				std::vector<Eigen::Vector3f> contour_points;
// 				for (auto & contour : contours)
// 				{
// 					contour_points.clear();
// 					for (auto riter = contour.rbegin(); riter != contour.rend(); riter++)
// 					{
// 						double x, y, z;
// 						points->get(riter->x, riter->y, x, y, z);
// 						Eigen::Vector3f raw_point(x, y, z);
// 						float d = (raw_point - center).dot(normal);
// 	  					contours_iter->emplace_back(raw_point - d * normal);
// 					}
// 					contours_iter++;
// 				}
// #else
// 				//L515: Device Serial No: f0233166 K: [460.2265625, 0.0, 325.44140625, 0.0, 460.25, 236.3984375, 0.0, 0.0, 1.0]
// 				const double kFx = 460.2265625;
// 				const double kFy = 460.25;
// 				const double kCx = 325.44140625;
// 				const double kCy = 236.3984375;
// 				// extractedPlanes.at(i)->contours.resize(contours.size());
// 				// std::vector<std::vector<Eigen::Vector3f>>::iterator contours_iter = extractedPlanes.at(i)->contours.begin();
// 				extractedPlanes.at(i)->contours.clear();
// 				std::vector<Eigen::Vector3f> contour_points;
// 				for (auto & contour : contours)
// 				{
// 					contour_points.clear();
// 					for (auto riter = contour.rbegin(); riter != contour.rend(); riter++)
// 					{
// 						float div_z = normal(0) * (riter->x - kCx) / kFx + normal(1) * (riter->y - kCy) / kFy + normal(2);
// 						float z = (normal.dot(center)) / div_z;      
// 						float x = ((float)(riter->x) - kCx) * z / kFx;
// 						float y = ((float)(riter->y) - kCy) * z / kFy;
// 						Eigen::Vector3f tmppoint(x, y, z);
// 						contour_points.emplace_back(tmppoint);
// 					}
// 					// contours_iter++;
// 					extractedPlanes.at(i)->contours.emplace_back(contour_points);
// #ifdef DEBUG_CONTOURS
// 					// 画出轮廓图，检查一下
// 					std::cout<<"contour polygen: "<<std::endl;
// 					std::vector<float> x, y;
					
// 					for (auto & point_3d : contour_points)
// 					{
// 						std::cout<<point_3d.transpose()<<std::endl;
// 						x.emplace_back(point_3d.x());
// 						y.emplace_back(point_3d.y());
// 					}
// 					x.emplace_back(*x.begin());
// 					y.emplace_back(*y.begin());
// 					plt::plot(x, y);
// 					plt::show();		
// #endif
// 				}
// #endif
			static const cv::Vec3b whiteColor(255,255,255);
			for(int k=0; pSeg && drawCoarseBorder && k<(int)border.size(); ++k) {
				pSeg->at<cv::Vec3b>(border[k])=whiteColor;
			}
			}
			//TODO: refine the plane equation as well after!!
		}

		/**
		 *  \brief find out all valid 4-connect neighbours pixels of pixel (i,j)
		 *
		 *  \param [in] i row index of the center pixel
		 *  \param [in] j column index of the center pixel
		 *  \param [in] H height of the image
		 *  \param [in] W weight of the image
		 *  \param [out] nbs pixel id of all valid neighbours
		 *  \return number of valid neighbours
		 *
		 *  \details invalid 4-connect neighbours means out of image boundary
		 */
		static inline int getValid4Neighbor(
			const int i, const int j,
			const int H, const int W,
			int nbs[4])
		{
			const int id=i*W+j;
			int cnt=0;
			if(j>0) nbs[cnt++]=(id-1);		//left
			if(j<W-1) nbs[cnt++]=(id+1);	//right
			if(i>0) nbs[cnt++]=(id-W);		//up
			if(i<H-1) nbs[cnt++]=(id+W);	//down
			return cnt;
		}

		/**
		 *  \brief find out pixel (pixX, pixY) belongs to which initial block/window
		 *
		 *  \param [in] pixX column index
		 *  \param [in] pixY row index
		 *  \return initial block id, or -1 if not in any block (usually because windowWidth%width!=0 or windowHeight%height!=0)
		 */
		inline int getBlockIdx(const int pixX, const int pixY) const {
			assert(pixX>=0 && pixY>=0 && pixX<this->width && pixY<this->height);
			const int Nw = this->width/this->windowWidth;
			const int Nh = this->height/this->windowHeight;
			const int by = pixY/this->windowHeight;
			const int bx = pixX/this->windowWidth;
			return (by<Nh && bx<Nw)?(by*Nw+bx):-1;
		}

		/**
		 *  \brief region grow from coarse segmentation boundaries
		 *
		 *  \details this function implemented line 14~25 of Algorithm 4 in our paper
		 */
		void floodFill()
		{
			std::vector<float> distMap(this->height*this->width,
				std::numeric_limits<float>::max());

			for(int k=0; k<(int)this->rfQueue.size(); ++k) {
				const int sIdx=rfQueue[k].first;
				const int seedy=sIdx/this->width;
				const int seedx=sIdx-seedy*this->width;
				const int plid=rfQueue[k].second;
				const PlaneSeg& pl = *extractedPlanes[plid];

				int nbs[4]={-1};
				const int Nnbs=this->getValid4Neighbor(seedy,seedx,this->height,this->width,nbs);
				for(int itr=0; itr<Nnbs; ++itr) {
					const int cIdx=nbs[itr];
					int& trail=membershipImg.at<int>(cIdx);
					if(trail<=-6) continue; //visited from 4 neighbors already, skip
					if(trail>=0 && trail==plid) continue; //if visited by the same plane, skip
					const int cy=cIdx/this->width;
					const int cx=cIdx-cy*this->width;
					const int blkid=this->getBlockIdx(cx,cy);
					if(blkid>=0 && this->blkMap[blkid]>=0) continue; //not in "black" block

					double pt[3]={0};
					float cdist=-1;
					if(this->points->get(cy,cx,pt[0],pt[1],pt[2]) &&
						std::pow(cdist=(float)std::abs(pl.signedDist(pt)),2)<9*pl.mse+1e-5) //point-plane distance within 3*std
					{
						if(trail>=0) {
							PlaneSeg& n_pl=*extractedPlanes[trail];
							if(pl.normalSimilarity(n_pl)>=params.T_ang(ParamSet::P_REFINE, pl.center[2])) {//potential for merging
								n_pl.connect(extractedPlanes[plid].get());
							}
						}
						float& old_dist=distMap[cIdx];
						if(cdist<old_dist) {
							trail=plid;
							old_dist=cdist;
							this->rfQueue.push_back(std::pair<int,int>(cIdx,plid));
						} else if(trail<0) {
							trail-=1;
						}
					} else {
						if(trail<0) trail-=1;
					}
				}
			}//for rfQueue
		}

		/**
		 *  \brief erode each segment at initial block/window level
		 *
		 *  \param [in] isValidExtractedPlane coarsely extracted plane i is completely eroded if isValidExtractedPlane(i)==false
		 *
		 *  \details this function implements line 5~13 of Algorithm 4 in our paper, called by refineDetails; FIXME: after this ds is not updated, i.e. is dirtied
		 */
		void findBlockMembership(std::vector<bool>& isValidExtractedPlane) {
			rid2plid.clear();
			for(int plid=0; plid<(int)extractedPlanes.size(); ++plid) {
				rid2plid.insert(std::pair<int,int>(extractedPlanes[plid]->rid,plid));
			}

			const int Nh = this->height/this->windowHeight;
			const int Nw = this->width/this->windowWidth;
			const int NptsPerBlk = this->windowHeight*this->windowWidth;

			membershipImg.create(height, width, CV_32SC1);
			membershipImg.setTo(-1);
			this->blkMap.resize(Nh*Nw);

			isValidExtractedPlane.resize(this->extractedPlanes.size(),false);
			for(int i=0,blkid=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j, ++blkid) {
					const int setid = ds->Find(blkid);
					const int setSize = ds->getSetSize(setid)*NptsPerBlk;
					if(setSize>=minSupport) {//cluster large enough
						int nbs[4]={-1};
						const int nNbs=this->getValid4Neighbor(i,j,Nh,Nw,nbs);
						bool nbClsAllTheSame=true;
						for(int k=0; k<nNbs && this->erodeType!=ERODE_NONE; ++k) {
							if( ds->Find(nbs[k])!=setid &&
								(this->erodeType==ERODE_ALL_BORDER ||
								 ds->getSetSize(nbs[k])*NptsPerBlk>=minSupport) )
							{
								nbClsAllTheSame=false; break;
							}
						}
						const int plid=this->rid2plid[setid];
						if(nbClsAllTheSame) {
							this->blkMap[blkid]=plid;
							const int by=blkid/Nw;
							const int bx=blkid-by*Nw;
							membershipImg(cv::Range(by*windowHeight,(by+1)*windowHeight),
								cv::Range(bx*windowWidth, (bx+1)*windowWidth)).setTo(plid);
							isValidExtractedPlane[plid]=true;
						} else {//erode border region
							this->blkMap[blkid]=-1;
							//this->extractedPlanes[plid]->stats.pop(this->blkStats[blkid]);
						}
					} else {//too small cluster, i.e. "black" cluster
						this->blkMap[blkid]=-1;
					}//if setSize>=blkMinSupport

					//save seed points for floodFill
					if(this->blkMap[blkid]<0) {//current block is not valid
						if(i>0) {
							const int u_blkid=blkid-Nw;
							if(this->blkMap[u_blkid]>=0) {//up blk is in border
								const int u_plid=this->blkMap[u_blkid];
								const int spixidx=(i*this->windowHeight-1)*this->width+j*this->windowWidth;
								for(int k=1; k<this->windowWidth; ++k) {
									this->rfQueue.push_back(std::pair<int,int>(spixidx+k,u_plid));
								}
							}
						}
						if(j>0) {
							const int l_blkid=blkid-1;
							if(this->blkMap[l_blkid]>=0) {//left blk is in border
								const int l_plid=this->blkMap[l_blkid];
								const int spixidx=(i*this->windowHeight)*this->width+j*this->windowWidth-1;
								for(int k=0; k<this->windowHeight-1; ++k) {
									this->rfQueue.push_back(std::pair<int,int>(spixidx+k*this->width,l_plid));
								}
							}
						}
					} else {//current block is still valid
						const int plid=this->blkMap[blkid];
						if(i>0) {
							const int u_blkid=blkid-Nw;
							if(this->blkMap[u_blkid]!=plid) {//up blk is in border
								const int spixidx=(i*this->windowHeight)*this->width+j*this->windowWidth;
								for(int k=0; k<this->windowWidth-1; ++k) {
									this->rfQueue.push_back(std::pair<int,int>(spixidx+k,plid));
								}
							}
						}
						if(j>0) {
							const int l_blkid=blkid-1;
							if(this->blkMap[l_blkid]!=plid) {//left blk is in border
								const int spixidx=(i*this->windowHeight)*this->width+j*this->windowWidth;
								for(int k=1; k<this->windowHeight; ++k) {
									this->rfQueue.push_back(std::pair<int,int>(spixidx+k*this->width,plid));
								}
							}
						}
					}//save seed points for floodFill
				}
			}//for blkik

			////update plane equation
			//for(int i=0; i<(int)this->extractedPlanes.size(); ++i) {
			//	if(isValidExtractedPlane[i]) {
			//		if(this->extractedPlanes[i]->stats.N>=this->minSupport)
			//			this->extractedPlanes[i]->update();
			//	} else {
			//		this->extractedPlanes[i]->nouse=true;
			//	}
			//}
		}

		//called by findMembership and/or plotSegmentImage when doRefine==false
		void findBlockMembership() {
			if(!this->dirtyBlkMbship) return;
			this->dirtyBlkMbship=false;

			rid2plid.clear();
			for(int plid=0; plid<(int)extractedPlanes.size(); ++plid) {
				rid2plid.insert(std::pair<int,int>(extractedPlanes[plid]->rid,plid));
			}

			const int Nh = this->height/this->windowHeight;
			const int Nw = this->width/this->windowWidth;
			const int NptsPerBlk = this->windowHeight*this->windowWidth;

			membershipImg.create(height, width, CV_32SC1);
			membershipImg.setTo(-1);
			this->blkMap.resize(Nh*Nw);

			for(int i=0,blkid=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j, ++blkid) {
					const int setid = ds->Find(blkid);
					const int setSize = ds->getSetSize(setid)*NptsPerBlk;
					if(setSize>=minSupport) {//cluster large enough
						const int plid=this->rid2plid[setid];
						this->blkMap[blkid]=plid;
						const int by=blkid/Nw;
						const int bx=blkid-by*Nw;
						membershipImg(cv::Range(by*windowHeight,(by+1)*windowHeight),
							cv::Range(bx*windowWidth, (bx+1)*windowWidth)).setTo(plid);
					} else {//too small cluster, i.e. "black" cluster
						this->blkMap[blkid]=-1;
					}//if setSize>=blkMinSupport
				}
			}//for blkik
		}

		//called by run when doRefine==false
		void findMembership(std::vector< std::vector<int> >& ret,
			const std::vector<int>* pIdxMap)
		{
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;
			this->findBlockMembership();
			const int cnt = (int)extractedPlanes.size();
			ret.resize(cnt);
			for(int i=0; i<cnt; ++i) ret[i].reserve(extractedPlanes[i]->N);
			for(int i=0,blkid=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j,++blkid) {
					const int plid=this->blkMap[blkid];
					if(plid<0) continue;
					for(int y=i*windowHeight; y<(i+1)*windowHeight; ++y) {
						for(int x=j*windowWidth; x<(j+1)*windowWidth; ++x) {
							const int pixIdx=x+y*width;
							ret[plid].push_back(pIdxMap?pIdxMap->at(pixIdx):pixIdx);
						}
					}
				}
			}
		}

		//called by run when doRefine==false
		void plotSegmentImage(cv::Mat* pSeg, const double supportTh)
		{
			if(pSeg==0) return;
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;
			std::vector<int> ret;
			int cnt=0;

			std::vector<int>* pBlkid2plid;
			if(supportTh==this->minSupport) {
				this->findBlockMembership();
				pBlkid2plid=&(this->blkMap);
				cnt=(int)this->extractedPlanes.size();
			} else { //mainly for DEBUG_CLUSTER since then supportTh!=minSupport
				std::map<int, int> map; //map setid->cnt
				ret.resize(Nh*Nw);
				for(int i=0,blkid=0; i<Nh; ++i) {
					for(int j=0; j<Nw; ++j, ++blkid) {
						const int setid = ds->Find(blkid);
						const int setSize = ds->getSetSize(setid)*windowHeight*windowWidth;
						if(setSize>=supportTh) {
							std::map<int,int>::iterator fitr=map.find(setid);
							if(fitr==map.end()) {//found a new set id
								map.insert(std::pair<int,int>(setid,cnt));
								ret[blkid]=cnt;
								++cnt;
							} else {//found a existing set id
								ret[blkid]=fitr->second;
							}
						} else {//too small cluster, ignore
							ret[blkid]=-1;
						}
					}
				}
				pBlkid2plid=&ret;
			}
			std::vector<int>& blkid2plid=*pBlkid2plid;

			if(cnt>colors.size()) {
				std::vector<cv::Vec3b> tmpColors=pseudocolor(cnt-(int)colors.size());
				colors.insert(colors.end(), tmpColors.begin(), tmpColors.end());
			}
			cv::Mat& seg=*pSeg;
			for(int i=0,blkid=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j,++blkid) {
					const int plid=blkid2plid[blkid];
					if(plid>=0) {
						seg(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(colors[plid]);
					} else {
						seg(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(cv::Vec3b(0,0,0));
					}
				}
			}
		}

#ifdef DEBUG_CLUSTER
		void floodFillColor(const int seedIdx, cv::Mat& seg, const cv::Vec3b& clr) {
			static const int step[8][2]={
				{1,0},{1,1},{0,1},{-1,1},
				{-1,0},{-1,-1},{0,-1},{1,-1}
			};
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;

			std::vector<bool> visited(Nh*Nw, false);
			std::vector<int> idxStack;
			idxStack.reserve(Nh*Nw/10);
			idxStack.push_back(seedIdx);
			visited[seedIdx]=true;
			const int sy=seedIdx/Nw;
			const int sx=seedIdx-sy*Nw;
			seg(cv::Range(sy*windowHeight,(sy+1)*windowHeight),
				cv::Range(sx*windowWidth, (sx+1)*windowWidth)).setTo(clr);

			const int clsId=ds->Find(seedIdx);
			while(!idxStack.empty()) {
				const int sIdx=idxStack.back();
				idxStack.pop_back();
				const int seedy=sIdx/Nw;
				const int seedx=sIdx-seedy*Nw;
				for(int i=0; i<8; ++i) {
					const int cx=seedx+step[i][0];
					const int cy=seedy+step[i][1];
					if(0<=cx && cx<Nw && 0<=cy && cy<Nh) {
						const int cIdx=cx+cy*Nw;
						if(visited[cIdx]) continue; //if visited, skip
						visited[cIdx]=true;
						if(clsId==ds->Find(cIdx)) {//if same plane, move
							idxStack.push_back(cIdx);
							seg(cv::Range(cy*windowHeight,(cy+1)*windowHeight),
							cv::Range(cx*windowWidth, (cx+1)*windowWidth)).setTo(clr);
						}
					}
				}
			}//while
		}
#endif

#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
		cv::Mat dInit;
		cv::Mat dSeg;
		cv::Mat dGraph;
#endif

		/**
		 *  \brief initialize a graph from pointsIn
		 *
		 *  \param [in/out] minQ a min MSE queue of PlaneSegs
		 *
		 *  \details this function implements Algorithm 2 in our paper
		 */
		void initGraph(PlaneSegMinMSEQueue& minQ) {
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;

			//1. init nodes
			std::vector<PlaneSeg::Ptr> G(Nh*Nw,0);
			//this->blkStats.resize(Nh*Nw);

#ifdef DEBUG_INIT
			dInit.create(this->height, this->width, CV_8UC3);
			dInit.setTo(cv::Vec3b(0,0,0));
#endif
			for(int i=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j) {
					PlaneSeg::shared_ptr p( new PlaneSeg(
						*this->points, (i*Nw+j),
						i*this->windowHeight, j*this->windowWidth,
						this->width, this->height,
						this->windowWidth, this->windowHeight,
						this->params) );
					if(p->mse<params.T_mse(ParamSet::P_INIT, p->center[2])
						&& !p->nouse)
					{
						G[i*Nw+j]=p.get();
						minQ.push(p);
						//this->blkStats[i*Nw+j]=p->stats;
#ifdef DEBUG_INIT
						//const uchar cl=uchar(p->mse*255/dynThresh);
						//const cv::Vec3b clr(cl,cl,cl);
						dInit(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(p->getColor(true));

						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,0,1);
						cv::circle(dInit, cv::Point(cx,cy), 1, blackColor, 2);
#endif
					} else {
						G[i*Nw+j]=0;
#ifdef DEBUG_INIT
						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,0,1);
						static const cv::Vec3b whiteColor(255,255,255);
						dInit(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(whiteColor);

						switch(p->type) {
						case PlaneSeg::TYPE_NORMAL: //draw a big dot
							{
								static const cv::Scalar yellow(255,0,0,1);
								cv::circle(dInit, cv::Point(cx,cy), 3, yellow, 4);
								break;
							}
						case PlaneSeg::TYPE_MISSING_DATA: //draw an 'o'
							{
								static const cv::Scalar black(0,0,0,1);
								cv::circle(dInit, cv::Point(cx,cy), 3, black, 2);
								break;
							}
						case PlaneSeg::TYPE_DEPTH_DISCONTINUE: //draw an 'x'
							{
								static const cv::Scalar red(255,0,0,1);
								static const int len=4;
								cv::line(dInit, cv::Point(cx-len, cy-len), cv::Point(cx+len,cy+len), red, 2);
								cv::line(dInit, cv::Point(cx+len, cy-len), cv::Point(cx-len,cy+len), red, 2);
								break;
							}
						}
#endif
					}
				}
			}
#ifdef DEBUG_INIT
			//cv::applyColorMap(dInit, dInit,  cv::COLORMAP_COOL);
#endif
#ifdef DEBUG_CALC
			int nEdge=0;
			this->numEdges.clear();
			this->numNodes.clear();
#endif

			//2. init edges
			//first pass, connect neighbors from row direction
			for(int i=0; i<Nh; ++i) {
				for(int j=1; j<Nw; j+=2) {
					const int cidx=i*Nw+j;
					if(G[cidx-1]==0) { --j; continue; }
					if(G[cidx]==0) continue;
					if(j<Nw-1 && G[cidx+1]==0) { ++j; continue; }

					const double similarityTh=params.T_ang(ParamSet::P_INIT, G[cidx]->center[2]);
					if((j<Nw-1 && G[cidx-1]->normalSimilarity(*G[cidx+1])>=similarityTh) ||
						(j==Nw-1 && G[cidx]->normalSimilarity(*G[cidx-1])>=similarityTh)) {
							G[cidx]->connect(G[cidx-1]);
							if(j<Nw-1) G[cidx]->connect(G[cidx+1]);
#ifdef DEBUG_INIT
						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						const int rx=(j+1)*windowWidth+0.5*(windowWidth-1);
						const int lx=(j-1)*windowWidth+0.5*(windowWidth-1);
						static const cv::Scalar blackColor(0,0,0,1);
						cv::line(dInit, cv::Point(cx,cy), cv::Point(lx,cy),blackColor);
						if(j<Nw-1) cv::line(dInit, cv::Point(cx,cy), cv::Point(rx,cy),blackColor);
#endif
#ifdef DEBUG_CALC
						nEdge+=(j<Nw-1)?4:2;
#endif
					} else {//otherwise current block is in edge region
						--j;
					}
				}
			}
			//second pass, connect neighbors from column direction
			for(int j=0; j<Nw; ++j) {
				for(int i=1; i<Nh; i+=2) {
					const int cidx=i*Nw+j;
					if(G[cidx-Nw]==0) { --i; continue; }
					if(G[cidx]==0) continue;
					if(i<Nh-1 && G[cidx+Nw]==0) { ++i; continue; }

					const double similarityTh=params.T_ang(ParamSet::P_INIT, G[cidx]->center[2]);
					if((i<Nh-1 && G[cidx-Nw]->normalSimilarity(*G[cidx+Nw])>=similarityTh) ||
						(i==Nh-1 && G[cidx]->normalSimilarity(*G[cidx-Nw])>=similarityTh)) {
							G[cidx]->connect(G[cidx-Nw]);
							if(i<Nh-1) G[cidx]->connect(G[cidx+Nw]);
#ifdef DEBUG_INIT
						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						const int uy=(i-1)*windowHeight+0.5*(windowHeight-1);
						const int dy=(i+1)*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,0,1);
						cv::line(dInit, cv::Point(cx,cy), cv::Point(cx,uy),blackColor);
						if(i<Nh-1) cv::line(dInit, cv::Point(cx,cy), cv::Point(cx,dy),blackColor);
#endif
#ifdef DEBUG_CALC
						nEdge+=(i<Nh-1)?4:2;
#endif
					} else {
						--i;
					}
				}
			}
#ifdef DEBUG_INIT
			static int cnt=0;
			cv::namedWindow("debug initGraph");
			cv::cvtColor(dInit,dInit,CV_RGB2BGR);
			cv::imshow("debug initGraph", dInit);
			std::stringstream ss;
			ss<<saveDir<<"/output/db_init"<<std::setw(5)<<std::setfill('0')<<cnt++<<".png";
			cv::imwrite(ss.str(), dInit);
#endif
#ifdef DEBUG_CALC
			this->numNodes.push_back(minQ.size());
			this->numEdges.push_back(nEdge);
			this->maxIndvidualNodeDegree=4;
			this->mseNodeDegree.clear();
#endif
		}

		/**
		 *  \brief main clustering step
		 *
		 *  \param [in] minQ a min MSE queue of PlaneSegs
		 *  \param [in] debug whether to collect some statistics when compiled with DEBUG_CALC
		 *  \return number of cluster steps
		 *
		 *  \details this function implements the Algorithm 3 in our paper
		 */
		int ahCluster(PlaneSegMinMSEQueue& minQ, bool debug=true) {
#if !defined(DEBUG_INIT) && defined(DEBUG_CLUSTER)
			dInit.create(this->height, this->width, CV_8UC3);
#endif
#ifdef DEBUG_CLUSTER
			const int Nw = this->width/this->windowWidth;
			int dSegCnt=0;
			//dSeg.create(this->height, this->width, CV_8UC3);
			dInit.copyTo(dSeg);
			dGraph.create(this->height, this->width, CV_8UC3);
			{
				//this->plotSegmentImage(&dSeg, 0);
				std::stringstream ss;
				ss<<saveDir<<"/output/cluster_"<<std::setw(5)<<std::setfill('0')<<dSegCnt<<".png";
				cv::imwrite(ss.str(), dSeg);
				cv::namedWindow("debug ahCluster");
				cv::imshow("debug ahCluster", dSeg);
				cv::namedWindow("debug Graph");
			}
#endif
			int step=0;
			while(!minQ.empty() && step<=maxStep) {
				PlaneSeg::shared_ptr p=minQ.top();
				minQ.pop();
				if(p->nouse) {
					assert(p->nbs.size()<=0);
					continue;
				}
#ifdef DEBUG_CALC
				this->maxIndvidualNodeDegree=std::max(this->maxIndvidualNodeDegree,(int)p->nbs.size());
				this->mseNodeDegree.push_back((int)p->nbs.size());
#endif
#ifdef DEBUG_CLUSTER
				int cx, cy;
				{
					dSeg.copyTo(dGraph);
					const int blkid=p->rid;
					this->floodFillColor(blkid, dGraph, p->getColor(false));
					const int i=blkid/Nw;
					const int j=blkid-i*Nw;
					cx=j*windowWidth+0.5*(windowWidth-1);
					cy=i*windowHeight+0.5*(windowHeight-1);
					static const cv::Scalar blackColor(0,0,255,1);
					cv::circle(dGraph, cv::Point(cx,cy),3,blackColor,2);
				}
#endif
				PlaneSeg::shared_ptr cand_merge;
				PlaneSeg::Ptr cand_nb(0);
				PlaneSeg::NbSet::iterator itr=p->nbs.begin();
				for(; itr!=p->nbs.end();itr++) {//test merge with all nbs, pick the one with min mse
					PlaneSeg::Ptr nb=(*itr);
#ifdef DEBUG_CLUSTER
					{
						const int n_blkid=nb->rid;
						this->floodFillColor(n_blkid, dGraph, nb->getColor(false));
					}
#endif
					//TODO: should we use dynamic similarityTh here?
					//const double similarityTh=ahc::depthDependNormalDeviationTh(p->center[2],500,4000,M_PI*15/180.0,M_PI/2);
					if(p->normalSimilarity(*nb) < params.T_ang(ParamSet::P_MERGING, p->center[2])) continue;
					PlaneSeg::shared_ptr merge(new PlaneSeg(*p, *nb));
					if(cand_merge==0 || cand_merge->mse>merge->mse ||
						(cand_merge->mse==merge->mse && cand_merge->N<merge->mse))
					{
						cand_merge=merge;
						cand_nb=nb;
					}
				}//for nbs
#ifdef DEBUG_CLUSTER
				itr=p->nbs.begin();
				for(; debug && itr!=p->nbs.end();itr++) {
					PlaneSeg::Ptr nb=(*itr);
					const int n_blkid=nb->rid;
					const int i=n_blkid/Nw;
					const int j=n_blkid-i*Nw;
					const int mx=j*windowWidth+0.5*(windowWidth-1);
					const int my=i*windowHeight+0.5*(windowHeight-1);
					static const cv::Scalar blackColor(0,0,255,1);
					cv::circle(dGraph, cv::Point(mx,my),3,blackColor,1);
					cv::line(dGraph, cv::Point(cx,cy), cv::Point(mx,my), blackColor,1);
				}//for nbs
#endif
				//TODO: maybe a better merge condition? such as adaptive threshold on MSE like Falzenszwalb's method
				if(cand_merge!=0 && cand_merge->mse<params.T_mse(
					ParamSet::P_MERGING, cand_merge->center[2]))
				{//merge and add back to minQ
#ifdef DEBUG_CLUSTER
					{
						const int n_blkid=cand_nb->rid;
						const int i=n_blkid/Nw;
						const int j=n_blkid-i*Nw;
						const int mx=j*windowWidth+0.5*(windowWidth-1);
						const int my=i*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,255,1);
						cv::circle(dGraph, cv::Point(mx,my),2,blackColor,2);
						cv::line(dGraph, cv::Point(cx,cy), cv::Point(mx,my), blackColor,2);
						std::stringstream ss;
						ss<<saveDir<<"/output/dGraph_"<<std::setw(5)<<std::setfill('0')<<++dSegCnt<<".png";
						cv::imwrite(ss.str(), dGraph);
						cv::imshow("debug Graph", dGraph);
					}
#endif
#ifdef DEBUG_CALC
					const int nEdge_p=(int)p->nbs.size()*2;
					const int nEdge_nb=(int)cand_nb->nbs.size()*2;
#endif
					minQ.push(cand_merge);
					cand_merge->mergeNbsFrom(*p, *cand_nb, *this->ds);
#ifdef DEBUG_CALC
					if(debug) {//don't do this when merging one last time
						this->numNodes.push_back(this->numNodes.back()-1);
						this->numEdges.push_back(this->numEdges.back() + cand_merge->nbs.size()*2
							- nEdge_p - nEdge_nb + 2);
					}
#endif
#ifdef DEBUG_CLUSTER
					{
						floodFillColor(cand_merge->rid, dSeg, cand_merge->getColor(false));
						std::stringstream ss;
						ss<<saveDir<<"/output/cluster_"<<std::setw(5)<<std::setfill('0')<<dSegCnt<<".png";
						cv::imwrite(ss.str(), dSeg);
						cv::imshow("debug ahCluster", dSeg);
						cv::waitKey(5);
					}
#endif
				} else {//do not merge, but extract p
					if(p->N>=this->minSupport) {
						this->extractedPlanes.push_back(p);
#ifdef DEBUG_CLUSTER
						const int blkid=p->rid;
						const int i=blkid/Nw;
						const int j=blkid-i*Nw;
						const int ex=j*windowWidth+0.5*(windowWidth-1);
						const int ey=i*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,0,1);
						const int len=3;
						{
							cv::line(dGraph, cv::Point(ex-len,ey), cv::Point(ex+len,ey), blackColor, 2);
							cv::line(dGraph, cv::Point(ex,ey-len), cv::Point(ex,ey+len), blackColor, 2);
							std::stringstream ss;
							ss<<saveDir<<"/output/dGraph_"<<std::setw(5)<<std::setfill('0')<<++dSegCnt<<".png";
							cv::imwrite(ss.str(), dGraph);
							cv::imshow("debug Graph", dGraph);
						}
						{
							floodFillColor(p->rid, dSeg, p->getColor(false));
							cv::line(dSeg, cv::Point(ex-len,ey), cv::Point(ex+len,ey), blackColor, 2);
							cv::line(dSeg, cv::Point(ex,ey-len), cv::Point(ex,ey+len), blackColor, 2);
							std::stringstream ss;
							ss<<saveDir<<"/output/cluster_"<<std::setw(5)<<std::setfill('0')<<dSegCnt<<".png";
							cv::imwrite(ss.str(), dSeg);
							cv::imshow("debug ahCluster", dSeg);
							cv::waitKey(5);
						}
#endif
					}
#ifdef DEBUG_CLUSTER
					else {
						{
							floodFillColor(p->rid, dGraph, cv::Vec3b(0,0,0));
							std::stringstream ss;
							ss<<saveDir<<"/output/dGraph_"<<std::setw(5)<<std::setfill('0')<<++dSegCnt<<".png";
							cv::imwrite(ss.str(), dGraph);
							cv::imshow("debug Graph", dGraph);
						}
						{
							floodFillColor(p->rid, dSeg, cv::Vec3b(0,0,0));
							std::stringstream ss;
							ss<<saveDir<<"/output/cluster_"<<std::setw(5)<<std::setfill('0')<<dSegCnt<<".png";
							cv::imwrite(ss.str(), dSeg);
							cv::imshow("debug ahCluster", dSeg);
							cv::waitKey(5);
						}
					}
#endif
#ifdef DEBUG_CALC
					if(debug) {
						this->numNodes.push_back(this->numNodes.back()-1);
						this->numEdges.push_back(this->numEdges.back()-(int)p->nbs.size()*2);
					}
#endif
					p->disconnectAllNbs();
				}
				++step;
			}//end while minQ
			while(!minQ.empty()) {//just check if any remaining PlaneSeg if maxstep reached
				const PlaneSeg::shared_ptr p=minQ.top();
				minQ.pop();
				if(p->N>=this->minSupport) {
					this->extractedPlanes.push_back(p);
				}
				p->disconnectAllNbs();
			}
#ifdef DEBUG_CLUSTER
			{
				std::stringstream ss;
				ss<<saveDir<<"/output/cluster_"<<std::setw(5)<<std::setfill('0')<<(++dSegCnt)<<".png";
				cv::imwrite(ss.str(), dSeg);
				exit(-1);
			}
#endif
			static PlaneSegSizeCmp sizecmp;
			std::sort(this->extractedPlanes.begin(),
				this->extractedPlanes.end(),
				sizecmp);
			return step;
		}
	};//end of PlaneFitter
}//end of namespace ahc
