#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <cmath>
#include <chrono>
#include "Eva.h"
#include "omp.h"
#include <unsupported/Eigen/MatrixFunctions>
using namespace Eigen;

double calcRotError(Eigen::Matrix3d& est, Eigen::Matrix3d& gt) {
	double tr = (est.transpose() * gt).trace();
	return acos(min(max((tr - 1.0) / 2.0, -1.0), 1.0)) * 180.0 / M_PI;
}

double calcTransError(Eigen::Vector3d& est, Eigen::Vector3d& gt) {
	Eigen::Vector3d t = est - gt;
	return sqrt(t.dot(t)) * 100;
}

bool evalEstimate(Eigen::Matrix4d est, Eigen::Matrix4d gt, double re_thresh, double te_thresh, double& RE, double& TE) {
	Eigen::Matrix3d rot_est, rot_gt;
	Eigen::Vector3d trans_est, trans_gt;
	rot_est = est.topLeftCorner(3, 3);
	rot_gt = gt.topLeftCorner(3, 3);
	trans_est = est.block(0, 3, 3, 1);
	trans_gt = gt.block(0, 3, 3, 1);

	RE = calcRotError(rot_est, rot_gt);
	TE = calcTransError(trans_est, trans_gt);
	if (0 <= RE && RE <= re_thresh && 0 <= TE && TE <= te_thresh)
	{
		return true;
	}
	return false;
}

void postRefinement(Matf3D& src, Matf3D& dst, 
	Eigen::Matrix4f& initial,
	float inlier_thresh, int iterations, 
	float norm_thresh, Mati1D bestinliers_idx3,
	const Matf3D& src_result,
	const Matf3D& des_result,
	float lambda) {
    const int N_original = src.cols();
    const int N_subset = src_result.cols();

    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr des_kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_des(new pcl::PointCloud<pcl::PointXYZ>());
	cloud_des->resize(des_result.cols());
	for(int i=0; i<des_result.cols(); ++i) {
		cloud_des->points[i].x = des_result(0,i);
		cloud_des->points[i].y = des_result(1,i);
		cloud_des->points[i].z = des_result(2,i);
	}
	des_kdtree->setInputCloud(cloud_des); 

    Matf3D static_src = src;
    Matf3D static_dst = dst;

    Eigen::Matrix4f cur_trans = initial;
    for(int iter = 0; iter < iterations; ++iter) {

        Matf3D transformed_subset = (cur_trans.block<3,3>(0,0) * src_result).colwise() 
                                  + cur_trans.block<3,1>(0,3);
        
        Matf3D dynamic_dst(3, N_subset);
		#pragma omp parallel for
        for(int i=0; i<N_subset; ++i){
            pcl::PointXYZ query;
            query.x = transformed_subset(0, i);
            query.y = transformed_subset(1, i);
            query.z = transformed_subset(2, i);
            
            std::vector<int> indices(1);
            std::vector<float> dists(1);
            if(des_kdtree->nearestKSearch(query, 1, indices, dists) > 0){
                dynamic_dst.col(i) << 
                    cloud_des->points[indices[0]].x,
                    cloud_des->points[indices[0]].y,
                    cloud_des->points[indices[0]].z;
            }
        }

        Matf3D combined_src(3, N_original + N_subset);
        Matf3D combined_dst(3, N_original + N_subset);
        combined_src << static_src, src_result;
        combined_dst << static_dst, dynamic_dst;

        Matf3D transformed_all = (cur_trans.block<3,3>(0,0) * combined_src).colwise() 
                               + cur_trans.block<3,1>(0,3);
        Matf1D residuals = (transformed_all - combined_dst).colwise().norm();

        Eigen::MatrixXf weights(1, N_original + N_subset);
        
        for(int j=0; j<N_original; ++j) {
            float w = exp(-0.5f * pow(residuals(j)/inlier_thresh, 2));
            weights(j) = w * lambda / N_original;
        }
        
        for(int j=N_original; j<weights.cols(); ++j) {
            float w = exp(-0.5f * pow(residuals(j)/inlier_thresh, 2));
			weights(j) = w / N_subset;
        }

        Eigen::Matrix4f delta_trans = weightedRigidTrans(combined_src, combined_dst, weights);
        
		if((delta_trans - cur_trans).norm() < norm_thresh){
			cur_trans = delta_trans;
			break;
		}
		cur_trans = delta_trans;
    }
    initial = cur_trans;
}