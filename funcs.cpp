#include <stdio.h>
#include <vector>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#define BOOST_TYPEOF_EMULATION
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "Eva.h"

float computeCloudResolution(PointCloudPtr& cloud) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    int search_num = std::floor(static_cast<int>(cloud->size()) / 20) * 10;
    int pt_num = cloud->size();

    std::vector<float> distances(search_num);
    for (int i = 0; i < search_num; ++i) {
        int idx = std::rand() % pt_num;
        std::vector<int> idx_nknsearch(2);
        std::vector<float> sqdist_nknsearch(2);
        tree->nearestKSearch(cloud->points[idx], 2, idx_nknsearch, sqdist_nknsearch);
        distances[i] = sqdist_nknsearch[1];
    }
    std::sort(distances.begin(), distances.end());
    int id_mid = (int)(search_num - 1) / 2;
    return std::sqrt(distances[id_mid]);
}

int voxelDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& new_cloud, float leaf_size) {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*new_cloud);
    return 0;
}

Matf6D ransacOnePoint(Matf6D& x, Matf6D& x_normal, float t, Mati1D& bestinliers_idx, float inlier_judge_thresh, float sub_th) {
    int s = 1;
    int max_trials = 10000;
    int npts = x.cols();
    float p = 0.99;
    int trialcount = 0;
    int bestscore = 0;
    Matf6D bestinliers;
    float N = 100;          
    float t2 = 2.0 * t;
    
    float eps = std::numeric_limits<float>::epsilon();
    std::vector<Mati1D> myVector;
    std::vector<int> score(npts, 0);
    std::vector<int> inds;
    while (N > trialcount) {
        bool update_pro = false;
        bool earlyStop = false;
        
		Mati1D matrix(1, npts);
		for (int i = 0; i < npts; ++i) {
			matrix(0, i) = i;
		}
        int ind = std::rand() % npts;
        inds.push_back(ind);
       
        Eigen::Matrix<float, 6, 1> seedpoint = x.col(ind);
        Matf6D lineset = x.colwise() - seedpoint;

        Matf1D D1 = lineset.topRows(3).colwise().norm();
        Matf1D D2 = lineset.bottomRows(3).colwise().norm();
        Matf1D len = (D1 - D2).array().abs();
        
        Mati1D flag = (len.array() < t2).cast<int>();
        Mati1D inlier_column = getNonZeroColIndices(flag);
        Matf6D inliers = x(Eigen::all, inlier_column);
		Mati1D inliers_idx = matrix(Eigen::all, inlier_column);

		Matf6D x_normal2 = x_normal(Eigen::all, inlier_column);
        Matf3D inliers_Source_points = inliers.topRows(3);
		Matf3D inliers_Target_points = inliers.bottomRows(3);
        Matf3D sourceNormals = x_normal2.topRows(3);
		Matf3D targetNormals = x_normal2.bottomRows(3);
        Eigen::Vector3f specificSourcePoint = x.block<3, 1>(0, ind);
        Eigen::Vector3f specificTargetPoint = x.block<3, 1>(3, ind);
        Matf1D sourceNorms = sourceNormals.colwise().norm(); 
        Matf1D targetNorms = targetNormals.colwise().norm(); 
        Matf1D distances_Target = Matf1D::Zero(inliers_Target_points.cols());
		Matf1D distances_Source = Matf1D::Zero(inliers_Source_points.cols());
        for (int i = 0; i < inliers_Source_points.cols(); ++i) {
            Eigen::Vector3f point_diff = inliers_Source_points.col(i) - specificSourcePoint;
            Eigen::Vector3f cross_result = point_diff.cross(sourceNormals.col(i));
            distances_Source(0, i) = cross_result.norm() / sourceNorms(0, i);

            Eigen::Vector3f point_diff2 = inliers_Target_points.col(i) - specificTargetPoint;
            Eigen::Vector3f cross_result2 = point_diff2.cross(targetNormals.col(i));
            distances_Target(0, i) = cross_result2.norm() / targetNorms(0, i);
        }
        
        Matf1D len_normal1 = (distances_Target - distances_Source).array().abs();
		
        Eigen::Vector3f specificSourceNormal = x_normal.block<3, 1>(0, ind);
		Eigen::Vector3f specificTargetNormal = x_normal.block<3, 1>(3, ind);
		float specificSourceNormalNorm = specificSourceNormal.norm();
		float specificTargetNormalNorm = specificTargetNormal.norm();
        Matf1D distances_Source2 = ((inliers_Source_points.colwise() - specificSourcePoint).colwise().cross(specificSourceNormal)).colwise().norm() / specificSourceNormalNorm;
        Matf1D distances_Target2 = ((inliers_Target_points.colwise() - specificTargetPoint).colwise().cross(specificTargetNormal)).colwise().norm() / specificTargetNormalNorm;
        Matf1D len_normal2 = (distances_Target2 - distances_Source2).array().abs();
        Matf1D len_normal = len_normal1.array().max(len_normal2.array());
        Mati1D flag_normal = (len_normal.array() < 2.0 * inlier_judge_thresh).cast<int>();
        
        Mati1D inlier_column_normal = getNonZeroColIndices(flag_normal);
        Matf6D inliers_normal = inliers(Eigen::all, inlier_column_normal);
		inliers = inliers_normal;
		Mati1D inliers_idx_normal = inliers_idx(Eigen::all, inlier_column_normal);
		inliers_idx = inliers_idx_normal;
        
        int s1 = inliers.cols();
        int inlier_size = 0;
        for (int i = 1; i <= 50; ++i) {
            Matf3D src = inliers.topRows(3);
            Matf3D dst = inliers.bottomRows(3);

            Eigen::MatrixXf src_dist_matrix(src.cols(), src.cols());
            Eigen::MatrixXf dst_dist_matrix(dst.cols(), dst.cols());
            computeDistMatrix(src, src_dist_matrix);
            computeDistMatrix(dst, dst_dist_matrix);
            Eigen::MatrixXf Z = (src_dist_matrix - dst_dist_matrix).array().abs();
            Eigen::MatrixXi F = (Z.array() < t2).cast<int>();
            inlier_size = std::ceil(std::sqrt(F.sum()));
            
            Mati1D F_colwise_sum = F.colwise().sum();
            std::vector<int> sorted_column_indices_total;
            sortRowDescending(F_colwise_sum, sorted_column_indices_total);

            std::vector<int> sorted_column_indices_inlier(sorted_column_indices_total.begin(), 
                sorted_column_indices_total.begin() + inlier_size);
            Matf6D selected_inliers = inliers(Eigen::all, sorted_column_indices_inlier);
            inliers = selected_inliers;
			Mati1D selected_inliers_idx = inliers_idx(Eigen::all, sorted_column_indices_inlier);
			inliers_idx = selected_inliers_idx;

            if ((s1 - inlier_size) < 5) {
                break;
            }
            s1 = inlier_size;
        }
        
        Matf3D src_1 = inliers.topRows(3);
        Matf3D dst_1 = inliers.bottomRows(3);
        int max_vec_sign;
        Eigen::Matrix4f estimated_transform = estimateRigidSVD(src_1, dst_1, max_vec_sign);

        if( (max_vec_sign == 0)){
            continue;
        }

        myVector.push_back(inliers_idx);

        if(inlier_size > sub_th * bestscore){
            for (int i = 0; i < inliers_idx.cols(); ++i) {
                score[inliers_idx(0, i)] += 1;
            }
        }
    
        if (inlier_size > bestscore) {
            update_pro = true;
            bestscore = inlier_size;
            bestinliers = inliers;
			bestinliers_idx = inliers_idx;
            float fracinliers = static_cast<float>(inlier_size) / npts;
            float pNoOutliers = 1 - std::pow(fracinliers, s);
            pNoOutliers = std::max(eps, pNoOutliers);
            pNoOutliers = std::min(1 - eps, pNoOutliers);
            N = log(1-p)/log(pNoOutliers);
		}
        ++trialcount; 
        if (trialcount > max_trials) {
            break;
        }
    }
    std::vector<float> myScore1(myVector.size(), 0.0f);
    std::vector<int> mySize(myVector.size(), 0);
    for(int i = 0; i < myVector.size(); i++){
        float sum_pro = 0;
        for(int j = 0; j < myVector[i].cols(); j++){
            sum_pro += score[myVector[i](0, j)];
        }
        myScore1[i] = sum_pro;
        mySize[i] = myVector[i].cols();
    }

    auto maxIt1 = std::max_element(myScore1.begin(), myScore1.end());
    size_t maxIndex1 = std::distance(myScore1.begin(), maxIt1);
    bestinliers_idx = myVector[maxIndex1];
    return bestinliers;
}

Matrix4f estimateRigidSVD(const Matf3D& A, const Matf3D& B, int& max_vec_sign) {
    Vector3f lc = A.rowwise().mean();
    Vector3f rc = B.rowwise().mean();

    Matf3D A_centered = A.colwise() - lc;
    Matf3D B_centered = B.colwise() - rc;

    Eigen::Matrix3f M = A_centered * B_centered.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    Eigen::Matrix3f R = V * U.transpose();

    max_vec_sign = (R.determinant() < 0) ? 0 : 1;
    if (R.determinant() < 0) {
        V.col(2) = -V.col(2);
        R = V * U.transpose();
    }

    Vector3f t = rc - R * lc;

    Matrix4f T = Matrix4f::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;

    return T;
}

Eigen::Matrix4f weightedRigidTrans(const Matf3D& A, const Matf3D& B, Eigen::MatrixXf& weights) {
    float sw = weights.sum();
    if (sw < std::numeric_limits<float>::epsilon()) {
        weights = Eigen::MatrixXf::Ones(1, A.cols());
        sw = weights.sum();
    }

    Eigen::MatrixXf w = weights / sw;
    Eigen::Matrix<float, 3, 1> lc = A * w.transpose();
    Eigen::Matrix<float, 3, 1> rc = B * w.transpose();
    Eigen::MatrixXf w2 = w.cwiseSqrt();
    Eigen::MatrixXf w2_repmat(A.rows(), A.cols());
    for (int i = 0; i < w2_repmat.rows(); ++i) {
        w2_repmat.row(i) = w2;
    }

    Eigen::MatrixXf left = (A.colwise() - lc).cwiseProduct(w2_repmat);
    Eigen::MatrixXf right = (B.colwise() - rc).cwiseProduct(w2_repmat);
    Eigen::MatrixXf M = left * right.transpose();

    float Sxx = M(0,0); float Syx = M(1,0); float Szx = M(2,0);
    float Sxy = M(0,1); float Syy = M(1,1); float Szy = M(2,1);
    float Sxz = M(0,2); float Syz = M(1,2); float Szz = M(2,2);
    Eigen::Matrix4f N = Eigen::Matrix4f::Identity();
    N << Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
        Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
        Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
        Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz;

    Eigen::EigenSolver<Eigen::Matrix4f> es(N);
	Eigen::Vector4f evalue = es.eigenvalues().real();
	Eigen::Matrix4f evector = es.eigenvectors().real();
    
    int maxRow = 0, maxCol = 0; 
    evalue.maxCoeff(&maxRow, &maxCol);
    Eigen::Vector4f q = evector.col(maxRow);

    q.cwiseAbs().maxCoeff(&maxRow, &maxCol); 
    float max_vec = q(maxRow, 0);
    if (max_vec < 0) {
        q = q * (-1);
    }
    q.normalize();
    float q0 = q(0);
    float qx = q(1);
    float qy = q(2);
    float qz = q(3);
    Eigen::Vector3f v = q.tail(3);

    Eigen::Matrix3f Z = Eigen::Matrix3f::Identity();
    Z << q0, -qz, qy,
        qz, q0, -qx,
        -qy, qx, q0;

    Eigen::Matrix3f R = v * v.transpose() + Z * Z;
    Eigen::Vector3f t = rc - R * lc;

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

void updateProbabilities(int order, int iteration, float classificationTableParam, int& outlierNumber, double threshold_1, std::vector<double>& inliers_1, std::vector<double>& inliers_no_1, std::vector<double>& inliers_2, std::vector<double>& inliers_no_2, std::vector<double>& inliers_k_1, std::vector<double>& inliers_no_k_1, std::vector<double>& correspondences_pro, std::vector<double> correspondences_pro1, std::vector<int>& inliers_new, bool& earlyStop, double& sum)
{
	outlierNumber = 0;
	if (order == 1 || (order == 2 && iteration == 1) || (order == 3 && iteration == 1)) {
		for (int i = 0; i < correspondences_pro.size(); ++i) {
			float PHIn_, PHIn__, PHIn_1, PHIn_2, PHIn__1, PHIn__2;
			if (inliers_new[i] == 1) {
				PHIn_ = 1.0 * classificationTableParam * inliers_k_1[i] + 0.2 * (1.0 - classificationTableParam) * inliers_no_k_1[i];
				PHIn__ = 0.0 * classificationTableParam * inliers_k_1[i] + 0.8 * (1.0 - classificationTableParam) * inliers_no_k_1[i];
			}
			else {
				PHIn_ = 1.0 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.0 * classificationTableParam * inliers_no_k_1[i];
				PHIn__ = 0.0 * (1.0 - classificationTableParam) * inliers_k_1[i] + 1.0 * classificationTableParam * inliers_no_k_1[i];
			}
			inliers_k_1[i] = PHIn_;
			inliers_no_k_1[i] = PHIn__;
			if(inliers_k_1[i] + inliers_no_k_1[i] == 0){
				correspondences_pro[i] = 0;
			}else{
				correspondences_pro[i] = inliers_k_1[i] / (inliers_k_1[i] + inliers_no_k_1[i]);
			}
		}
		if (order == 2){
			inliers_1 = inliers_k_1;
			inliers_no_1 = inliers_no_k_1;
			inliers_2 = inliers_k_1;
			inliers_no_2 = inliers_no_k_1;
		}
	}
	else if ((order == 2 && iteration != 1) || (order == 3 && iteration == 2)) {
		for (int i = 0; i < correspondences_pro.size(); ++i) {
			float PHIn_, PHIn__, PHIn_1, PHIn_2, PHIn__1, PHIn__2;
			if (inliers_new[i] == 1) {
				PHIn_1 = 1.0 * classificationTableParam * inliers_1[i] + 0.9 * classificationTableParam * inliers_2[i];
				PHIn_2 = 0.2 * (1.0 - classificationTableParam) * inliers_no_1[i] + 0.1 * (1.0 - classificationTableParam) * inliers_no_2[i];
				PHIn__1 = 0.0 * classificationTableParam * inliers_1[i] + 0.1 * classificationTableParam * inliers_2[i];
				PHIn__2 = 0.8 * (1.0 - classificationTableParam) * inliers_no_1[i] + 0.9 * (1.0 - classificationTableParam) * inliers_no_2[i];
			}
			else {
				PHIn_1 = 0.8 * (1.0 - classificationTableParam) * inliers_1[i] + 0.7 * (1.0 - classificationTableParam) * inliers_2[i];
				PHIn_2 = 0.1 * classificationTableParam * inliers_no_1[i] + 0.0 * classificationTableParam * inliers_no_2[i];
				PHIn__1 = 0.2 * (1.0 - classificationTableParam) * inliers_1[i] + 0.3 * (1.0 - classificationTableParam) * inliers_2[i];
				PHIn__2 = 0.9 * classificationTableParam * inliers_no_1[i] + 1.0 * classificationTableParam * inliers_no_2[i];
			}
			inliers_1[i] = PHIn_1;
			inliers_2[i] = PHIn_2;
			inliers_no_1[i] = PHIn__1;
			inliers_no_2[i] = PHIn__2;
			inliers_k_1[i] = PHIn_1 + PHIn_2;
			inliers_no_k_1[i] = PHIn__1 + PHIn__2;
			if(inliers_k_1[i] + inliers_no_k_1[i] == 0){
				correspondences_pro[i] = 0;
			}else{
				correspondences_pro[i] = inliers_k_1[i] / (inliers_k_1[i] + inliers_no_k_1[i]);
			}
		}
	}
	else if (order == 3 && iteration != 1 && iteration != 2) {
		for (int i = 0; i < correspondences_pro.size(); ++i) {
			float PHIn_, PHIn__, PHIn_1, PHIn_2, PHIn__1, PHIn__2;
			if (inliers_new[i] == 1) {
				PHIn_ = 1.0 * classificationTableParam * inliers_k_1[i] + 0.9 * classificationTableParam * inliers_k_1[i] + 0.6 * classificationTableParam * inliers_k_1[i] + 0.4 * classificationTableParam * inliers_k_1[i] + 0.3 * (1.0 - classificationTableParam) * inliers_no_k_1[i] + 0.1 * (1.0 - classificationTableParam) * inliers_no_k_1[i] + 0.2 * (1.0 - classificationTableParam) * inliers_no_k_1[i] + 0.05 * (1.0 - classificationTableParam) * inliers_no_k_1[i];
				PHIn__ = 0.0 * classificationTableParam * inliers_k_1[i] + 0.1 * classificationTableParam * inliers_k_1[i] + 0.4 * classificationTableParam * inliers_k_1[i] + 0.6 * classificationTableParam * inliers_k_1[i] + 0.7 * (1.0 - classificationTableParam) * inliers_no_k_1[i] + 0.9 * (1.0 - classificationTableParam) * inliers_no_k_1[i] + 0.8 * (1.0 - classificationTableParam) * inliers_no_k_1[i] + 0.95 * (1.0 - classificationTableParam) * inliers_no_k_1[i];
			}
			else {
				PHIn_ = 0.8 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.7 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.5 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.2 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.2 * classificationTableParam * inliers_no_k_1[i] + 0.3 * classificationTableParam * inliers_no_k_1[i] + 0.1 * classificationTableParam * inliers_no_k_1[i] + 0.0 * classificationTableParam * inliers_no_k_1[i];
				PHIn__ = 0.2 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.3 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.5 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.8 * (1.0 - classificationTableParam) * inliers_k_1[i] + 0.8 * classificationTableParam * inliers_no_k_1[i] + 0.7 * classificationTableParam * inliers_no_k_1[i] + 0.9 * classificationTableParam * inliers_no_k_1[i] + 1.0 * classificationTableParam * inliers_no_k_1[i];
			}
			inliers_k_1[i] = PHIn_;
			inliers_no_k_1[i] = PHIn__;
			correspondences_pro[i] = inliers_k_1[i] / (inliers_k_1[i] + inliers_no_k_1[i]);
		}
	}
}

Matf3D radiusSearchPoints(const Matf3D& query_points,
    const PointCloudPtr& target_cloud,
    float radius) {
    Matf3D result(3, 0);

    if (target_cloud->empty() || query_points.cols() == 0) {
        return result;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target_cloud);

    std::unordered_set<int> unique_indices;

    for (int i = 0; i < query_points.cols(); ++i) {
        pcl::PointXYZ query;
        query.x = query_points(0, i);
        query.y = query_points(1, i);
        query.z = query_points(2, i);

        std::vector<int> indices;
        std::vector<float> distances;
        if (kdtree.radiusSearch(query, radius, indices, distances) > 0) {
            for (const int idx : indices) {
                if (unique_indices.insert(idx).second) {
                    result.conservativeResize(3, result.cols() + 1);
                    result.col(result.cols() - 1) << 
                    target_cloud->points[idx].x,
                    target_cloud->points[idx].y,
                    target_cloud->points[idx].z;
                }
            }
        }
    }
    return result;
}

Mati1D getNonZeroColIndices(const Mati1D& flags) {
    int count = 0;
    Mati1D nonzero_column(1, flags.count());
    for (int i = 0; i < flags.cols(); ++i) {
        if (flags(0, i) > 0) {
            nonzero_column(0, count++) = i;
        } 
    }
    return nonzero_column;
}

void computeDistMatrix(const Matf3D& data, Eigen::MatrixXf& dist_matrix) {
    for (int i = 0; i < data.cols(); ++i)
        dist_matrix.row(i) = (data.colwise() - data.col(i)).colwise().norm();
}

void sortRowDescending(const Mati1D& data, std::vector<int>& sorted_col_indices) {
    std::vector<int> sorted_data(data.cols());
    sorted_col_indices.resize(data.cols());
    for (int i = 0; i < data.cols(); ++i) {
        sorted_data[i] = data(0, i);
        sorted_col_indices[i] = i;
    }

    std::sort(sorted_col_indices.begin(), sorted_col_indices.end(), [&sorted_data](int i, int j) {
        return sorted_data[i] > sorted_data[j]; });  
}

Eigen::Matrix4f ransacThreePoint(const Matf6D& x, Mati1D& bestinliers_idx, vector<double>& probs, int s, float t) {
    int max_data_trials = 1000;
    int max_trials = 10000;
	bool earlyStop = false;
    int npts = x.cols();
    
    vector<double> probs1;
    vector<double> inlk = probs;
    vector<double> outlk;
    outlk.resize(npts);
    transform(probs.begin(), probs.end(), outlk.begin(), [](double value) {
        return 1 - value;
    });

    vector<double> inl1;
    vector<double> outl1;
    vector<double> inl2;
    vector<double> outl2;
    float p = 0.99;
    int trialcount = 0;
    int bestscore = 0;
    Mati1D bestinliers;

    float N = 1; 
    float eps = std::numeric_limits<float>::epsilon();
    std::vector<int> new_inliers;
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f trans_best = Eigen::Matrix4f::Identity();
    while (N > trialcount) {
        bool update_pro = false;
        int count = 1;
        bool degenerate = true;
        probs1 = probs;
        transform(probs.begin(), probs.end(), probs.begin(),
                [](double val) { return std::round(val * 100.0); });
        vector<int> indices_map;
        indices_map.clear();
        for (int i = 0; i < npts; ++i) {
            indices_map.insert(indices_map.end(), probs[i], i);
        }
        Matf6D inliers_idx_1;
        while (degenerate) {
            int id_1 = -1;
            int ind_rand1 = std::rand() % indices_map.size();
            id_1 = indices_map[ind_rand1];
            int id_2 = -1;
            while(id_2 == -1 || id_2 == id_1){
                int ind_rand2 = std::rand() % indices_map.size();
                if(indices_map[ind_rand2] != id_1){
                    id_2 = indices_map[ind_rand2];
                }
            }
            int id_3 = -1;
            while(id_3 == -1 || id_3 == id_1 || id_3 == id_2){
                int ind_rand3 = std::rand() % indices_map.size();
                if((indices_map[ind_rand3] != id_1) && (indices_map[ind_rand3] != id_2)){
                    id_3 = indices_map[ind_rand3];
                }
            }

            if (id_1 == id_2 || id_1 == id_3 || id_2 == id_3) {
                continue;
            }

            Eigen::Vector3i ind(id_1, id_2, id_3);
            degenerate = isDegenerate(x(Eigen::all, ind));

            if (!degenerate) {
                trans = rigidTransform(x(Eigen::seq(0, 2), ind), x(Eigen::seq(3, 5), ind));
                Matf6D inliers_idx_3 = x(Eigen::all, ind);
                inliers_idx_1 = inliers_idx_3;
            }

            ++count;
            if (count > max_data_trials) {
                break;
            }
        }
        
		Mati1D inlier_column = dist3d(trans, x, t);
        int inlier_size = inlier_column.cols();

        Mati1D inliers_idx_2 = bestinliers_idx(Eigen::all, inlier_column);
		

        if (inlier_size >= bestscore) {
            update_pro = true;
            bestscore = inlier_size;
            bestinliers = inlier_column;
            trans_best = trans;
            
            float fracinliers = 0, pNoOutliers = 0;  

            Mati1D loInliers = bestinliers;
            int loIter = 0;
			int loRansacMaxIter = 50;
            int NOSample = std::min(s * 7, (int)loInliers.cols());

            if (inlier_size < s) {
                fracinliers = static_cast<float>(inlier_size) / npts;
                pNoOutliers = 1 - std::pow(fracinliers, s);
                pNoOutliers = std::max(eps, pNoOutliers);
                pNoOutliers = std::min(1 - eps, pNoOutliers);
                N = log(1-p)/log(pNoOutliers);
                probs = probs1;
                continue;                    
            }

            if (trialcount < 50) {
                loIter = loRansacMaxIter;
            }

            std::vector<int> loInliers_vec(&loInliers(0, 0), loInliers.data() + loInliers.size());
            while (loIter < loRansacMaxIter) {
                loIter = loIter + 1;
                std::random_shuffle(loInliers_vec.begin(), loInliers_vec.end());
                std::vector<int> loind(loInliers_vec.begin(), loInliers_vec.begin() + NOSample);
                trans = rigidTransform(x(Eigen::seq(0, 2), loind), x(Eigen::seq(3, 5), loind));
				Mati1D loUpdatedInliers = dist3d(trans, x, t);
                if (loUpdatedInliers.cols() > bestscore) {
                    bestscore = loUpdatedInliers.cols();
                    inlier_size = bestscore;
                    bestinliers = loUpdatedInliers;
                    inlier_column = loUpdatedInliers;
                    trans_best = trans;
                }
            }
            fracinliers =  inlier_size / static_cast<float>(npts);
            pNoOutliers = 1.0 - std::pow(fracinliers, s);
            pNoOutliers = std::max(eps, pNoOutliers);
            pNoOutliers = std::min(1 - eps, pNoOutliers);
            N = log(1-p)/log(pNoOutliers);
            N = std::max(N, 0.0f);
        }
        if(trialcount < 16)
        {
            update_pro = true;
        }
        if(update_pro)
        {
			new_inliers.clear();
			new_inliers.assign(npts, 0);
			std::for_each(inlier_column.data(), inlier_column.data() + inlier_column.size(),
						[&new_inliers](int index) { 
							new_inliers[index] = 1;
						});
            double inlierRatio = (double)inlier_size / npts;
            double bestOutlierRatio = (double)1.0 - ((double)bestscore /(double)npts);
            double param = (inlierRatio < (double)0.7143) ? (inlierRatio * (double)0.62) + (double)0.5 : (inlierRatio * (double)0.2) + (double)0.8;
            int order = 2;
            int outlierNumber = 0;
            double sum_early = 0.0;
            updateProbabilities(order, trialcount + 1, param, outlierNumber, t, inl1, outl1, inl2, outl2, inlk, outlk, probs, probs1, new_inliers, earlyStop, sum_early);
		}else{
            probs = probs1;
        }

        ++trialcount;

        if (trialcount > max_trials) {
            break;
        }
    }
    if (trans_best.array().isNaN().any()) {
        std::cout << "Three-POint RANSAC's aatrix exist NaN.\n";
        trans_best.setIdentity();
    }
    return trans_best;
}

bool isDegenerate(const Eigen::Matrix<float, 6, 3>& x) {
    Eigen::Matrix<float, 3, 3> x1 = x.topRows(3);
    Eigen::Matrix<float, 3, 3> x2 = x.bottomRows(3);
    bool flag_1 = isCollinear(x1.col(0), x1.col(1), x1.col(2));
    bool flag_2 = isCollinear(x2.col(0), x2.col(1), x2.col(2));
    if (flag_1 || flag_2) {
        return true;
    } else {
        return false;
    }
}

bool isCollinear(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3) {
    Eigen::Vector3f p1p2 = p2 - p1;
    Eigen::Vector3f p1p3 = p3 - p1;
    return (p1p2.cross(p1p3)).norm() < std::numeric_limits<float>::epsilon();
}

Eigen::Matrix4f rigidTransform(const Matf3D& A, const Matf3D& B) {
    Eigen::Matrix<float, 3, 1> lc = A.rowwise().mean();
    Eigen::Matrix<float, 3, 1> rc = B.rowwise().mean();
    Eigen::Matrix3f M = (A.colwise() - lc) * (B.colwise() - rc).transpose();
    
    float Sxx = M(0,0); float Syx = M(1,0); float Szx = M(2,0);
    float Sxy = M(0,1); float Syy = M(1,1); float Szy = M(2,1);
    float Sxz = M(0,2); float Syz = M(1,2); float Szz = M(2,2);
    Eigen::Matrix4f N = Eigen::Matrix4f::Identity();
    N << Sxx+Syy+Szz, Syz-Szy, Szx-Sxz, Sxy-Syx,
        Syz-Szy, Sxx-Syy-Szz, Sxy+Syx, Szx+Sxz,
        Szx-Sxz, Sxy+Syx, -Sxx+Syy-Szz, Syz+Szy,
        Sxy-Syx, Szx+Sxz, Syz+Szy, -Sxx-Syy+Szz;

    Eigen::EigenSolver<Eigen::Matrix4f> es(N);
	Eigen::Vector4f evalue = es.eigenvalues().real();
	Eigen::Matrix4f evector = es.eigenvectors().real();
    
    int max_row = 0, max_col = 0; 
    evalue.maxCoeff(&max_row, &max_col);
    Eigen::Vector4f q = evector.col(max_row);

    q.cwiseAbs().maxCoeff(&max_row, &max_col); 
    float max_vec = q(max_row, 0);
    if (max_vec < 0) {
        q = q * (-1);
    }
    q.normalize();
    float q0 = q(0);
    float qx = q(1);
    float qy = q(2);
    float qz = q(3);
    Eigen::Vector3f v = q.tail(3);

    Eigen::Matrix3f Z = Eigen::Matrix3f::Identity();
    Z << q0, -qz, qy,
        qz, q0, -qx,
        -qy, qx, q0;
    Eigen::Matrix3f R = v * v.transpose() + Z * Z;
    Eigen::Vector3f t = rc - R * lc;

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

Mati1D dist3d(const Eigen::Matrix4f& trans, const Matf6D& x, const float t) {
    Eigen::Matrix3f R = trans.block<3, 3>(0, 0);
    Eigen::Vector3f T = trans.block<3, 1>(0, 3);
    Matf3D x2_ = (R * x.topRows(3)).colwise() + T;
    Matf1D d2 = (x2_ - x.bottomRows(3)).colwise().norm();
    Mati1D flag = (d2.array() < t).cast<int>();
    return getNonZeroColIndices(flag);
}

float computeThresholdFromProbs(Mati1D bestinliers_idx3, Mati1D inlier_column, std::vector<double> probs, Eigen::Matrix4f init, Matf3D& src, Matf3D& dst, float& prob){
	float best_dist;
	int inliers_num = bestinliers_idx3.cols();
	Matf3D fit = (init.block<3, 3>(0, 0) * src).colwise() + init.block<3, 1>(0, 3);
    Matf1D residuals = (fit - dst).colwise().norm();
	while(prob > 0){
		best_dist = 0.0001;
		int sum = 0;
		for (int i = 0; i < inlier_column.cols(); i++) {
			int idx = bestinliers_idx3(0, i);
			if(static_cast<float>(probs[inlier_column(0, i)]) > prob){
				float dist = residuals(0, i);
				if(best_dist < dist){
					best_dist = dist;
				}
				sum++;
			}
		}
		if(sum > 0.4 * inliers_num){
			break;
		}
		prob = prob - 0.05;
	}
	best_dist = best_dist / 3;
	return best_dist;
}