#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "omp.h"
#include "Eva.h"
#include <stdarg.h>
#include <chrono>
#include <vector>
#include <unordered_map>

using namespace Eigen;
using namespace std;
extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;

void computeGTOverlap(vector<Correspondence>& corres, PointCloudPtr &src, PointCloudPtr &tgt, Eigen::Matrix4d &GTmat,  bool ind, double GT_thresh, double &max_weight){
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *src_trans, GTmat);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src_trans, kdtree_des;
    kdtree_src_trans.setInputCloud(src_trans);
    kdtree_des.setInputCloud(tgt);
    vector<int>src_ind(1), des_ind(1);
    vector<float>src_dis(1), des_dis(1);
    PointCloudPtr src_corr(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr src_corr_trans(new pcl::PointCloud<pcl::PointXYZ>);
    if(!ind){
        for(auto & i :corres){
            src_corr->points.push_back(i.src);
        }
        pcl::transformPointCloud(*src_corr, *src_corr_trans, GTmat);
        src_corr.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    for(int i  = 0; i < corres.size(); i++){
        pcl::PointXYZ src_query, des_query;
        if(!ind){
            src_query = src_corr_trans->points[i];
            des_query = corres[i].des;
        }
        else{
            src_query = src->points[corres[i].src_index];
            des_query = tgt->points[corres[i].des_index];
        }
        kdtree_des.nearestKSearch(src_query, 1, des_ind, src_dis);
        kdtree_src_trans.nearestKSearch(des_query, 1, src_ind, des_dis);
        int src_ov_score = src_dis[0] > pow(GT_thresh,2) ? 0 : 1;
        int des_ov_score = des_dis[0] > pow(GT_thresh,2) ? 0 : 1;
        if(src_ov_score && des_ov_score){
            corres[i].score = 1;
            max_weight = 1;
        }
        else{
            corres[i].score = 0;
        }
    }
    src_corr_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

bool registration(const string &name, string src_file, string tgt_file, const string &corr_path, const string &label_path, const string &ov_label, const string &gt_mat, const string &out_dir, double& RE, double& TE, double& inlier_cnt, double& total_cnt, double& inlier_ratio, double& success_cnt, double& total_est, double& avg_dist, const string &descriptor, vector<double>& times) {
	success_cnt = 0;
	if (!no_logs && access(out_dir.c_str(), 0))
	{
		if (mkdir(out_dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
			cout << " 创建数据项目录失败 " << endl;
			exit(-1);
		}
	}
	cout << out_dir << endl;
	string dataPath = corr_path.substr(0, corr_path.rfind("/"));
	string item_name = out_dir.substr(out_dir.rfind("/") + 1, out_dir.length());

	FILE* corr, * gt;
	corr = fopen(corr_path.c_str(), "r");
	gt = fopen(label_path.c_str(), "r");
	if (corr == NULL) {
		std::cout << " error in loading correspondence data. " << std::endl;
        cout << corr_path << endl;
		exit(-1);
	}
	if (gt == NULL) {
		std::cout << " error in loading ground truth label data. " << std::endl;
        cout << label_path << endl;
	}

	FILE* ov;
	vector<double> ov_corr_label;
    double max_weight = 0;
	if (add_overlap && ov_label != "NULL")
	{
		ov = fopen(ov_label.c_str(), "r");
		if (ov == NULL) {
			std::cout << " error in loading overlap data. " << std::endl;
			exit(-1);
		}
		while (!feof(ov))
		{
			double value;
			fscanf(ov, "%lf\n", &value);
            if(value > max_weight){
                max_weight = value;
            }
			ov_corr_label.push_back(value);

		}
		fclose(ov);
		cout << "load overlap data finished." << endl;
	}
	PointCloudPtr raw_src(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr raw_tgt(new pcl::PointCloud<pcl::PointXYZ>);
	float raw_tgt_res = 0;
	float raw_src_res = 0;

	PointCloudPtr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr norm_src(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr norm_tgt(new pcl::PointCloud<pcl::Normal>);
	vector<Correspondence> corres;
	vector<int> true_labels;
	inlier_cnt = 0;
	float resolution = 0;
	float th = 0;
	bool kitti = false;
    Eigen::Matrix4d GTmat;

    FILE* fp = fopen(gt_mat.c_str(), "r");
    if (fp == NULL)
    {
        printf("Mat File can't open!\n");
        return -1;
    }
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(0, 0), &GTmat(0, 1), &GTmat(0, 2), &GTmat(0, 3));
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(1, 0), &GTmat(1, 1), &GTmat(1, 2), &GTmat(1, 3));
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(2, 0), &GTmat(2, 1), &GTmat(2, 2), &GTmat(2, 3));
    fscanf(fp, "%lf %lf %lf %lf\n", &GTmat(3, 0), &GTmat(3, 1), &GTmat(3, 2), &GTmat(3, 3));
    fclose(fp);
	if (low_inlieratio)
	{
		if (pcl::io::loadPCDFile(src_file.c_str(), *cloud_src) < 0) {
			std::cout << " error in loading source pointcloud. " << std::endl;
			exit(-1);
		}

		if (pcl::io::loadPCDFile(tgt_file.c_str(), *cloud_tgt) < 0) {
			std::cout << " error in loading target pointcloud. " << std::endl;
			exit(-1);
		}
        while (!feof(corr)) {
            Correspondence t;
            pcl::PointXYZ src, des;
            fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
            t.src = src;
            t.des = des;
            corres.push_back(t);
        }
        if(add_overlap && ov_label == "NULL") {
            cout << "load gt overlap" << endl;
            computeGTOverlap(corres, cloud_src, cloud_tgt, GTmat, false, 0.0375, max_weight);
        }
        else if (add_overlap && ov_label != "NULL"){
            for(int i  = 0; i < corres.size(); i++){
                corres[i].score = ov_corr_label[i];
                if(ov_corr_label[i] > max_weight){
                    max_weight = ov_corr_label[i];
                }
            }
        }
		fclose(corr);
	}
	else {
		if (name == "KITTI")
		{
			float bbox_src, bbox_tgt, scale_src, scale_tgt;
			if (!(src_file == "NULL" && tgt_file == "NULL"))
			{
				if (pcl::io::loadPCDFile<pcl::PointXYZ>(src_file.c_str(), *cloud_src) == -1) {
					std::cout << "Error in loading source pointcloud." << std::endl;
					exit(-1);
				}

				if (pcl::io::loadPCDFile<pcl::PointXYZ>(tgt_file.c_str(), *cloud_tgt) == -1) {
					std::cout << "Error in loading target pointcloud." << std::endl;
					exit(-1);
				}

				float rs = computeCloudResolution(cloud_src);
				float rt = computeCloudResolution(cloud_tgt);
				th = std::max(rs, rt);
			}

			int idx = 0;
			kitti = true;
			while (!feof(corr))
			{
				Correspondence t;
				fscanf(corr, "%f %f %f %f %f %f %f %f %f %f %f %f\n", &t.src.x, &t.src.y, &t.src.z, 
						&t.des.x, &t.des.y, &t.des.z, &t.src_norm[0], &t.src_norm[1], &t.src_norm[2],
           				&t.des_norm[0], &t.des_norm[1], &t.des_norm[2]);
				
				if (add_overlap)
				{
					t.score = ov_corr_label[idx];
				}
				else
				{
					t.score = 0;
				}
				corres.push_back(t);
				idx++;
			}
			fclose(corr);

		}
		else if (name == "3dmatch" || name == "3dlomatch") {

			if (!(src_file == "NULL" && tgt_file == "NULL"))
			{
				if (pcl::io::loadPLYFile<pcl::PointXYZ>(src_file.c_str(), *cloud_src) == -1) {
					std::cout << " error in loading source pointcloud. " << std::endl;
					exit(-1);
				}
				if (pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_file.c_str(), *cloud_tgt) == -1) {
					std::cout << " error in loading target pointcloud. " << std::endl;
					exit(-1);
				}

				float rs = computeCloudResolution(cloud_src);
				float rt = computeCloudResolution(cloud_tgt);
				th = std::max(rs, rt);

				int idx = 0;
				while (!feof(corr))
				{
					Correspondence t;
					fscanf(corr, "%f %f %f %f %f %f %f %f %f %f %f %f\n", &t.src.x, &t.src.y, &t.src.z, 
							&t.des.x, &t.des.y, &t.des.z, &t.src_norm[0], &t.src_norm[1], &t.src_norm[2],
							&t.des_norm[0], &t.des_norm[1], &t.des_norm[2]);
					if (add_overlap && ov_label != "NULL")
                    {
                        t.score = ov_corr_label[idx];
                    }
                    else{
                        t.score = 0;
                    }
					t.inlier_weight = 0;
					corres.push_back(t);
                    idx ++;
				}
				fclose(corr);
                if(add_overlap && ov_label == "NULL"){
                    cout << "load gt overlap" << endl;
                    computeGTOverlap(corres, cloud_src, cloud_tgt, GTmat, false, 0.0375, max_weight);
                }
			}
			else {
				int idx = 0;
				while (!feof(corr))
				{
					Correspondence t;
					pcl::PointXYZ src, des;
					fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
					t.src = src;
					t.des = des;
					t.inlier_weight = 0;
					if (add_overlap)
					{
						t.score = ov_corr_label[idx];
					}
					else
					{
						t.score = 0;
					}
					corres.push_back(t);
					idx++;
				}
				fclose(corr);
					}
					}
		else {
			exit(-1);
		}
	}
	
	total_cnt = corres.size();
	if(gt != NULL){
		while (!feof(gt))
		{
			int value;
			fscanf(gt, "%d\n", &value);
			true_labels.push_back(value);
			if (value == 1)
			{
				inlier_cnt++;
			}
		}
		fclose(gt);
	}

	inlier_ratio = 0;
	if (inlier_cnt == 0)
	{
		cout << " NO INLIERS！ " << endl;
	}
	inlier_ratio = inlier_cnt / (total_cnt / 1.0);

	double RE_th, TE_th, inlier_th, inlier_judge;
	float weight_lambda, sub_th;
	if (name == "KITTI")
	{
		RE_th = 5;
		TE_th = 60;
		inlier_th = 0.6;
		weight_lambda = 1.0f;
		inlier_judge = 3 * th;
		if(descriptor == "fpfh") sub_th = 0.9;
		else sub_th = 0.95;
	}
	else if (name == "3dmatch" || name == "3dlomatch")
	{
		RE_th = 15;
		TE_th = 30;
		inlier_th = 0.1;
		weight_lambda = 0.05f;
		inlier_judge = inlier_th;
		if(name == "3dmatch") sub_th = 0.2;
		else sub_th = 0.95;
	}
	

	Eigen::Matrix4f Mat;
	bool found = false;
	Matf6D pts_vec(6, corres.size());
	Matf6D norm_vec(6, corres.size());
	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < corres.size(); i++)
	{
		pts_vec(0, i) = corres[i].src.x;
		pts_vec(1, i) = corres[i].src.y;
		pts_vec(2, i) = corres[i].src.z;
		pts_vec(3, i) = corres[i].des.x;
		pts_vec(4, i) = corres[i].des.y;
		pts_vec(5, i) = corres[i].des.z;
		
		norm_vec(0, i) = corres[i].src_norm[0];
		norm_vec(1, i) = corres[i].src_norm[1];
		norm_vec(2, i) = corres[i].src_norm[2];
		norm_vec(3, i) = corres[i].des_norm[0];
		norm_vec(4, i) = corres[i].des_norm[1];
		norm_vec(5, i) = corres[i].des_norm[2];
	}

	Mati1D best_idx;
	Matf6D pts_sub = ransacOnePoint(pts_vec, norm_vec, 3 * th, best_idx, inlier_th, sub_th);
	Matf6D pts_sub2 = pts_vec(Eigen::all, best_idx);

	vector<double> probs;
	probs.resize(best_idx.cols());
	probs.assign(best_idx.cols(), 0.5);
	
	Mat = ransacThreePoint(pts_sub2, best_idx, probs, 3, inlier_th); 
	Mati1D inlier_col;
	inlier_col = dist3d(Mat, pts_sub2, inlier_judge);
	Matf6D pts_sub3 = pts_sub2(Eigen::all, inlier_col);
	
	Mati1D best_idx3 = best_idx(Eigen::all, inlier_col);
	Matf3D src = pts_sub3.topRows(3);
	Matf3D dst = pts_sub3.bottomRows(3);

	PointCloudPtr src_down(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr tgt_down(new pcl::PointCloud<pcl::PointXYZ>);
	voxelDownsample(cloud_src, src_down, 5 * th);
	voxelDownsample(cloud_tgt, tgt_down, 5 * th);
	
	Matf3D src_sub = radiusSearchPoints(src, src_down, 50 * th);
	Matf3D tgt_sub = radiusSearchPoints(dst, tgt_down, 50 * th);

	float norm_th = 0.001;
	float xi;
	float prob = 0.95;
	xi = computeThresholdFromProbs(best_idx3, inlier_col, probs, Mat, src, dst, prob);
	Eigen::Matrix4f GTmat_f = GTmat.cast<float>();
	postRefinement(src, dst, Mat, xi, 200, norm_th, best_idx3, src_sub, tgt_sub, weight_lambda);
	
	auto end = chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
	std::cout << "Time: " << duration.count() << "s" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_src, *cloud_trans, Mat);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_GT(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_src, *cloud_trans_GT, GTmat_f);
	double end_dist = 0.0;

	for (size_t i = 0; i < cloud_trans->size(); ++i) {
		pcl::PointXYZ pt_1 = cloud_trans->points[i];
		pcl::PointXYZ pt_GT = cloud_trans_GT->points[i];
		double dist = pow(pt_1.x - pt_GT.x, 2) + pow(pt_1.y - pt_GT.y, 2) + pow(pt_1.z - pt_GT.z, 2);
		end_dist += dist;
	}
	double avg_end_dist = end_dist / cloud_trans->size();
	avg_end_dist = sqrt(avg_end_dist);
	avg_dist = avg_end_dist * 100;
	cout << "RMSE: " << avg_dist << endl;

	
	string save_time = out_dir + "/time.txt";
	ofstream out_time(save_time, ios::trunc);
	out_time.setf(ios::fixed, ios::floatfield);
	out_time << duration.count() << endl;

	Eigen::MatrixXd Mat_d = Mat.cast<double>();
	found = evalEstimate(Mat_d, GTmat, RE_th, TE_th, RE, TE);
	times.push_back(duration.count());
	

	string save_mat = out_dir + "/est.txt";
	ofstream out_mat(save_mat, ios::trunc);
	out_mat.setf(ios::fixed, ios::floatfield);
	out_mat << setprecision(6) << Mat << endl;
	out_mat.close();
	out_time.close();
	cout << "RE=" << RE << " " << "TE=" << TE << endl;
	
	if (found)
	{
		cout << Mat_d << endl;
		return true;
	}
	return false;
}