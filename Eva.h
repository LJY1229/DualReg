#ifndef _EVA_H_ 
#define _EVA_H_
#define Pi 3.1415926
#define constE 2.718282
#define NULL_POINTID -1
#define NULL_Saliency -1000
#define Random(x) (rand()%x)
#define Corres_view_gap -200
#define Align_precision_threshold 0.1
#define tR 116//30
#define tG 205//144
#define tB 211//255
#define sR 253//209//220
#define sG 224//26//20
#define sB 2//32//60
#define L2_thresh 0.5
#define Ratio_thresh 0.2
#define GC_dist_thresh 3
#define Hough_bin_num 15
#define SI_GC_thresh 0.8
#define RANSAC_Iter_Num 5000
#define GTM_Iter_Num 100
#define CV_voting_size 20
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

using namespace std;
extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;

#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <unordered_set>
#include <Eigen/Eigen>
#include <igraph/igraph.h>
#include <sys/stat.h>
#include <unistd.h>

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr KdTreePtr;
typedef pcl::PointXYZ PointInT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD;
typedef Eigen::Matrix<float, 6, Eigen::Dynamic> Matf6D;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Matf3D;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3> MatfD3;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic> Mati1D;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> Matf1D;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXf;
typedef Eigen::Matrix<float, 4, 4> Matrix4f;
typedef Eigen::Vector3f Vector3f;

typedef struct {
    float x;
    float y;
    float z;
} Vertex;

typedef struct {
    int pointID;
    Vertex x_axis;
    Vertex y_axis;
    Vertex z_axis;
} LRF; // Local Reference Frame

typedef struct {
    int src_index;
    int des_index;
    pcl::PointXYZ src;
    pcl::PointXYZ des;
    Eigen::Vector3f src_norm;
    Eigen::Vector3f des_norm;
    Eigen::Matrix3f covariance_src, covariance_des;
    Eigen::Vector4f centeroid_src, centeroid_des;
    double score;
    int inlier_weight;
} Correspondence; // formerly Corre_3DMatch

typedef struct {
    int PointID;
    float eig1_2;
    float eig2_3;
    float saliency;
    bool isTrue; // formerly TorF
} ISSKeypoint; // formerly ISS_Key_Type

/********************************************** funcs ***************************************/
float computeCloudResolution(PointCloudPtr& cloud);
int voxelDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& new_cloud, float leaf_size);
Matrix4f estimateRigidSVD(const Matf3D& A, const Matf3D& B, int& max_vec_sign);
Matf3D radiusSearchPoints(const Matf3D& query_points, const PointCloudPtr& target_cloud, float radius);
void postRefinement(Matf3D& src, Matf3D& dst, Eigen::Matrix4f& initial, float inlier_thresh, int iterations, float norm_thresh, Mati1D bestinliers_idx3, const Matf3D& src_result, const Matf3D& des_result, float lambda);
Matf6D ransacOnePoint(Matf6D& x, Matf6D& x_normal, float t, Mati1D& bestinliers_idx, float inlier_judge_thresh, float sub_th);
Mati1D getNonZeroColIndices(const Mati1D& flags);
void computeDistMatrix(const Matf3D& data, Eigen::MatrixXf& dist_matrix);
void sortRowDescending(const Mati1D& data, std::vector<int>& sorted_col_indices);
Eigen::Matrix4f ransacThreePoint(const Matf6D& x, Mati1D& bestinliers_idx, vector<double>& probs, int s, float t);
Eigen::Matrix4f weightedRigidTrans(const Matf3D& A, const Matf3D& B, Eigen::MatrixXf& weights);
bool isDegenerate(const Eigen::Matrix<float, 6, 3>& x);
bool isCollinear(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3);
Eigen::Matrix4f rigidTransform(const Matf3D& A, const Matf3D& B);
Mati1D dist3d(const Eigen::Matrix4f& trans, const Matf6D& x, float t);
float computeThresholdFromProbs(Mati1D bestinliers_idx3, Mati1D inlier_column, std::vector<double> probs, Eigen::Matrix4f init, Matf3D& src, Matf3D& dst, float& prob);
void updateProbabilities(int order, int iter, float param, int& outlierNum, double th, std::vector<double>& inl1, std::vector<double>& outl1, std::vector<double>& inl2, std::vector<double>& outl2, std::vector<double>& inlk, std::vector<double>& outlk, std::vector<double>& prob, std::vector<double> prob1, std::vector<int>& new_inliers, bool& earlyStop, double& sum);
double calcRotError(Eigen::Matrix3d& est, Eigen::Matrix3d& gt);
double calcTransError(Eigen::Vector3d& est, Eigen::Vector3d& gt);
bool evalEstimate(Eigen::Matrix4d est, Eigen::Matrix4d gt, double re_th, double te_th, double& RE, double& TE);
bool registration(const string &name, string src_file, string tgt_file, const string &corr_path, const string &label_path, const string &ov_label, const string &gt_mat, const string &out_dir, double& RE, double& TE, double& inlier_cnt, double& total_cnt, double& inlier_ratio, double& success_cnt, double& total_est, double& avg_dist, const string &descriptor, vector<double>& times);

#endif