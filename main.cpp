#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <string>
#include <numeric>
#include <cstdlib>
#include <getopt.h>
#include "Eva.h"
#include <unistd.h>
#include <fstream>      
#include <iomanip>    
using namespace std;

string folderPath;
bool add_overlap;
bool low_inlieratio;
bool no_logs;

string threeDMatch[8] = {
    "7-scenes-redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_uc-scan3",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-hotel_umd-maryland_hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
};

string threeDlomatch[8] = {
    "7-scenes-redkitchen_3dlomatch",
    "sun3d-home_at-home_at_scan1_2013_jan_1_3dlomatch",
    "sun3d-home_md-home_md_scan9_2012_sep_30_3dlomatch",
    "sun3d-hotel_uc-scan3_3dlomatch",
    "sun3d-hotel_umd-maryland_hotel1_3dlomatch",
    "sun3d-hotel_umd-maryland_hotel3_3dlomatch",
    "sun3d-mit_76_studyroom-76-1studyroom2_3dlomatch",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_3dlomatch",
};

double RE, TE, success_estimate_rate;
vector<int> scene_num;

vector<string> analyse(const string& name,
                       const string& result_scene,
                       const string& dataset_scene,
                       const string& descriptor,
                       ofstream& outfile,
                       int iters,
                       int data_index,
                       double& out_rmse_sum,
                       double& out_time_sum)
{
    if (!no_logs && access(result_scene.c_str(), 0))
    {
        if (mkdir(result_scene.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            cout << " Create scene folder failed! " << endl;
            exit(-1);
        }
    }

    vector<string> error_pair;
    string error_txt;
    if (descriptor == "fpfh")
    {
        error_txt = dataset_scene + "/dataload.txt";
    }
    else if (descriptor == "fcgf")
    {
        error_txt = dataset_scene + "/dataload_fcgf.txt";
    }
    if (access(error_txt.c_str(), 0))
    {
        cout << " Could not find dataloader file! " << endl;
        exit(-1);
    }

    ifstream f1(error_txt);
    string line;
    while (getline(f1, line))
    {
        error_pair.push_back(line);
    }
    f1.close();
    scene_num.push_back(error_pair.size());

    vector<string> match_success_pair;
    int index = 1;
    RE = 0;
    TE = 0;
    success_estimate_rate = 0;
    double rmse_sum = 0.0, time_sum = 0.0; 
    vector<double> time;

    for (const auto& pair : error_pair)
    {
        time.clear();
        cout << "Pair " << index << ", " << "total " << error_pair.size() << " pairs." << endl;
        index++;
        string result_folder = result_scene + "/" + pair;
        string::size_type i = pair.find("+") + 1;
        string src_filename = dataset_scene + "/" + pair.substr(0, i - 1) + ".ply";
        string des_filename = dataset_scene + "/" + pair.substr(i, pair.length() - i) + ".ply";
        string corr_path = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@corr_fcgf_n.txt" : "@corr_n.txt");
        string gt_label = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@label_fcgf.txt" : "@label.txt");
        string gt_mat_path = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@GTmat_fcgf.txt" : "@GTmat.txt");

        string ov_label = "NULL";
        double re = 0, te = 0;
        double inlier_num, total_num, inlier_ratio, success_estimate, total_estimate, average_distance;
        std::srand(index);
        int corrected = registration(name, src_filename, des_filename, corr_path, gt_label, ov_label,
                                     gt_mat_path, result_folder, re, te, inlier_num, total_num,
                                     inlier_ratio, success_estimate, total_estimate, average_distance,
                                     descriptor, time);

        double est_rr = success_estimate / (total_estimate / 1.0);
        success_estimate_rate += est_rr;
        double rmse = average_distance;
        double time_construction = time[0]; 

        if (corrected)
        {
            RE += re;
            TE += te;
            rmse_sum += rmse;
            time_sum += time_construction;
            match_success_pair.push_back(pair);
        }

        outfile << pair << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
        outfile << fixed << setprecision(4) << inlier_ratio << ',';
        outfile << setprecision(6) << rmse << ',' << re << ',' << te << ',';
        outfile << setprecision(6) << time_construction << endl;

        if (corrected)
            cout << pair << " Success." << endl;
        else
            cout << pair << " Fail." << endl;
        cout << endl;
    }

    out_rmse_sum = rmse_sum;
    out_time_sum = time_sum;
    return match_success_pair;
}

void usage(){
    cout << "Usage:" << endl;
    cout << "\tHELP --help" <<endl;
    cout << "\tREQUIRED ARGS:" << endl;
    cout << "\t\t--output_path\toutput path for saving results." << endl;
    cout << "\t\t--input_path\tinput data path." << endl;
    cout << "\t\t--dataset_name\tdataset name. [3dmatch/3dlomatch/KITTI]" << endl;
    cout << "\t\t--descriptor\tdescriptor name. [fpfh/fcgf]" << endl;
    cout << "\t\t--start_index\tstart from given index. (begin from 0)" << endl;
    cout << "\tOPTIONAL ARGS:" << endl;
    cout << "\t\t--no_logs\tforbid generation of log files." << endl;
}

int main(int argc, char** argv) {
    add_overlap = false;
    low_inlieratio = false;
    no_logs = false;
    int id = 0;
    string resultPath;
    string datasetPath;
    string datasetName;
    string descriptor;

    int opt;
    int digit_opind = 0;
    int option_index = 0;
    static struct option long_options[] = {
            {"output_path", required_argument, NULL, 'o'},
            {"input_path", required_argument, NULL, 'i'},
            {"dataset_name", required_argument, NULL, 'n'},
            {"descriptor", required_argument, NULL, 'd'},
            {"start_index", required_argument, NULL, 's'},
            {"no_logs", optional_argument, NULL, 'g'},
            {"help", optional_argument, NULL, 'h'},
            {NULL, 0, 0, '\0'}
    };

    while((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1){
        switch (opt) {
            case 'h':
                usage();
                exit(0);
            case 'o':
                resultPath = optarg;
                break;
            case 'i':
                datasetPath = optarg;
                break;
            case 'n':
                datasetName = optarg;
                break;
            case 'd':
                descriptor = optarg;
                break;
            case 'g':
                no_logs = true;
                break;
            case 's':
                id = atoi(optarg);
                break;
            case '?':
                printf("Unknown option: %c\n",(char)optopt);
                usage();
                exit(-1);
        }
    }
    if(argc  < 11){
        cout << 11 - argc <<" more args are required." << endl;
        usage();
        exit(-1);
    }

    cout << "Check your args setting:" << endl;
    cout << "\toutput_path: " << resultPath << endl;
    cout << "\tinput_path: " << datasetPath << endl;
    cout << "\tdataset_name: " << datasetName << endl;
    cout << "\tdescriptor: " << descriptor << endl;
    cout << "\tstart_index: " << id << endl;
    cout << "\tno_logs: " << no_logs << endl;
    sleep(5);

    int corrected = 0;
    int total_num = 0;
    double total_re = 0;
    double total_te = 0;
    double total_rmse = 0; 
    double total_time = 0;  
    vector<double> total_success_est_rate;
    vector<int> scene_correct_num;
    vector<double> scene_re_sum;
    vector<double> scene_te_sum;
    vector<double> scene_rmse_sum; 
    vector<double> scene_time_sum; 

    if (access(resultPath.c_str(), 0))
    {
        if (mkdir(resultPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            cout << " Create save folder failed! " << endl;
            exit(-1);
        }
    }

    if (datasetName == "3dmatch")
    {
        for (size_t i = id; i < 8; i++)
        {
            cout << i + 1 << ":" << threeDMatch[i] << endl;
            string analyse_csv = resultPath + "/" + threeDMatch[i] + "_" + descriptor + ".csv";
            ofstream outFile;
            outFile.open(analyse_csv.c_str(), ios::out);
            outFile.setf(ios::fixed, ios::floatfield);
            outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ','
                    << "inlier_ratio" << ',' << "RMSE" << ',' << "RE" << ',' << "TE" << ',' << "Time" << endl;

            double scene_rmse = 0.0, scene_time = 0.0;
            vector<string> matched = analyse("3dmatch", resultPath + "/" + threeDMatch[i],
                                             datasetPath + "/" + threeDMatch[i], descriptor, outFile,
                                             id, i, scene_rmse, scene_time);
            scene_re_sum.push_back(RE);
            scene_te_sum.push_back(TE);
            scene_rmse_sum.push_back(scene_rmse);
            scene_time_sum.push_back(scene_time);

            if (!matched.empty())
            {
                cout << endl;
                cout << threeDMatch[i] << ":" << endl;
                for (auto t : matched)
                {
                    cout << "\t" << t << endl;
                }
                cout << endl;
                cout << threeDMatch[i] << ":" << matched.size() / (scene_num[i] / 1.0) << endl;
                cout << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
                corrected += matched.size();
                total_success_est_rate.push_back(success_estimate_rate);
                scene_correct_num.push_back(matched.size());
            }
            else
            {
                scene_correct_num.push_back(0);
                total_success_est_rate.push_back(0);
            }
            outFile.close();
            matched.clear();
        }

        string detail_txt = resultPath + "/details.txt";
        ofstream outFile;
        outFile.open(detail_txt.c_str(), ios::out);
        outFile.setf(ios::fixed, ios::floatfield);
        for (size_t i = 0; i < 8; i++)
        {
            total_num += scene_num[i];
            total_re += scene_re_sum[i];
            total_te += scene_te_sum[i];
            total_rmse += scene_rmse_sum[i];
            total_time += scene_time_sum[i];

            cout << i + 1 << ":" << endl;
            outFile << i + 1 << ":" << endl;
            if (scene_correct_num[i] > 0)
            {
                cout << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
                outFile << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " "
                        << setprecision(4) << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
                cout << "\tRE: " << scene_re_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tRE: " << setprecision(4) << scene_re_sum[i] / scene_correct_num[i] << endl;
                cout << "\tTE: " << scene_te_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tTE: " << setprecision(4) << scene_te_sum[i] / scene_correct_num[i] << endl;
                cout << "\tRMSE: " << scene_rmse_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tRMSE: " << setprecision(6) << scene_rmse_sum[i] / scene_correct_num[i] << endl;
                cout << "\tTime: " << scene_time_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tTime: " << setprecision(6) << scene_time_sum[i] / scene_correct_num[i] << endl;
            }
            else
            {
                outFile << "\tRR: 0/" << scene_num[i] << " 0.0000" << endl;
                outFile << "\tRE: -" << endl;
                outFile << "\tTE: -" << endl;
                outFile << "\tRMSE: -" << endl;
                outFile << "\tTime: -" << endl;
            }
        }
        cout << "total:" << endl;
        outFile << "total:" << endl;
        if (corrected > 0)
        {
            cout << "\tRR: " << corrected / (total_num / 1.0) << endl;
            outFile << "\tRR: " << setprecision(4) << corrected / (total_num / 1.0) << endl;
            cout << "\tRE: " << total_re / corrected << endl;
            outFile << "\tRE: " << setprecision(4) << total_re / corrected << endl;
            cout << "\tTE: " << total_te / corrected << endl;
            outFile << "\tTE: " << setprecision(4) << total_te / corrected << endl;
            cout << "\tRMSE: " << total_rmse / corrected << endl;
            outFile << "\tRMSE: " << setprecision(6) << total_rmse / corrected << endl;
            cout << "\tTime: " << total_time / corrected << endl;
            outFile << "\tTime: " << setprecision(6) << total_time / corrected << endl;
        }
        else
        {
            outFile << "\tRR: 0" << endl;
            outFile << "\tRE: -" << endl;
            outFile << "\tTE: -" << endl;
            outFile << "\tRMSE: -" << endl;
            outFile << "\tTime: -" << endl;
        }
        outFile.close();
    }
    else if (datasetName == "3dlomatch")
    {
        for (size_t i = id; i < 8; i++)
        {
            string analyse_csv = resultPath + "/" + threeDlomatch[i] + "_" + descriptor + ".csv";
            ofstream outFile;
            outFile.open(analyse_csv.c_str(), ios::out);
            outFile.setf(ios::fixed, ios::floatfield);
            outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ','
                    << "inlier_ratio" << ',' << "RMSE" << ',' << "RE" << ',' << "TE" << ',' << "Time" << endl;

            double scene_rmse = 0.0, scene_time = 0.0;
            vector<string> matched = analyse("3dlomatch", resultPath + "/" + threeDlomatch[i],
                                             datasetPath + "/" + threeDlomatch[i], descriptor, outFile,
                                             id, i, scene_rmse, scene_time);
            scene_re_sum.push_back(RE);
            scene_te_sum.push_back(TE);
            scene_rmse_sum.push_back(scene_rmse);
            scene_time_sum.push_back(scene_time);

            if (!matched.empty())
            {
                cout << endl;
                cout << threeDlomatch[i] << ":" << endl;
                for (auto t : matched)
                {
                    cout << "\t" << t << endl;
                }
                cout << endl;
                cout << threeDlomatch[i] << ":" << matched.size() / (scene_num[i] / 1.0) << endl;
                cout << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
                corrected += matched.size();
                total_success_est_rate.push_back(success_estimate_rate);
                scene_correct_num.push_back(matched.size());
            }
            else
            {
                scene_correct_num.push_back(0);
                total_success_est_rate.push_back(0);
            }
            outFile.close();
            matched.clear();
        }

        string detail_txt = resultPath + "/details.txt";
        ofstream outFile;
        outFile.open(detail_txt.c_str(), ios::out);
        outFile.setf(ios::fixed, ios::floatfield);
        for (size_t i = 0; i < 8; i++)
        {
            total_num += scene_num[i];
            total_re += scene_re_sum[i];
            total_te += scene_te_sum[i];
            total_rmse += scene_rmse_sum[i];
            total_time += scene_time_sum[i];

            cout << i + 1 << ":" << endl;
            outFile << i + 1 << ":" << endl;
            if (scene_correct_num[i] > 0)
            {
                cout << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
                outFile << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " "
                        << setprecision(4) << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
                cout << "\tRE: " << scene_re_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tRE: " << setprecision(4) << scene_re_sum[i] / scene_correct_num[i] << endl;
                cout << "\tTE: " << scene_te_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tTE: " << setprecision(4) << scene_te_sum[i] / scene_correct_num[i] << endl;
                cout << "\tRMSE: " << scene_rmse_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tRMSE: " << setprecision(6) << scene_rmse_sum[i] / scene_correct_num[i] << endl;
                cout << "\tTime: " << scene_time_sum[i] / scene_correct_num[i] << endl;
                outFile << "\tTime: " << setprecision(6) << scene_time_sum[i] / scene_correct_num[i] << endl;
            }
            else
            {
                outFile << "\tRR: 0/" << scene_num[i] << " 0.0000" << endl;
                outFile << "\tRE: -" << endl;
                outFile << "\tTE: -" << endl;
                outFile << "\tRMSE: -" << endl;
                outFile << "\tTime: -" << endl;
            }
        }
        cout << "total:" << endl;
        outFile << "total:" << endl;
        if (corrected > 0)
        {
            cout << "\tRR: " << corrected / (total_num / 1.0) << endl;
            outFile << "\tRR: " << setprecision(4) << corrected / (total_num / 1.0) << endl;
            cout << "\tRE: " << total_re / corrected << endl;
            outFile << "\tRE: " << setprecision(4) << total_re / corrected << endl;
            cout << "\tTE: " << total_te / corrected << endl;
            outFile << "\tTE: " << setprecision(4) << total_te / corrected << endl;
            cout << "\tRMSE: " << total_rmse / corrected << endl;
            outFile << "\tRMSE: " << setprecision(6) << total_rmse / corrected << endl;
            cout << "\tTime: " << total_time / corrected << endl;
            outFile << "\tTime: " << setprecision(6) << total_time / corrected << endl;
        }
        else
        {
            outFile << "\tRR: 0" << endl;
            outFile << "\tRE: -" << endl;
            outFile << "\tTE: -" << endl;
            outFile << "\tRMSE: -" << endl;
            outFile << "\tTime: -" << endl;
        }
        outFile.close();
    }
    else if (datasetName == "KITTI")
    {
        int pair_num = 555;
        const string& txt_path = datasetPath;
        string analyse_csv = resultPath + "/KITTI_" + descriptor + ".csv";
        ofstream outFile;
        outFile.open(analyse_csv.c_str(), ios::out);
        outFile.setf(ios::fixed, ios::floatfield);
        outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ','
                << "inlier_ratio" << ',' << "RMSE" << ',' << "RE" << ',' << "TE" << ',' << "Time" << endl;

        vector<string> fail_pair;
        vector<double> time;
        double rmse_sum = 0.0, time_sum = 0.0, re_sum = 0.0, te_sum = 0.0;
        int success_count = 0;

        for (int i = id; i < pair_num; i++)
        {
            time.clear();
            cout << "Pair " << i + 1 << "，total " << pair_num << "，fail " << fail_pair.size() << endl;

            string filename = to_string(i);
            string corr_path = txt_path + "/" + filename + '/' + descriptor + "@corr_n.txt";
            string gt_mat_path = txt_path + "/" + filename + '/' + descriptor + "@gtmat.txt";
            string gt_label_path = txt_path + "/" + filename + '/' + descriptor + "@gtlabel.txt";
            string ov_label = "NULL";
            string folderPath = resultPath + "/" + filename;
            double re, te;
            double inlier_num, total_num;
            double inlier_ratio, success_estimate, total_estimate, average_distance;
            int corrected = registration("KITTI", datasetPath + "/" + std::to_string(i) + "/src_kpts.pcd",
                                         datasetPath + "/" + std::to_string(i) + "/tgt_kpts.pcd",
                                         corr_path, gt_label_path, ov_label, gt_mat_path, folderPath,
                                         re, te, inlier_num, total_num, inlier_ratio,
                                         success_estimate, total_estimate, average_distance, descriptor, time);

            double rmse = average_distance;
            double time_val = time[0];

            if (corrected)
            {
                cout << filename << " Success." << endl;
                re_sum += re;
                te_sum += te;
                rmse_sum += rmse;
                time_sum += time_val;
                success_count++;
            }
            else
            {
                fail_pair.push_back(filename);
                cout << filename << " Fail." << endl;
            }

            outFile << filename << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
            outFile << setprecision(4) << inlier_ratio << ',' << setprecision(6) << rmse << ',';
            outFile << setprecision(6) << re << ',' << setprecision(6) << te << ',';
            outFile << setprecision(6) << time_val << endl;
            cout << endl;
        }
        outFile.close();

        double success_num = pair_num - fail_pair.size();
        cout << "total:" << endl;
        if (success_num > 0)
        {
            cout << "\tRR:" << success_num << "/" << pair_num << " " << success_num / pair_num << endl;
            cout << "\tRE:" << re_sum / success_num << endl;
            cout << "\tTE:" << te_sum / success_num << endl;
            cout << "\tRMSE:" << rmse_sum / success_num << endl;
            cout << "\tTime:" << time_sum / success_num << endl;
        }
        else
        {
            cout << "\tRR: 0" << endl;
        }
        cout << "fail pairs:" << endl;
        for (size_t i = 0; i < fail_pair.size(); i++)
        {
            cout << "\t" << fail_pair[i] << endl;
        }
    }
    else {
        exit(0);
    }
    return 0;
}