#include <stdio.h>
#include "functions.h"
#include <iostream>
#include <ctype.h>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>
#include <string>
#include <sstream>
#include<stdlib.h>
#include <unistd.h>
#include <ctime>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <sys/time.h>
#include <fstream>
#include <Eigen/StdVector>
#include <opencv2/core/eigen.hpp>
#include "DBoW2.h"
#include <time.h>
#include <stdio.h>
#include <unordered_set>
#include <stdint.h>
#include <algorithm>
#include <iterator> 
#include <vector>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/config.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/timeutil.h"
#include "g2o/stuff/macros.h"
#include "g2o/stuff/misc.h"
#include "g2o/config.h"

#include "g2o/core/ownership.h"

int main(int argc,char ** argv)
{
    if(argc<6)
    {
        cout<< "bad input\n";
        cout << "please enter:\n";
        cout << "argv[1]= path to rgb images. the images must be called left_i being i the image number\n";
        cout << "argv[2]= number of images to compose the initial map\n";
        cout << "argv[3]= number of dataset images\n";
        cout<<  "argv[4]= path to vocabulary\n";
        cout << "argv[5]= flag to enable local optimization(1 to enable, 0 to disable)\n";
        exit(-1);
    }
    Mat distcoef = (Mat_<float>(1, 5) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	Mat distor = (Mat_<float>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	distor.convertTo(distor, CV_64F);
	Mat intrinsic = (Mat_<float>(3, 3) << 517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
	intrinsic.convertTo(intrinsic, CV_64F);
	distcoef.convertTo(distcoef, CV_64F);
    double focal_length=516.9;
    cv::Point2d pp=cv::Point2d(318.6,255.3);
    Eigen::Vector2d principal_point(318.6,255.3); 
    //number of images to compose the initial map
    int nImages =  atoi(argv[2]);
    //number of dataset images
    int total_images;
    sscanf(argv[3],"%d",&total_images);
    //flag to enable local optimization
    int use_local_opt;
    sscanf(argv[5],"%d",&use_local_opt);
    //identifier for 3d_point
    int ident=0;
    //threshold to detect points tha are seen in more than img_threshol images
    int img_threshold=3;
    //threshold for the first filter to recject bad matches between features
    double dst_ratio=0.7;
    //variables for the RANSAC algorithm
    double confidence=0.999;
    double reproject_err=1.0;
    //g2o iterations
    int niter=50;
    //threshold to compare the histograms of the appearance of vocabulary words in the image
    double dbow2_threshold=0.18;
    //number of keyframes to be used by the local optimization module
    int window_size=20;
    //initial depth for the three-dimensional points of the environment
    double z_plane=1.5;
    //first index of dataset
    int current_frame,last_frame;
    last_frame=0;
    //we need variables to store the last image and the last features & descriptors
    auto pt =ORB::create();
    Mat foto1_u;
    vector<KeyPoint> features1;
    Mat descriptors1;
    //initialize viewer and pointcloud
    pcl::visualization::PCLVisualizer viewer("Viewer");
	viewer.setBackgroundColor(0.35, 0.35, 0.35);
    viewer.initCameraParameters();
    pcl::PointCloud<pcl::PointXYZ> cloud;
    //custom class to store matching between images
    tracking_store tracks;
    //load first image who will be the first keyframe
    foto1_u = loadImage(argv[1], current_frame, intrinsic, distcoef);
    pt->detectAndCompute(foto1_u, Mat(), features1, descriptors1);
    current_frame=last_frame;
    vector<int> keyframes;
    vector<int> valid_points;
    keyframes.push_back(current_frame);
    tracks.valid_frames.push_back(1);
    tracks.frames_id.push_back(current_frame);
    current_frame++;
    while(current_frame<nImages)
    {   
        //load new image
        Mat foto2_u = loadImage(argv[1], current_frame, intrinsic, distcoef);
        //create pair of features
        vector<KeyPoint> features2;
	    Mat descriptors2;
	    pt->detectAndCompute(foto2_u, Mat(), features2, descriptors2);
        vector<int> left_index_matches, right_index_matches;
        matchFeatures(features1, descriptors1, features2, descriptors2, left_index_matches, right_index_matches,
		      dst_ratio,confidence,reproject_err,focal_length,pp);
        displayMatches(foto1_u, features1, left_index_matches,foto2_u, features2, right_index_matches);
        Mat used_features=Mat::zeros(1,int(left_index_matches.size()),CV_64F);
        if(ident>0)
        {
            add_new_projection_for_existent_point(ident,last_frame,current_frame,features1,features2,
						  left_index_matches,right_index_matches,used_features,tracks);
        }
        add_new_points_proyections(features1,features2,left_index_matches,right_index_matches,ident,last_frame,current_frame,
				   used_features,tracks);
        foto1_u=foto2_u;
        features1=features2;
        descriptors1=descriptors2;
        used_features.release();
        last_frame=current_frame;
        keyframes.push_back(current_frame);
        tracks.valid_frames.push_back(1);
        tracks.frames_id.push_back(current_frame);
        current_frame++;
    }
    //prepare varibles for g2o optimization
    initial_map_generation(tracks,nImages,img_threshold,ident,focal_length,principal_point,valid_points,keyframes,niter,z_plane);
    int last_found=0;
    double score=1;
    while(current_frame<total_images || !last_found)
    {
        vector<KeyPoint> features2;
        cv::Mat descriptors2;
        cv::Mat foto2_u=loadImage(argv[1],current_frame,intrinsic,distcoef);
        pt->detectAndCompute(foto2_u,cv::Mat(),features2,descriptors2);
        vector<cv::Mat> left_descriptors,right_descriptors;
        changeStructure(descriptors1,left_descriptors);
        changeStructure(descriptors2,right_descriptors);
        score=calculate_score(last_frame,current_frame,left_descriptors,right_descriptors,argv[4]);
        if(score<= dbow2_threshold)
        {
            //keyframe found. Let's match features between them
            vector<Point2f> left_points,right_points;
            vector<int> left_idx,right_idx;
            cv::Mat mask;
            cv::Mat E;
            int res;
            matchFeatures_and_compute_essential(foto1_u,foto2_u,last_frame,current_frame,features1,features2,descriptors1,descriptors2
            ,left_points,right_points,left_idx,right_idx,mask,dst_ratio,
            E,focal_length,confidence,reproject_err,pp,tracks);
            double scale=1;
            vector<Point3d> triangulated_points,new_points;
            res=estimate_motion_and_calculate_3d_points(last_frame,current_frame,foto1_u,foto2_u,left_points,
							right_points,left_idx,right_idx,ident,intrinsic,E,
							focal_length,pp,scale,mask,triangulated_points,tracks,valid_points,new_points);
            if(use_local_opt)
            {
                local_optimization(window_size,tracks,niter,ident,focal_length,principal_point);
            }
            if(res==1)
            {
                score=1;
                last_frame=current_frame;
                foto1_u=foto2_u;
                descriptors1=descriptors2;
                features1=features2;
                last_found=1;
                current_frame++;
            }
            else
            {
                tracks.valid_frames.push_back(0);
                tracks.frames_id.push_back(current_frame);
                last_found=0;
                current_frame++;
            }
        }
        else
        {
            tracks.valid_frames.push_back(0);
            tracks.frames_id.push_back(current_frame);
            last_found=0;
            current_frame++;
        }
    }
    mkdir("./lectura_datos", 0777);
    std::ofstream file("./lectura_datos/odometry.txt");
    if (!file.is_open()) return -1;
    cv::Mat last_t,curr_t;
	for (unsigned int i = 0; i < tracks.valid_frames.size(); i++)
    {
        if(tracks.valid_frames[i]==1)
        {
            stringstream sss;
		    string name;
		    sss << tracks.frames_id[i];
		    name = sss.str();
		    Eigen::Affine3f cam_pos;
            Eigen::Matrix4d eig_cam_pos=Eigen::Matrix4d::Identity();
            Eigen::Vector3d cam_translation;
            Eigen::Matrix3d cam_rotation;
            cv::Mat cv_cam_rot=cv::Mat::zeros(3,3,CV_64F);
            cv::Mat cv_cam_tras=cv::Mat::zeros(3,1,CV_64F);
            cv_cam_rot=tracks.cam_poses[i].rowRange(0,3).colRange(0,3);
            cv_cam_tras=tracks.cam_poses[i].rowRange(0,3).col(3);
            cv2eigen(cv_cam_rot,cam_rotation);
            cv2eigen(cv_cam_tras,cam_translation);
            eig_cam_pos.block<3,3>(0,0)=cam_rotation;
            eig_cam_pos.block<3,1>(0,3) = cam_translation;
            cam_pos=eig_cam_pos.cast<float>();
            viewer.addCoordinateSystem(0.05, cam_pos, name);
		    pcl::PointXYZ textPoint(cam_pos(0,3), cam_pos(1,3), cam_pos(2,3));
		    viewer.addText3D(std::to_string(i), textPoint, 0.01, 1, 1, 1, "text_"+std::to_string(i));
            Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));
            file << tracks.frames_id[i] << " " << cam_pos(0,3) << " " <<  cam_pos(1,3) << " " << cam_pos(2,3) << " " << 
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
    }
    file.close();
	
    for(int j=0;j<ident;j++)
    {
        if(valid_points[j]==1)
        {
            pcl::PointXYZ p(tracks.triangulated_3d_points[j].x, tracks.triangulated_3d_points[j].y,tracks.triangulated_3d_points[j].z);
            cloud.push_back(p);
        }
    }
	viewer.addPointCloud<pcl::PointXYZ>(cloud.makeShared(), "map");
	while (!viewer.wasStopped()) {
		viewer.spin();
	}
	return 0;
}
