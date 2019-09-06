#include <stdio.h>
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
#include <fstream>
#include <Eigen/StdVector>
#include "DBoW2.h"
#include <opencv2/core/eigen.hpp>
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


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace DBoW2;

class tracking_store
{

    public:
    //map for 3d points projections
    unordered_map<int,vector<Point2f>> pt_2d;

    //map for image index of 3d points projections;
    unordered_map<int,vector<int>> img_index;

    //map for match index
    unordered_map<int,vector<int>> match_index;

    //map for initialization of 3d_points
    unordered_map<int,Eigen::Vector3d> init_guess;

    //map for correspondences between vertex identificator and point identificator
    unordered_map<int,int> map_to_custom_struct;

    //map of 3d points
    unordered_map<int,cv::Point3d> triangulated_3d_points;

    //valid_frames_for_triangulation
    vector<int> frames_id;
    vector<int> valid_frames;
    //camera poses
    unordered_map<int,cv::Mat> cam_poses;
    //function to add the projections of the new 3d points for two consecutive frames
   
    //default destructor
    ~tracking_store(){}
};
void matchFeatures(	vector<KeyPoint> &_features1, cv::Mat &_desc1, 
					vector<KeyPoint> &_features2, cv::Mat &_desc2,
					vector<int> &_ifKeypoints, vector<int> &_jfKeypoints,double dst_ratio,double confidence,double reproject_err,
                    double focal_lenght,cv::Point2d pp){

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<vector<DMatch>> matches;
	vector<Point2d> source, destination;
	vector<uchar> mask;
	vector<int> i_keypoint, j_keypoint;
    cv::Mat E;
	matcher->knnMatch(_desc1, _desc2, matches, 2);
	for (unsigned int k = 0; k < matches.size(); k++)
	{
		if (matches[k][0].distance < dst_ratio * matches[k][1].distance)
		{
			source.push_back(_features1[matches[k][0].queryIdx].pt);
			destination.push_back(_features2[matches[k][0].trainIdx].pt);
			i_keypoint.push_back(matches[k][0].queryIdx);
			j_keypoint.push_back(matches[k][0].trainIdx);
		}
	}
    E=findEssentialMat(source,destination,focal_lenght,pp,RANSAC,confidence,reproject_err,mask);
	for (unsigned int m = 0; m < mask.size(); m++)
	{
		if (mask[m])
		{
			_ifKeypoints.push_back(i_keypoint[m]);
			_jfKeypoints.push_back(j_keypoint[m]);
		}
	}
}
void fill_new_3d_points(int map_vertex,cv::Point3d pt,tracking_store &obj)
{
    int custom_structure_id;
    custom_structure_id=obj.map_to_custom_struct[map_vertex];
    obj.triangulated_3d_points[custom_structure_id]=pt;

}
 void add_new_points_proyections(vector<KeyPoint> _features1,vector<KeyPoint> _features2,vector<int> _left_matches,vector<int> _right_matches,int &_point_identifier,int last_frame,int current_frame,Mat used_features,tracking_store &obj)
    {
        for (unsigned int i=0;i<_left_matches.size();i++)
        {
            if(used_features.at<double>(i)==0)
            {
                obj.pt_2d[_point_identifier]=vector<Point2f>();
                obj.pt_2d[_point_identifier].push_back(_features1[_left_matches[i]].pt);
                obj.img_index[_point_identifier]=vector<int>();
                obj.img_index[_point_identifier].push_back(last_frame);
                obj.match_index[_point_identifier]=vector<int>();
                obj.match_index[_point_identifier].push_back(_left_matches[i]);
                obj.pt_2d[_point_identifier].push_back(_features2[_right_matches[i]].pt);
                obj.img_index[_point_identifier].push_back(current_frame);
                obj.match_index[_point_identifier].push_back(_right_matches[i]);
                _point_identifier++;
            }
        }
    }

    //function to add a new projection for an existent point
    void add_new_projection_for_existent_point(int _point_ident,int last_frame,int current_frame,vector<KeyPoint> _features1,vector<KeyPoint> _features2,vector<int> _left_matches,vector<int> _right_matches,Mat &used_points,tracking_store &obj)
    {
        for(int j=0;j<_point_ident;j++)
        {
            auto search_match=obj.match_index.find(j);
            auto search_img=obj.img_index.find(j);
            if(search_match!=obj.match_index.end() && search_img!=obj.img_index.end())
            {
                auto it_match=search_match->second.end();
                it_match--;
                auto it_img=search_img->second.end();
                it_img--;
                int last_match=*it_match;
                int last_img=*it_img;
                int flag=0;
                for(unsigned int k=0;k<_left_matches.size() && !flag;k++)
                {
                    if(_left_matches[k]==last_match && last_img==last_frame)
                    {
                            //we add the new projection for the same 3d point
                            obj.pt_2d[j].push_back(_features2[_right_matches[k]].pt);
                            obj.img_index[j].push_back(current_frame);
                            obj.match_index[j].push_back(_right_matches[k]);
                            used_points.at<double>(k)=1;
                            flag=1;
                    }
                }
            }
        }
    }

    int extract_values(int ident,Eigen::Vector3d &xyz_coordinates,tracking_store &obj)
    {
        auto search_value=obj.init_guess.find(ident);
        if(search_value !=obj.init_guess.end())
        {
            xyz_coordinates=obj.init_guess[ident];
            return 1;
        }
        else
        {
            return 0;
        }
        
    }

    int extract_values(int ident,vector<Point2f> &projections,vector<int> &imgs,tracking_store &obj)
    {
        auto search_value=obj.pt_2d.find(ident);
        if(search_value !=obj.pt_2d.end())
        {
            projections=obj.pt_2d[ident];
            imgs=obj.img_index[ident];
            return 1;
        }
        else
        {
            return 0;
        }
        
    }
    void delete_invalid_points(int img_threshold,int total_points,int &remaining_points,vector<int> &valid_points,tracking_store &obj)
    {
        for(int i=0;i<total_points;i++)
        {
            auto search_point=obj.pt_2d.find(i);
            if(search_point!=obj.pt_2d.end())
            {
                int dimension= search_point->second.size();
                if(dimension>=img_threshold)
                {
                    valid_points.push_back(1);
                    remaining_points++;
                }
                else
                {
                    valid_points.push_back(0);
                }
            }
        }
    }

    void initial_guess_for_3d_points(int total_points,vector<int> valid_points,Eigen::Vector2d principal_point,double focal_length,tracking_store &obj,double z_plane)
    {
        for(int i=0;i<total_points;i++)
        {
            auto search_point=obj.pt_2d.find(i);
            if(search_point!=obj.pt_2d.end())
            {
                if(valid_points[i]==1)
                {
                    vector<Point2f> aux=obj.pt_2d[i];
                    double dimension=aux.size();
                    double  z=z_plane; //initial z invented
                    Eigen::Vector3d init_guess_aux;
                    init_guess_aux << 0.,0.,0.;
                    Eigen::Vector3d value;
                    for (unsigned int j=0;j<aux.size();j++)
                    {
                        init_guess_aux[0]+=(((double)search_point->second[j].x - principal_point[0])/focal_length)*z;
                        init_guess_aux[1]+=(((double)search_point->second[j].y - principal_point[1])/focal_length)*z;
                        init_guess_aux[2]+=z;
                    }
                    value[0]=init_guess_aux[0]/dimension;
                    value[1]=init_guess_aux[1]/dimension;
                    value[2]=init_guess_aux[2]/dimension;
                    obj.init_guess[i]=value;
                }  
            }
        }
    }
    void set_correspondence(int map_ident,int structure_ident,tracking_store &obj)
    {
        obj.map_to_custom_struct[map_ident]=structure_ident;
    }


void displayMatches(	cv::Mat &_img1, std::vector<cv::KeyPoint> &_features1, std::vector<int> &_filtered1,
						cv::Mat &_img2, std::vector<cv::KeyPoint> &_features2, std::vector<int> &_filtered2){
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);

	for(unsigned i = 0; i < _filtered1.size(); i++){
		auto p1 = _features1[_filtered1[i]].pt;
		auto p2 = _features2[_filtered2[i]].pt + cv::Point2f(_img1.cols, 0);
		cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
	}

	cv::imshow("display", display);
	cv::waitKey(3);
}

cv::Mat loadImage(std::string _folder, int _number, cv::Mat &_intrinsics, cv::Mat &_coeffs) {
	stringstream ss;
	ss << _folder << "/left_" << _number << ".png";
	std::cout << "Loading image: " << ss.str() << std::endl;
	Mat image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
	cvtColor(image, image, COLOR_BGR2GRAY);
	cv::Mat image_u;
	undistort(image, image_u, _intrinsics, _coeffs);
	return image_u;
}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}
void loadFeatures(vector<vector<cv::Mat > > &features,int NIMAGES,string path)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << path << "/left_" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
}
double calculate_score(int last_frame,int current_frame,const vector<cv::Mat> &descriptors_left,const vector<cv::Mat> &descriptors_right,string path)
{
    // lets do something with this vocabulary
    // load the vocabulary from disk
    OrbVocabulary voc(path);
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    voc.transform(descriptors_left, v1);
    voc.transform(descriptors_right, v2);
    double score = voc.score(v1, v2);
    cout << "Image " << last_frame << " vs Image " << current_frame << ": " << score << endl;
    return score;
}
void get_rt_from_t(Eigen::Quaterniond &q,Eigen::Vector3d &t,Eigen::Matrix4d pose)
{
    Eigen::Matrix4d pose_inv=pose.inverse();
    Eigen::Matrix3d rot=pose_inv.block<3,3>(0,0);
    Eigen::Vector3d trans=pose_inv.block<3,1>(0,3);
    q=Eigen::Quaterniond(rot);
    t=trans;
}
cv::Mat eigentocv(Eigen::Matrix4d &mat_eigen)
{
    cv::Mat inter=cv::Mat::zeros(4,4,CV_64F);
    inter.at<double>(0,0)=mat_eigen(0,0);
    inter.at<double>(0,1)=mat_eigen(0,1);
    inter.at<double>(0,2)=mat_eigen(0,2);
    inter.at<double>(0,3)=mat_eigen(0,3);
    inter.at<double>(1,0)=mat_eigen(1,0);
    inter.at<double>(1,1)=mat_eigen(1,1);
    inter.at<double>(1,2)=mat_eigen(1,2);
    inter.at<double>(1,3)=mat_eigen(1,3);
    inter.at<double>(2,0)=mat_eigen(2,0);
    inter.at<double>(2,1)=mat_eigen(2,1);
    inter.at<double>(2,2)=mat_eigen(2,2);
    inter.at<double>(2,3)=mat_eigen(2,3);
    inter.at<double>(3,0)=mat_eigen(3,0);
    inter.at<double>(3,1)=mat_eigen(3,1);
    inter.at<double>(3,2)=mat_eigen(3,2);
    inter.at<double>(3,3)=mat_eigen(3,3);
    return inter;
}
void setPointPos(pcl::PointXYZRGB &point,cv::Mat tras)
{
    point.x=tras.at<double>(0);
    point.y=tras.at<double>(1);
    point.z=tras.at<double>(2);
}
void initial_map_generation(tracking_store &obj,int nImages,int img_threshold,int &ident,double focal_length,Eigen::Vector2d principal_point,vector<int> &valid_points,vector<int> &keyframes,int niter,double z_plane)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);
    //filter to reject points that are not visible in more than 3 images
    int remaining_points=0;
    delete_invalid_points(img_threshold,ident,remaining_points,valid_points,obj);
    //we add camera vertices to optimizator
    vector<g2o::SE3Quat,Eigen::aligned_allocator<g2o::SE3Quat> > camera_poses;
    g2o::CameraParameters * cam_params = new g2o::CameraParameters (focal_length, principal_point, 0.);
    cam_params->setId(0);
    optimizer.addParameter(cam_params);
    int vertex_id=0;
    //Eigen::Matrix4d initPose = Eigen::Matrix4d::Identity();
    for(int i=0;i<nImages;i++)
    {
        g2o::VertexSE3Expmap * v_se3=new g2o::VertexSE3Expmap();
        g2o::SE3Quat pose;
        if(i==0)
        { 
            Eigen::Matrix4d poseMatrix=Eigen::Matrix4d::Identity();
            Eigen::Quaterniond qb;
            Eigen::Vector3d tb;
            get_rt_from_t(qb,tb,poseMatrix);
            pose=g2o::SE3Quat(qb,tb);
            v_se3->setId(vertex_id);
            v_se3->setEstimate(pose);
            v_se3->setFixed(true);
            optimizer.addVertex(v_se3);
            camera_poses.push_back(pose);
            vertex_id++;
        }
        else
        {
            pose=camera_poses[0];
            v_se3->setId(vertex_id);
            v_se3->setEstimate(pose);
            optimizer.addVertex(v_se3);
            camera_poses.push_back(pose);
            vertex_id++;
        }   
    } 
    //calculation of initial guess for 3d points
    int point_id=vertex_id;
    initial_guess_for_3d_points(ident,valid_points,principal_point,focal_length,obj,z_plane);
    //we add 3dpoints vertices to optimizator and the edges conecting cameras and points
    
    for(int j=0;j<ident;j++)
    {
        if(valid_points[j]==1)
        {
            Eigen::Vector3d init_guess;
            extract_values(j,init_guess,obj);
            set_correspondence(point_id,j,obj);
            g2o::VertexSBAPointXYZ * v_p= new g2o::VertexSBAPointXYZ();
            v_p->setId(point_id);
            v_p->setMarginalized(true);
            v_p->setEstimate(init_guess);
            optimizer.addVertex(v_p);
            vector<Point2f> aux_pt;
            vector<int> aux_im;
            extract_values(j,aux_pt,aux_im,obj);
            //we search point j on image i
            for(unsigned int p=0;p<aux_im.size();p++)
            {
                //we add the edge connecting the vertex of camera position and the vertex point
                Eigen::Vector2d measurement(aux_pt[p].x,aux_pt[p].y);
                g2o::EdgeProjectXYZ2UV * e= new g2o::EdgeProjectXYZ2UV();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
                for (unsigned int k=0;k<keyframes.size();k++)
                {
                    if(aux_im[p]==keyframes[k])
                    {
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(k)->second));
                    }
                }
                e->setMeasurement(measurement);
                e->information() = Eigen::Matrix2d::Identity();
                e->setParameterId(0, 0);
                optimizer.addEdge(e);
            }
            point_id++;
        }
    }
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(niter);
    optimizer.save("test.g2o");

	for (unsigned int i = 0; i < keyframes.size(); i++)
    {
		Eigen::Affine3f cam_pos;
        g2o::SE3Quat updated_pose;
        Eigen::Matrix4f eig_cam_pos=Eigen::Matrix4f::Identity();
        Eigen::Quaterniond cam_quat;
        Eigen::Vector3d cam_translation;
        g2o::HyperGraph::VertexIDMap::iterator pose_it= optimizer.vertices().find(i);
        g2o::VertexSE3Expmap * v_se3= dynamic_cast< g2o::VertexSE3Expmap * >(pose_it->second);
        updated_pose=v_se3->estimate();
        cam_translation=updated_pose.translation();
        cam_quat=updated_pose.rotation();
        eig_cam_pos.block<3,3>(0,0) = cam_quat.matrix().cast<float>();
        eig_cam_pos.block<3,1>(0,3) = cam_translation.cast<float>();
        cam_pos=eig_cam_pos.inverse();
        Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));
        Eigen::Matrix4d eig_inv=eig_cam_pos.inverse().cast<double>();
        cv::Mat cam_pos_cv=eigentocv(eig_inv);
        obj.cam_poses[i]=cam_pos_cv;
    }
    for(int j=0;j<remaining_points;j++)
    {
        g2o::HyperGraph::VertexIDMap::iterator point_it= optimizer.vertices().find(vertex_id+j);
        g2o::VertexSBAPointXYZ * v_p= dynamic_cast< g2o::VertexSBAPointXYZ * > (point_it->second);
        Eigen::Vector3d p_aux=v_p->estimate();
        cv::Point3d point_cv;
        point_cv.x=p_aux[0];
        point_cv.y=p_aux[1];
        point_cv.z=p_aux[2];
        fill_new_3d_points(vertex_id+j,point_cv,obj);
    }
    optimizer.clear();
}
void displayMatches(cv::Mat &_img1, std::vector<cv::Point2d> &_features1,cv::Mat &_img2, std::vector<cv::Point2d> &_features2)
{
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);
    for(unsigned int i = 0; i < _features1.size(); i++)
    {
		auto p1 = _features1[i];
		auto p2 = _features2[i] + cv::Point2d(_img1.cols, 0);
		cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
	}
    cv::imshow("display", display);
	cv::waitKey(3);
}
void displayMatches(cv::Mat &_img1, std::vector<cv::KeyPoint> &_features1,cv::Mat &_img2, std::vector<cv::KeyPoint> &_features2)
{
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);
    for(unsigned int i = 0; i < _features1.size(); i++)
    {
		auto p1 = _features1[i].pt;
		auto p2 = _features2[i].pt + cv::Point2f(_img1.cols, 0);
		cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
	}
    cv::imshow("display", display);
	cv::waitKey(3);
}
void pixel_to_cam_plane(Point2d &pixel_plane,Point2d &cam_plane,cv::Mat &intrinsic)
{
    cam_plane.x=(pixel_plane.x-intrinsic.at<double>(0,2))/intrinsic.at<double>(0,0);
    cam_plane.y=(pixel_plane.y-intrinsic.at<double>(1,2))/intrinsic.at<double>(1,1);
}
cv::Mat generate_projection_matrix(cv::Mat &R,cv::Mat &t)
{
    cv::Mat inter;
    hconcat(R,t,inter);
    return inter;
}
cv::Mat generate_4x4_transformation(cv::Mat &R,cv::Mat &t)
{
    Mat T = (Mat_<double>(4, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
             R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0),
             0, 0, 0, 1);
    return T;
}
void relative_triangulation(vector<Point2d> triang_left,vector<Point2d> triang_right,cv::Mat intrinsic,
cv::Mat R_2_to_1,cv::Mat t_2_to_1,vector<Point3d> &pts3d)
{
    vector<Point2d> normalized_left,normalized_right;
    for(unsigned int i=0;i<triang_left.size();i++)
    {
        cv::Point2d left,right;
        pixel_to_cam_plane(triang_left[i],left,intrinsic);
        normalized_left.push_back(left);
        pixel_to_cam_plane(triang_right[i],right,intrinsic);
        normalized_right.push_back(right);
    }
    cv::Mat left_project_mat=cv::Mat::eye(3,4,CV_64F);
    cv::Mat right_project_mat=generate_projection_matrix(R_2_to_1,t_2_to_1);
    cv::Mat point3d_homo;
    triangulatePoints(left_project_mat,right_project_mat,normalized_left,normalized_right,point3d_homo);
    pts3d.clear();
    for( int i=0;i<point3d_homo.cols;i++)
    {
        Point3d aux;
        aux.x=(point3d_homo.col(i).at<double>(0)/point3d_homo.col(i).at<double>(3));
        aux.y=(point3d_homo.col(i).at<double>(1)/point3d_homo.col(i).at<double>(3));
        aux.z=(point3d_homo.col(i).at<double>(2)/point3d_homo.col(i).at<double>(3));
        pts3d.push_back(aux);
    }
}
void search_existent_points(int ident,int last_frame,vector<int> ref_match_idx,vector<int> &used_points,vector<int> &identifiers,tracking_store obj)
{
    for(unsigned int i=0;i<ref_match_idx.size();i++)
    {   
        int stop_flag=0;
        for(int j=0;j<ident && !stop_flag;j++)
        {
            auto search_match=obj.match_index.find(j);
            auto search_img=obj.img_index.find(j);
            if(search_match!=obj.match_index.end() && search_img!=obj.img_index.end())
            {
                auto it_match=search_match->second.end();
                it_match--;
                auto it_img=search_img->second.end();
                it_img--;
                int last_match=*it_match;
                int last_img=*it_img;
                if(ref_match_idx[i]==last_match && last_img==last_frame)
                {
                    used_points.push_back(1);
                    identifiers.push_back(j);
                    stop_flag=1;
                }
            }
        }
        if(!stop_flag)
        {
            used_points.push_back(0);
            identifiers.push_back(0);
        }   
    }
}
void update_scale(vector<Point3d> existing,vector<Point3d> corresponding,double &scale)
{
    vector<double> scales;
    for (size_t j=0; j < existing.size()-1; j++)
    {
      for (size_t k=j+1; k< existing.size(); k++)
      {
        double s = norm(existing[j] - existing[k]) / norm(corresponding[j] - corresponding[k]);
        scales.push_back(s);
      }
    }
    sort(scales.begin(),scales.end());
    int n=scales.size();
    if (n % 2 != 0) scale=scales[n/2];
    else scale=(scales[(n-1)/2] + scales[n/2])/2.0;
}
cv::Point3d change_points_to_other_ref_system(const cv::Point3d &old_pt,cv::Mat &curr_pos)
{
    const Mat &T = curr_pos;
    double p[4] = {old_pt.x, old_pt.y, old_pt.z, 1};
    double res[3] = {0, 0, 0};
    for (int row = 0; row < 3; row++)
    {
        for (int j = 0; j < 4; j++)
            res[row] += T.at<double>(row, j) * p[j];
    }
    return Point3d(res[0], res[1], res[2]);
}
void add_new_3d_point(int last_frame,int current_frame,cv::Point3d pt,int last_match,int curr_match,cv::Point2d left_project,cv::Point2d right_project,int &ident,tracking_store &obj)
{
    
    obj.triangulated_3d_points[ident]=pt;
    obj.img_index[ident].push_back(last_frame);
    obj.img_index[ident].push_back(current_frame);
    obj.pt_2d[ident].push_back(cv::Point2f((float)left_project.x,(float)left_project.y));
    obj.pt_2d[ident].push_back(cv::Point2f((float)right_project.x,(float)right_project.y));
    obj.match_index[ident].push_back(last_match);
    obj.match_index[ident].push_back(curr_match);
    ident++;
}
void local_optimization(int window_size,tracking_store &obj,int niter,int ident,double focal_length,Eigen::Vector2d principal_point)
{
    g2o::SparseOptimizer opt;
    opt.setVerbose(false);
    unordered_map<int,int> opt_to_custom;
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    opt.setAlgorithm(solver);
    //we add camera vertices to optimizator
    vector<g2o::SE3Quat,Eigen::aligned_allocator<g2o::SE3Quat> > camera_poses;
    g2o::CameraParameters * cam_params = new g2o::CameraParameters (focal_length, principal_point, 0.);
    cam_params->setId(0);
    opt.addParameter(cam_params);
    //search the "window_size" last valid_frames
    int dim=0;
    vector<Eigen::Matrix4d> inter_cam_poses;
    vector<int> identifiers;
    for(unsigned int j=(obj.valid_frames.size())-1;j>=0 && dim<window_size;j--)
    {
        if(obj.valid_frames[j]==1)
        {
            cv::Mat pos_cv=obj.cam_poses[obj.frames_id[j]];
            dim++;
            Eigen::Matrix4d eig_pos;
            cv2eigen(pos_cv,eig_pos);
            inter_cam_poses.push_back(eig_pos);
            identifiers.push_back(obj.frames_id[j]);
        }
    }
    int vertex_id=0;
    //Eigen::Matrix4d initPose = Eigen::Matrix4d::Identity();
    for(int i=1;i<=window_size;i++)
    {
        g2o::VertexSE3Expmap * v_se3=new g2o::VertexSE3Expmap();
        g2o::SE3Quat pose;
        Eigen::Matrix4d poseMatrix=inter_cam_poses[window_size-i];
        if(i!=window_size)
        {
            Eigen::Quaterniond qb;
            Eigen::Vector3d tb;
            get_rt_from_t(qb,tb,poseMatrix);
            pose=g2o::SE3Quat(qb,tb);
            v_se3->setId(vertex_id);
            v_se3->setEstimate(pose);
            v_se3->setFixed(true);
            opt.addVertex(v_se3);
            camera_poses.push_back(pose);
            vertex_id++;
        }
        else
        {
            Eigen::Quaterniond qb;
            Eigen::Vector3d tb;
            get_rt_from_t(qb,tb,poseMatrix);
            pose=g2o::SE3Quat(qb,tb);
            v_se3->setId(vertex_id);
            v_se3->setEstimate(pose);
            opt.addVertex(v_se3);
            camera_poses.push_back(pose);
            vertex_id++;
        }   
    } 
    //calculation of initial guess for 3d points
    int point_id=vertex_id;
    //we add 3dpoints vertices to optimizator and the edges conecting cameras and points
    for(int j=0;j<ident;j++)
    {
        int not_yet=0;
        g2o::VertexSBAPointXYZ * v_p= new g2o::VertexSBAPointXYZ();
        for(int p=1;p<=window_size;p++)
        {
            int img=identifiers[window_size-p];
            vector<int> point_vis=obj.img_index[j];
            vector<Point2f> projections=obj.pt_2d[j];
            int stop_flag=0;
            for(unsigned int k=0;k<point_vis.size() && !stop_flag;k++)
            {
                if(point_vis[k]==img)
                {
                    if(!not_yet)
                    {
                        cv::Point3d actual_value=obj.triangulated_3d_points[j];
                        Eigen::Vector3d init_guess;
                        init_guess[0]=actual_value.x;
                        init_guess[1]=actual_value.y;
                        init_guess[2]=actual_value.z;
                        opt_to_custom[point_id]=j;
                        v_p->setId(point_id);
                        v_p->setMarginalized(true);
                        v_p->setEstimate(init_guess);
                        opt.addVertex(v_p);
                        not_yet=1;
                        point_id++;
                    }
                    //we add the edge connecting the vertex of camera position and the vertex point
                    Eigen::Vector2d measurement(projections[k].x,projections[k].y);
                    g2o::EdgeProjectXYZ2UV * e= new g2o::EdgeProjectXYZ2UV();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(opt.vertices().find(p-1)->second));
                    e->setMeasurement(measurement);
                    e->information() = Eigen::Matrix2d::Identity(); 
                    e->setParameterId(0, 0);
                    opt.addEdge(e);
                    stop_flag=1;
                }
            }
        }
    }
    opt.initializeOptimization();
    opt.setVerbose(true);
    opt.optimize(niter);

	for (int i = 1; i <= window_size; i++)
    {
		Eigen::Affine3f cam_pos;
        g2o::SE3Quat updated_pose;
        Eigen::Matrix4f eig_cam_pos=Eigen::Matrix4f::Identity();
        Eigen::Quaterniond cam_quat;
        Eigen::Vector3d cam_translation;
        g2o::HyperGraph::VertexIDMap::iterator pose_it= opt.vertices().find(i-1);
        g2o::VertexSE3Expmap * v_se3= dynamic_cast< g2o::VertexSE3Expmap * >(pose_it->second);
        updated_pose=v_se3->estimate();
        cam_translation=updated_pose.translation();
        cam_quat=updated_pose.rotation();
        eig_cam_pos.block<3,3>(0,0) = cam_quat.matrix().cast<float>();
        eig_cam_pos.block<3,1>(0,3) = cam_translation.cast<float>();
        cam_pos=eig_cam_pos.inverse();
        Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));
        Eigen::Matrix4d eig_inv=eig_cam_pos.inverse().cast<double>();
        cv::Mat cam_pos_cv=eigentocv(eig_inv);
        obj.cam_poses[identifiers[window_size-i]]=cam_pos_cv;
    }
    for(int j=vertex_id;j<point_id;j++)
    {
        g2o::HyperGraph::VertexIDMap::iterator point_it= opt.vertices().find(j);
        g2o::VertexSBAPointXYZ * v_p= dynamic_cast< g2o::VertexSBAPointXYZ * > (point_it->second);
        Eigen::Vector3d p_aux=v_p->estimate();
        cv::Point3d point_cv;
        point_cv.x=p_aux[0];
        point_cv.y=p_aux[1];
        point_cv.z=p_aux[2];
        obj.triangulated_3d_points[opt_to_custom[j]]=point_cv;
    }
}
void update_3d_point(tracking_store &obj,int identifier,int match_id,int current_frame,cv::Point2d projection,cv::Point3d pt)
{
    obj.triangulated_3d_points[identifier]=pt;
    //obj.triangulated_3d_points[identifier]/=2;
    obj.match_index[identifier].push_back(match_id);
    obj.pt_2d[identifier].push_back(cv::Point2f((float)projection.x,(float)projection.y));
    obj.img_index[identifier].push_back(current_frame);
}
void matchFeatures_and_compute_essential(cv::Mat &img1,cv::Mat &img2, int last_frame,int current_frame,vector<KeyPoint> _features1,vector<KeyPoint> _features2,
                                    cv::Mat _desc1,cv::Mat _desc2,
                                    vector<Point2f> &corresponding_left,vector<Point2f> &corresponding_right,
                                    vector<int> &left_index,vector<int> &right_index,
                                    cv::Mat &mask,double dst_ratio,cv::Mat &E,
                                    double focal_lenght,double confidence,double reproject_err,Point2d pp,tracking_store &obj)
{
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<vector<DMatch>> matches;
	matcher->knnMatch(_desc1, _desc2, matches, 2);
    displayMatches(img1,_features2,img2,_features2);
	for (unsigned int k = 0; k < matches.size(); k++)
	{
		if (matches[k][0].distance < dst_ratio * matches[k][1].distance)
		{ 
			corresponding_left.push_back(_features1[matches[k][0].queryIdx].pt);
			corresponding_right.push_back(_features2[matches[k][0].trainIdx].pt);
			left_index.push_back(matches[k][0].queryIdx);
			right_index.push_back(matches[k][0].trainIdx);
		}
	}
    E=findEssentialMat(corresponding_left,corresponding_right,focal_lenght,pp,RANSAC,confidence,reproject_err,mask);
}
int estimate_motion_and_calculate_3d_points(int last_frame,int current_frame,cv::Mat &img1,cv::Mat &img2,
                                            vector<Point2f> points_left,vector<Point2f> points_right,
                                            vector<int> index_left,vector<int> index_right,int &ident,
                                            cv::Mat &intrinsic,cv::Mat &E,double focal_lenght,Point2d pp,
                                            double scale,cv::Mat &mask,vector<Point3d> &pts3d,tracking_store &obj,vector<int> &valid_points,vector<Point3d> &new_points)
{
    vector<Point2d> triangulation_points_left,triangulation_points_right;
    vector<int> inliers_recover_left,inliers_recover_right;
    Mat R,t;
    if((int)points_left.size() < 8)
    {
        return 0;
    }
    recoverPose(E,points_left,points_right,R,t,focal_lenght,pp,mask);
    if(E.cols!=3 || E.rows!=3)
    {
        return 0;
    }
    for(int i=0;i<mask.rows;i++)
    {
        if(mask.at<unsigned char>(i))
        {
            triangulation_points_left.push_back(Point2d((double)points_left[i].x,(double)points_left[i].y));
            triangulation_points_right.push_back(Point2d((double)points_right[i].x,(double)points_right[i].y));
            inliers_recover_left.push_back(index_left[i]);
            inliers_recover_right.push_back(index_right[i]);
        }
    }
    displayMatches(img1,triangulation_points_left,img2,triangulation_points_right);
    //do tracking
    cv::Mat last_pose=obj.cam_poses[last_frame];
    cv::Mat curr_rel_motion=generate_4x4_transformation(R,t);
    cv::Mat new_camera_pose=last_pose*(curr_rel_motion.inv());
    relative_triangulation(triangulation_points_left,triangulation_points_right,intrinsic,R,t,pts3d);
    //map update && store camera_pose
    vector<int> identifiers,used_features;
    search_existent_points(ident,last_frame,inliers_recover_left,used_features,identifiers,obj);
    vector<Point3d> existent_3d,corresponding_3d;
    for(unsigned int i=0;i<used_features.size();i++)
    {
        if(used_features[i]==1)
        {
            existent_3d.push_back(obj.triangulated_3d_points[identifiers[i]]);
            corresponding_3d.push_back(pts3d[i]);
        }    
    }
    update_scale(existent_3d,corresponding_3d,scale);
    t*=scale;
    relative_triangulation(triangulation_points_left,triangulation_points_right,intrinsic,R,t,pts3d);
    vector<Point3d> pts3d_in_world,new_points_for_cloud;
    for(unsigned int i=0;i<pts3d.size();i++)
    {
        cv::Point3d aux_point=change_points_to_other_ref_system(pts3d[i],last_pose);
        pts3d_in_world.push_back(aux_point);
    }
    curr_rel_motion=generate_4x4_transformation(R,t);
    new_camera_pose=last_pose*(curr_rel_motion.inv());
    obj.cam_poses[current_frame]=new_camera_pose;
    for(unsigned int i=0;i<used_features.size();i++)
    {
        if(used_features[i]==0)
        {
            add_new_3d_point(last_frame,current_frame,pts3d_in_world[i],inliers_recover_left[i],inliers_recover_right[i],
            triangulation_points_left[i],triangulation_points_right[i],ident,obj);
            valid_points.push_back(1);
            new_points_for_cloud.push_back(pts3d_in_world[i]);
        }
        else
        {
            update_3d_point(obj,identifiers[i],inliers_recover_right[i],current_frame,triangulation_points_right[i],pts3d_in_world[i]);
        }     
    }
    obj.valid_frames.push_back(1);
    obj.frames_id.push_back(current_frame);
    return 1;
}