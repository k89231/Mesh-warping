#include<iostream>
#include<opencv2\highgui.hpp>
#include<opencv2\core.hpp>
#include<opencv2\imgproc.hpp>
#include<fstream>
#include<vector>
#include<Windows.h>
#include"Mapping.h"
#include<thread>
using namespace std;
using namespace cv;
const int Marksnum = 83;
Mat src, dst;
vector<Point2f> srclandmark, dstlandmark;
vector<Point2f> midlandmark_l2r, midlandmark_r2l;
void Init_Vec();
void Cut(vector<Vec6f> &t);
bool prepareData();
static void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color);
Subdiv2D subdiv_mid_l2r, subdiv_mid_r2l; 
Subdiv2D subdiv_src; 
Subdiv2D subdiv_dst; 
vector<Vec6f> triangleList_mid_l2r, triangleList_mid_r2l, triangleList_src, triangleList_dst;
int main()
{
	src = imread("demo1.jpg");
	dst = imread("demo2.jpg");
	if ((!src.data) || (!dst.data))
		return -1;
	double ratio[] = { 0, 0.2, 0.4, 0.5, 0.8, 1.0 };
	int height = src.rows;
	int width = src.cols;
	imshow("源1", src);
	imshow("源2", dst);
	if (!prepareData())
		return -1;
	for (int index = 0; index < 6; index++)
	{
		midlandmark_l2r.clear();
		Rect rect(0, 0, src.cols, src.rows);
		subdiv_mid_l2r.initDelaunay(rect);
		for (int i = 0; i < srclandmark.size(); i++)
		{
			float x, y;
			x = (1 - ratio[index]) * srclandmark[i].x + ratio[index] * dstlandmark[i].x;
			y = (1 - ratio[index]) * srclandmark[i].y + ratio[index] * dstlandmark[i].y;
			midlandmark_l2r.push_back(Point2f(x, y));
		}
		for (vector<Point2f>::iterator it = midlandmark_l2r.begin(); it != midlandmark_l2r.end(); it++)
		{
			subdiv_mid_l2r.insert(*it);
		}
		subdiv_mid_l2r.getTriangleList(triangleList_mid_l2r);
		Cut(triangleList_mid_l2r);
		Mat result_l2r = src.clone();          //warping from left 2 right
		Mat MLSresult_l2r = src.clone();       //warping using MLS from left 2 right
		Mat result_r2l = dst.clone();			//warping from right 2 left
		Mat MLSresult_r2l = dst.clone();       //warping using MLS from right 2 left
		Mat mixed(height, width, src.type());
		mapping(result_l2r, MLSresult_l2r, triangleList_src, triangleList_mid_l2r, height, width, src, srclandmark, midlandmark_l2r);
		mapping(result_r2l, MLSresult_r2l, triangleList_dst, triangleList_mid_l2r, height, width, dst, dstlandmark, midlandmark_l2r);
		mixed = (1.0-ratio[index])*result_l2r + ratio[index]*result_r2l;
		string name1;  string name2; string name3;
		name1.append("l2r"); name1.append(to_string(index)); //name1.append(".jpg");
		name2.append("mixed"); name2.append(to_string(index)); //name2.append(".jpg");
		name3.append("r2l"); name3.append(to_string(index)); //name3.append(".jpg");
		namedWindow(name1, WINDOW_AUTOSIZE);
		imshow(name1, result_l2r);
		namedWindow(name2, WINDOW_AUTOSIZE);
		imshow(name2, mixed);
		namedWindow(name3, WINDOW_AUTOSIZE);
		imshow(name3, result_r2l);
		//draw_subdiv(src, subdiv_src, Scalar(255, 255, 255));
		waitKey(1);
	}
	waitKey(0);
	return 0;
}

bool prepareData()
{
	ifstream srcf, dstf;
	srcf.open("demo1.jpg.txt");
	dstf.open("demo2.jpg.txt");
	if ((srcf.fail()) || (dstf.fail()))
		return false;
	Init_Vec();

	for (int i = 0; i < Marksnum; i++)
	{
		float x, y;
		srcf >> x;
		srcf >> y;
		srclandmark.push_back(Point2f(x / 100 * src.cols, y / 100 * src.rows));

		dstf >> x;
		dstf >> y;
		dstlandmark.push_back(Point2f(x / 100 * dst.cols, y / 100 * dst.rows));
	}
	Rect rect(0, 0, src.cols, src.rows);
	subdiv_src.initDelaunay(rect);
	subdiv_dst.initDelaunay(rect);
	for (vector<Point2f>::iterator it = srclandmark.begin(); it != srclandmark.end(); it++)
	{
		subdiv_src.insert(*it);
	}
	for (vector<Point2f>::iterator it = dstlandmark.begin(); it != dstlandmark.end(); it++)
	{
		subdiv_dst.insert(*it);
	}
	subdiv_src.getTriangleList(triangleList_src);
	subdiv_dst.getTriangleList(triangleList_dst);
	Cut(triangleList_src);
	Cut(triangleList_dst);
}
void Init_Vec()//将四个顶点、四条边的中点作为特征点加入
{
	float height = (float)src.rows;
	float width = (float)src.cols;
	srclandmark.push_back(Point_<float>(0, 0));
	srclandmark.push_back(Point_<float>(width - 1, 0));
	srclandmark.push_back(Point_<float>(0, height - 1));
	srclandmark.push_back(Point_<float>(width - 1, height - 1));
	
	dstlandmark.push_back(Point_<float>(0, 0));
	dstlandmark.push_back(Point_<float>(width - 1, 0));
	dstlandmark.push_back(Point_<float>(0, height - 1));
	dstlandmark.push_back(Point_<float>(width - 1, height - 1));

	srclandmark.push_back(Point_<float>(width / 2, 0));
	srclandmark.push_back(Point_<float>(0, height / 2));
	srclandmark.push_back(Point_<float>(width - 1, height / 2));
	srclandmark.push_back(Point_<float>(width / 2, height - 1));

	dstlandmark.push_back(Point_<float>(width / 2, 0));
	dstlandmark.push_back(Point_<float>(0, height / 2));
	dstlandmark.push_back(Point_<float>(width - 1, height / 2));
	dstlandmark.push_back(Point_<float>(width / 2, height - 1));
}
void Cut(vector<Vec6f> &t)
{
	cout << t.size();
	vector<Vec6f>::iterator it = t.begin();
	t.erase(it + 17);
	t.erase(it + 14, it + 16);
	t.erase(it + 10);
	t.erase(it + 7, it + 9);
	t.erase(it, it + 6);
}
static void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));//对double类型四舍五入，t为Vec6f类型
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		bool draw = true;
		for (int i = 0; i<3; i++) {
			if (pt[i].x>img.cols || pt[i].y>img.rows || pt[i].x<0 || pt[i].y<0)
				draw = false;
		}
		if (draw) {
			line(img, pt[0], pt[1], delaunay_color, 1);
			line(img, pt[1], pt[2], delaunay_color, 1);
			line(img, pt[2], pt[0], delaunay_color, 1);
		}
	}
}