#pragma once
#include<iostream>
#include<opencv2\highgui.hpp>
#include<opencv2\core.hpp>
#include<opencv2\imgproc.hpp>
#include<fstream>
#include<vector>
#include<Windows.h>
using namespace std;
using namespace cv;

float S(float x_)
{
	float x = abs(x_);
	if (x <= 1)
		return (1 - 2 * x*x + x*x*x);
	else if ((x > 1) && (x < 2))
		return (4 - 8 * x + 5 * x*x - x*x*x);
	else
		return 0;
}

// 判断(x, y), (xx, yy)是否在由(x1, y1)和(x2, y2)组成的直线的同一侧
bool Issameside(float const x, float const y,
	float const xx, float const yy,
	float const x1, float const y1,
	float const x2, float const y2) 
{
	float const
		dx21 = x2 - x1, dy21 = y2 - y1,
		dx = x - x1, dy = y - y1,
		dxx = xx - x1, dyy = yy - y1,
		tmp = dxx*dy21 - dyy*dx21;
	return (dx*dy21 - dy*dx21 >= 0) == (tmp >= 0);
}
Mat_<float> AffineMatrix(const Mat_<float> raw, const Mat_<float> dst)
{
	//cout << raw*dst.inv();
	return dst*raw.inv();
}
Point2f MLS(const vector<Point2f> src, const vector<Point2f> mid, float j, float i)//返回MLS仿射映射到原图的坐标
{
	Point2f pstar, qstar, dv;
	float sw;
	int nPoint = src.size();
	vector<float> w;
	w.resize(nPoint);

	int k;

	sw = 0;                                 
	pstar.x = pstar.y = 0;
	qstar.x = qstar.y = 0;

	for (k = 0; k < nPoint; k++) {                
		if (j == mid[k].x && i == mid[k].y)
			continue;

		w[k] = 1 / ((j - mid[k].x)*(j - mid[k].x) + (i - mid[k].y)*(i - mid[k].y));
		sw += w[k];
		pstar += w[k] * mid[k];
		qstar += w[k] * src[k];
	}
	pstar *= 1/sw;                   //求得p_*和q_*                      
	qstar *= 1/sw;

	Mat M = Mat::zeros(2, 2, CV_32FC1);
	Mat P(1, 2, CV_32FC1), Q(1, 2, CV_32FC1);
	Mat A = Mat::zeros(2, 2, CV_32FC1), B = Mat::zeros(2, 2, CV_32FC1);
	for (k = 0; k < nPoint; k++) {
		if (j == mid[k].x && i == mid[k].y)
			continue;

		P.at<float>(0, 0) = mid[k].x - pstar.x;
		P.at<float>(0, 1) = mid[k].y - pstar.y;
		Q.at<float>(0, 0) = src[k].x - qstar.x;
		Q.at<float>(0, 1) = src[k].y - qstar.y;

		Mat T = P.t() * P;
		A += w[k] * T;
		T = P.t() * Q;
		B += w[k] * T;
	}

	if (determinant(A) <= 0.00001) {
		dv.x = pstar.x - qstar.x;
		dv.y = pstar.y - qstar.y;
		return dv;
	}

	M = A.inv() * B;
	Mat V(1, 2, CV_32FC1);
	V.at<float>(0) = j - pstar.x;
	V.at<float>(1) = i - pstar.y;
	Mat R = V * M;

	dv.x = R.at<float>(0) + qstar.x ;
	dv.y = R.at<float>(1) + qstar.y ;
	return dv;
}

Vec3b Bilinear(const Mat srcImg, float x, float y)
{
	int xindex = (int)x, yindex = (int)y;
	int xindex1 = ceil(x), yindex1 = ceil(y);
	float v = x - xindex;
	float u = y - yindex;
	float Bch = (1 - v) * ((1 - u) * (float)srcImg.at<Vec3b>(yindex, xindex)[0] + u * (float)srcImg.at<Vec3b>(yindex1, xindex)[0]) +
		v * ((1 - u) * (float)srcImg.at<Vec3b>(yindex, xindex1)[0] + u * (float)srcImg.at<Vec3b>(yindex1, xindex1)[0]);
	float Gch = (1 - v) * ((1 - u) * (float)srcImg.at<Vec3b>(yindex, xindex)[1] + u * (float)srcImg.at<Vec3b>(yindex1, xindex)[1]) +
		v * ((1 - u) * (float)srcImg.at<Vec3b>(yindex, xindex1)[1] + u * (float)srcImg.at<Vec3b>(yindex1, xindex1)[1]);
	float Rch = (1 - v) * ((1 - u) * (float)srcImg.at<Vec3b>(yindex, xindex)[2] + u * (float)srcImg.at<Vec3b>(yindex1, xindex)[2]) +
		v * ((1 - u) * (float)srcImg.at<Vec3b>(yindex, xindex1)[2] + u * (float)srcImg.at<Vec3b>(yindex1, xindex1)[2]);
	Vec3b newvec((uchar)Bch, (uchar)Gch, (uchar)Rch);
	return newvec;
}

Vec3b Nearest(const Mat srcImg, float x, float y)
{
	int xindex = (int)x, yindex = (int)y;
	int xindex1 = ceil(x), yindex1 = ceil(y);
	float remainderX = remainder(x, 1);
	float remainderY = remainder(y, 1);
	int X_new, Y_new;
	if (remainderX <= 0.5)
		X_new = xindex;
	else
	{
		X_new = xindex1;
	}
	if (remainderY <= 0.5)
		Y_new = yindex;
	else
	{
		Y_new = yindex1;
	}
	Vec3b newvec(srcImg.at<Vec3b>(Y_new, X_new)[0], srcImg.at<Vec3b>(Y_new, X_new)[1], srcImg.at<Vec3b>(Y_new, X_new)[2]);
	return newvec;
}

Vec3b Bicubic(const Mat srcImg, float x, float y)
{
	int xindex = (int)x, yindex = (int)y;
	int xindex1 = ceil(x), yindex1 = ceil(y);
	float u = y - yindex, v = x - xindex;
	int xindex_p1, xindex_p2, xindex_m1, yindex_p1, yindex_p2, yindex_m1;
	if (xindex - 1 < 0)
		xindex_m1 = xindex;
	else
		xindex_m1 = xindex - 1;
	if (xindex + 1 > srcImg.cols - 1)
		xindex_p1 = srcImg.cols - 1;
	else
		xindex_p1 = xindex + 1;
	if (xindex + 2 > srcImg.cols - 1)
		xindex_p2 = xindex_p1;
	else
		xindex_p2 = xindex + 2;

	if (yindex - 1 < 0)
		yindex_m1 = yindex;
	else
		yindex_m1 = yindex - 1;
	if (yindex + 1 > srcImg.rows - 1)
		yindex_p1 = srcImg.rows - 1;
	else
		yindex_p1 = yindex + 1;
	if (yindex + 2 > srcImg.rows - 1)
		yindex_p2 = yindex_p1;
	else
		yindex_p2 = yindex + 2;
	Vec3b newvec;
	for (int k = 0; k < 3; k++)
	{
		float result = 0.0;
		result += (S(u + 1)*(float)srcImg.at<Vec3b>(yindex_m1, xindex_m1)[k] +
			S(u)*(float)srcImg.at<Vec3b>(yindex, xindex_m1)[k] +
			S(u - 1)*(float)srcImg.at<Vec3b>(yindex_p1, xindex_m1)[k] +
			S(u - 2)*(float)srcImg.at<Vec3b>(yindex_p2, xindex_m1)[k])*S(v + 1);
		result += (S(u + 1)*(float)srcImg.at<Vec3b>(yindex_m1, xindex)[k] +
			S(u)*(float)srcImg.at<Vec3b>(yindex, xindex)[k] +
			S(u - 1)*(float)srcImg.at<Vec3b>(yindex_p1, xindex)[k] +
			S(u - 2)*(float)srcImg.at<Vec3b>(yindex_p2, xindex)[k])*S(v);
		result += (S(u + 1)*(float)srcImg.at<Vec3b>(yindex_m1, xindex_p1)[k] +
			S(u)*(float)srcImg.at<Vec3b>(yindex, xindex_p1)[k] +
			S(u - 1)*(float)srcImg.at<Vec3b>(yindex_p1, xindex_p1)[k] +
			S(u - 2)*(float)srcImg.at<Vec3b>(yindex_p2, xindex_p1)[k])*S(v - 1);
		result += (S(u + 1)*(float)srcImg.at<Vec3b>(yindex_m1, xindex_p2)[k] +
			S(u)*(float)srcImg.at<Vec3b>(yindex, xindex_p2)[k] +
			S(u - 1)*(float)srcImg.at<Vec3b>(yindex_p1, xindex_p2)[k] +
			S(u - 2)*(float)srcImg.at<Vec3b>(yindex_p2, xindex_p2)[k])*S(v - 2);
		if (result > 255)
			result = 255;
		else if (result < 0)
			result = 0;
		newvec[k] = (uchar)result;
	}
	return newvec;
}


//求出中间图到源图的映射坐标
void mapping(Mat& result, Mat& MLSresult, const vector<Vec6f> srctri, 
	const vector<Vec6f> midtri, int height, int width, const Mat srcImg, vector<Point2f> srclandmark, vector<Point2f> midlandmark)
{
	//Mat result = srcImg.clone();
	int l1, l2, l3;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < midtri.size(); k++)
			{
				Vec6f pt = midtri[k];
				float x1 = pt[0]; float y1 = pt[1]; float x2 = pt[2]; float y2 = pt[3]; float x3 = pt[4]; float y3 = pt[5];
				bool same1 = Issameside((float)j, (float)i, x1, y1, x2, y2, x3, y3);
				bool same2 = Issameside((float)j, (float)i, x2, y2, x1, y1, x3, y3);
				bool same3 = Issameside((float)j, (float)i, x3, y3, x1, y1, x2, y2);
				if (same1 && same2 && same3)//(j, i)点在第k个三角形内
				{					
					for (l1 = 0; l1 < midlandmark.size(); l1++)
						if ((x1 == midlandmark[l1].x) && (y1 == midlandmark[l1].y))
							break;
					for (l2 = 0; l2 < srclandmark.size(); l2++)
						if ((x2 == midlandmark[l2].x) && (y2 == midlandmark[l2].y))
							break;
					for (l3 = 0; l3 < srclandmark.size(); l3++)
						if ((x3 == midlandmark[l3].x) && (y3 == midlandmark[l3].y))
							break;
					Mat_<float> mid = (Mat_<float>(3, 3) << x1, x2, x3, y1, y2, y3, 1.0, 1.0, 1.0);
					Mat_<float> src = (Mat_<float>(3, 3) << srclandmark[l1].x, srclandmark[l2].x,
						srclandmark[l3].x, srclandmark[l1].y, srclandmark[l2].y, srclandmark[l3].y, 1.0, 1.0, 1.0);
					Mat_<float> AffineMatx = AffineMatrix(mid, src);
					float x, y;
					vector<Point2f>t, s;
					t.push_back(Point2f(x1, y1)); t.push_back(Point2f(x2, y2)); t.push_back(Point2f(x3, y3));
					s.push_back(srclandmark[l1]); s.push_back(srclandmark[l2]); s.push_back(srclandmark[l3]);
					x = AffineMatx.at<float>(0, 0)*(float)j + AffineMatx.at<float>(0, 1)*(float)i + AffineMatx.at<float>(0, 2);	//求得的图坐标映射到原图上的坐标
					y = AffineMatx.at<float>(1, 0)*(float)j + AffineMatx.at<float>(1, 1)*(float)i + AffineMatx.at<float>(1, 2);
					if (x < 0)
						x = 0;
					if (x > width - 1)
						x = width - 1;
					if (y < 0)
						y = 0;
					if (y > height - 1)
						y = height - 1;
					if ((i == 2) && (j == 598))
					{
						int p = 0;
					}

					
					result.at<Vec3b>(i, j) = Bilinear(srcImg, x, y); //此处可将Bilinear替换为Nearest或Bicubic，实现最近邻或双三次插值

					//方法二 MLS映射
					/*
					vector<Point2f>pt, ps;
					pt.push_back(Point2f(x1, y1)); pt.push_back(Point2f(x2, y2)); pt.push_back(Point2f(x3, y3));
					ps.push_back(srclandmark[l1]); ps.push_back(ps); ps.push_back(pt);
					Point2f mapping = MLS(midlandmark, srclandmark, j, i);
					x = mapping.x;
					y = mapping.y;
					if (x < 0)
						x = 0;
					if (x > width - 1)
						x = width - 1;
					if (y < 0)
						y = 0;
					if (y > height - 1)
						y = height - 1;
					
					MLSresult.at<Vec3b>(i, j) = Bilinear(srcImg, x, y);
					*/
					break;
				}
			}
		}
	}
}