#ifndef PATCH_MATCH
#define PATCH_MATCH
#include <iostream>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include <exception>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>
template <typename T>


// Matrix2D类型  
class Matrix2D {													
public:
	Matrix2D();
	Matrix2D(unsigned int rows, unsigned int cols);
	Matrix2D(unsigned int rows, unsigned int cols, const T&def);
	//重载（）运算符
	T&operator()(unsigned int row, unsigned int col);							
	const T&operator()(unsigned int row, unsigned int col) const;
	unsigned int rows;
	unsigned int cols;
	//Matrix2D 类的本质元素存储的是T
private:
	std::vector<std::vector<T>> data;
};





//平面类
class Plane {
public:
	Plane();
	Plane(cv::Vec3f point, cv::Vec3f normal);//Vec3f 3维，存储的数据是float类型，就类似空间的一个点
	float operator[](int idx) const;			//重载下标运算符
	cv::Vec3f operator()();
	cv::Vec3f getPoint();
	cv::Vec3f getNormal();
	cv::Vec3f getCoeff();
	Plane viewTranform(int x, int y, int sign, int & qx, int &qy);
private:
	cv::Vec3f point;
	cv::Vec3f normal;
	cv::Vec3f coeff;
};
namespace pm {



	//PatchMatch类型




	class PatchMatch {
	public:
		//定义PatchMatch类型
		PatchMatch(float alpha, float gamma, float tau_c, float tau_g);
		
		
		//无法用一个PatchMatch类 对象初始化另一个
		PatchMatch(const PatchMatch& pm) = delete;


		//赋值运算符也不能用
		PatchMatch& operator = (const PatchMatch& pm) = delete;

		//PatchMatch重载（）运算符
		void operator()(const cv::Mat3b &img1, const cv::Mat3b&img2, int iterations, bool reverse = false);

		//PatchMatch 的 set函数
		void set(const cv::Mat3b &img1, const cv::Mat3b &img2);

		//PatchMatch的process函数
		void process(int iterations, bool reverse = false);

		//PatchMatch的postProcess函数，后处理
		void postProcess();

		//得到左视差图
		cv::Mat1f getLeftDisparityMap() const;


		//得到右视差图
		cv::Mat1f getRightDisparityMap()const;


		//PatchMatch的四个参数
		float alpha;

		float gamma;

		float tau_c;

		float tau_g;

		

	private:

		//判断不相似性函数
		float dissimilarity(const cv::Vec3f& pp, const cv::Vec3f&qq, const cv::Vec2f&pg, const cv::Vec2f&qg);

		//平面匹配代价函数
		float plane_match_cost(const Plane&p, int cx, int cy, int ws, int cpv);

		//预先计算像素权重函数
		void precompute_pixels_weights(const cv::Mat3b&frame, cv::Mat &weights, int ws);

		//初始化平面函数
		void initialize_random_planes(Matrix2D<Plane>&planes, float max_d);

		//评估平面代价函数
		void evaluate_planes_cost(int cpv);

		//空间传播函数
		void spatial_propagation(int x, int y, int cpv, int iter);

		//图像传播函数
		void view_propagation(int x, int y, int cpv);

		//平面精化函数
		void plane_refinement(int x, int y, int cpv, float max_delta_z, float max_delta_n, float end_dz);

		void process_pixel(int x, int y, int cpv, int iter);

		//由平面到视差
		void planes_to_disparity(const Matrix2D<Plane>&planes, cv::Mat1f&disp);

		//填充不合格的像素
		void fill_invalid_pixels(int y, int x, Matrix2D<Plane>& planes, const cv::Mat1b&validity);

		//加权中值滤波
		void weighted_median_filter(int cx, int cy, cv::Mat1f&disparity, const cv::Mat&weights, const cv::Mat1b&valid, int ws, bool use_invalid);

		//左右视图图像
		cv::Mat3b views[2];		//left and right view images

		//左右两幅图像的灰度梯度函数,之所以用二维是因为x和y两个方向都有灰度梯度
		cv::Mat2f grads[2];		//piexls greyscale gradient for both views

		//左右图像的视差图，只有黑白所以只用一维就可以
		cv::Mat1f disps[2];		//left and right disparity maps

		//左右图像的像素平面，Matrix用两个vector代表每个像素点，但里面本质存储的数据是Plane类型，也就是平面
		Matrix2D<Plane> planes[2];	//pixels'planes for left and right vies

		//左右两幅图像，每个像素的代价
		cv::Mat1f costs[2];		//planes' cost
		

		//预先计算像素的支持窗口
		cv::Mat weigs[2];		//precomputed piexels window weights

		//cv::Mat dissm[2];		//pixels dissimilarities, precompute in parallel


		//PatchMatch的行和列
		int rows;

		int cols;
	};
}
//对于PatchMatch类型的对象可以调用此函数的到左视差图
inline cv::Mat1f pm::PatchMatch::getLeftDisparityMap()const {
	return this->disps[0];
}

//对于PatchMatch类型的对象调用此函数得到右视差图
inline cv::Mat1f pm::PatchMatch::getRightDisparityMap()const {
	return this->disps[1];
}
//consider preallocated gradients matrix

//计算灰度梯度
void compute_greyscale_gradient(const::cv::Mat3b& frame, cv::Mat2f&gradient);


//点x，y是否在该区域内
inline bool inside(int x, int y, int lbx, int lby, int ubx, int uby)
{
	return lbx <= x && x < ubx&&lby <= y && y < uby;
}

//对于一个点x，y，以及一个平面p，可以得到视差d，就是论文里面的第一个公式
inline float disparity(float x, float y, const Plane&p) {
	return p[0] * x + p[1] * y + p[2];
}

//对于p，q两点，求RGB空间的L1距离，然后根据公式4，可以得到窗口中的点q对窗口中心点p的权重，也就是影响力
inline float weight(const cv::Vec3f& p, const cv::Vec3f&q, float gamma = 10.0f)//weight(frame(cy, cx), frame(y, x), this->gamma);
{
	return std::exp(-cv::norm(p - q, cv::NORM_L1 / gamma));
}

template<typename T>


//公式5的简化形式
inline T vecAverage(const T&x, const T&y, float wx)
{
	return wx * x + (1 - wx) * y;
}
#endif