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


// Matrix2D����  
class Matrix2D {													
public:
	Matrix2D();
	Matrix2D(unsigned int rows, unsigned int cols);
	Matrix2D(unsigned int rows, unsigned int cols, const T&def);
	//���أ��������
	T&operator()(unsigned int row, unsigned int col);							
	const T&operator()(unsigned int row, unsigned int col) const;
	unsigned int rows;
	unsigned int cols;
	//Matrix2D ��ı���Ԫ�ش洢����T
private:
	std::vector<std::vector<T>> data;
};





//ƽ����
class Plane {
public:
	Plane();
	Plane(cv::Vec3f point, cv::Vec3f normal);//Vec3f 3ά���洢��������float���ͣ������ƿռ��һ����
	float operator[](int idx) const;			//�����±������
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



	//PatchMatch����




	class PatchMatch {
	public:
		//����PatchMatch����
		PatchMatch(float alpha, float gamma, float tau_c, float tau_g);
		
		
		//�޷���һ��PatchMatch�� �����ʼ����һ��
		PatchMatch(const PatchMatch& pm) = delete;


		//��ֵ�����Ҳ������
		PatchMatch& operator = (const PatchMatch& pm) = delete;

		//PatchMatch���أ��������
		void operator()(const cv::Mat3b &img1, const cv::Mat3b&img2, int iterations, bool reverse = false);

		//PatchMatch �� set����
		void set(const cv::Mat3b &img1, const cv::Mat3b &img2);

		//PatchMatch��process����
		void process(int iterations, bool reverse = false);

		//PatchMatch��postProcess����������
		void postProcess();

		//�õ����Ӳ�ͼ
		cv::Mat1f getLeftDisparityMap() const;


		//�õ����Ӳ�ͼ
		cv::Mat1f getRightDisparityMap()const;


		//PatchMatch���ĸ�����
		float alpha;

		float gamma;

		float tau_c;

		float tau_g;

		

	private:

		//�жϲ������Ժ���
		float dissimilarity(const cv::Vec3f& pp, const cv::Vec3f&qq, const cv::Vec2f&pg, const cv::Vec2f&qg);

		//ƽ��ƥ����ۺ���
		float plane_match_cost(const Plane&p, int cx, int cy, int ws, int cpv);

		//Ԥ�ȼ�������Ȩ�غ���
		void precompute_pixels_weights(const cv::Mat3b&frame, cv::Mat &weights, int ws);

		//��ʼ��ƽ�溯��
		void initialize_random_planes(Matrix2D<Plane>&planes, float max_d);

		//����ƽ����ۺ���
		void evaluate_planes_cost(int cpv);

		//�ռ䴫������
		void spatial_propagation(int x, int y, int cpv, int iter);

		//ͼ�񴫲�����
		void view_propagation(int x, int y, int cpv);

		//ƽ�澫������
		void plane_refinement(int x, int y, int cpv, float max_delta_z, float max_delta_n, float end_dz);

		void process_pixel(int x, int y, int cpv, int iter);

		//��ƽ�浽�Ӳ�
		void planes_to_disparity(const Matrix2D<Plane>&planes, cv::Mat1f&disp);

		//��䲻�ϸ������
		void fill_invalid_pixels(int y, int x, Matrix2D<Plane>& planes, const cv::Mat1b&validity);

		//��Ȩ��ֵ�˲�
		void weighted_median_filter(int cx, int cy, cv::Mat1f&disparity, const cv::Mat&weights, const cv::Mat1b&valid, int ws, bool use_invalid);

		//������ͼͼ��
		cv::Mat3b views[2];		//left and right view images

		//��������ͼ��ĻҶ��ݶȺ���,֮�����ö�ά����Ϊx��y���������лҶ��ݶ�
		cv::Mat2f grads[2];		//piexls greyscale gradient for both views

		//����ͼ����Ӳ�ͼ��ֻ�кڰ�����ֻ��һά�Ϳ���
		cv::Mat1f disps[2];		//left and right disparity maps

		//����ͼ�������ƽ�棬Matrix������vector����ÿ�����ص㣬�����汾�ʴ洢��������Plane���ͣ�Ҳ����ƽ��
		Matrix2D<Plane> planes[2];	//pixels'planes for left and right vies

		//��������ͼ��ÿ�����صĴ���
		cv::Mat1f costs[2];		//planes' cost
		

		//Ԥ�ȼ������ص�֧�ִ���
		cv::Mat weigs[2];		//precomputed piexels window weights

		//cv::Mat dissm[2];		//pixels dissimilarities, precompute in parallel


		//PatchMatch���к���
		int rows;

		int cols;
	};
}
//����PatchMatch���͵Ķ�����Ե��ô˺����ĵ����Ӳ�ͼ
inline cv::Mat1f pm::PatchMatch::getLeftDisparityMap()const {
	return this->disps[0];
}

//����PatchMatch���͵Ķ�����ô˺����õ����Ӳ�ͼ
inline cv::Mat1f pm::PatchMatch::getRightDisparityMap()const {
	return this->disps[1];
}
//consider preallocated gradients matrix

//����Ҷ��ݶ�
void compute_greyscale_gradient(const::cv::Mat3b& frame, cv::Mat2f&gradient);


//��x��y�Ƿ��ڸ�������
inline bool inside(int x, int y, int lbx, int lby, int ubx, int uby)
{
	return lbx <= x && x < ubx&&lby <= y && y < uby;
}

//����һ����x��y���Լ�һ��ƽ��p�����Եõ��Ӳ�d��������������ĵ�һ����ʽ
inline float disparity(float x, float y, const Plane&p) {
	return p[0] * x + p[1] * y + p[2];
}

//����p��q���㣬��RGB�ռ��L1���룬Ȼ����ݹ�ʽ4�����Եõ������еĵ�q�Դ������ĵ�p��Ȩ�أ�Ҳ����Ӱ����
inline float weight(const cv::Vec3f& p, const cv::Vec3f&q, float gamma = 10.0f)//weight(frame(cy, cx), frame(y, x), this->gamma);
{
	return std::exp(-cv::norm(p - q, cv::NORM_L1 / gamma));
}

template<typename T>


//��ʽ5�ļ���ʽ
inline T vecAverage(const T&x, const T&y, float wx)
{
	return wx * x + (1 - wx) * y;
}
#endif