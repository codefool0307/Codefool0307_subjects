
#include "pch.h"
#include "pm.h"

#include <iostream>
#include <opencv2/opencv.hpp>
//检查是否正确读入了图像
bool check_image(const cv::Mat & image, std::string name = "Image") {
	if (!image.data)
	{
		std::cerr << name << "data not loaded.\n";
		return false;
	}
	return true;
}



//检查图像的尺寸是否相同
bool check_dimensions(const cv::Mat&img1, const cv::Mat&img2) {
	if (img1.cols != img2.cols || img1.rows != img2.rows)
	{
		std::cerr << "Images' dimensions do not corresponds.";

		return false;
	}

	return true;
}




int main(int argc, char**argv) {
	const float alpha = 0.9f;											//以下四个是公式中的值
	const float gamma = 10.0f;
	const float tau_c = 10.0f;
	const float tau_g = 2.0f;
	cv::Mat3b img1 = cv::imread("D:\exercise\毕业设计代码\毕业设计代码\样本\\left.png", cv::IMREAD_COLOR);				//读入两幅图像，Mat3b 表示 图像是三通道，并且数据类型是unchar int，cv::IMREAD_COLOR表示读入原图像
	cv::Mat3b img2 = cv::imread("D:\exercise\毕业设计代码\毕业设计代码\样本\\right.png", cv::IMREAD_COLOR);


	if (!check_image(img1, "Image 1") || !check_image(img2, "Image2"))  //检查图像
		return 1;

	if (!check_dimensions(img1, img2))
		return 1;

	pm::PatchMatch patch_match(alpha, gamma, tau_c, tau_g);				//用四个参数初始化patchmatch类的对象 patch_match

	patch_match.set(img1, img2);
	//首先将img1和img2用来初始化Patch_Match的左右视图，rows和cols。然后预先处理像素权重

	patch_match.process(3);

	patch_match.postProcess();



	cv::Mat1f disp1 = patch_match.getLeftDisparityMap();

	cv::Mat1f disp2 = patch_match.getRightDisparityMap();



	cv::normalize(disp1, disp1, 0, 255, cv::NORM_MINMAX);

	cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);


	



	try

	{

		cv::imwrite("left_disparity.png", disp1);

		cv::imwrite("right_disparity.png", disp2);

	}

	catch (std::exception &e)

	{

		std::cerr << "Disparity save error.\n" << e.what();

		return 1;

	}
	






	return 0;

}
