#include<opencv2\opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;
class OCR
{
public:
	OCR(Mat x) :InputImg(x) {};
	Mat Correction();
	~OCR();

private:
	Mat InputImg;
};
Mat OCR::Correction()
{
	Mat srcGray;
	cvtColor(InputImg, srcGray, CV_RGB2GRAY);
	const int nRows = srcGray.rows;
	const int nCols = srcGray.cols;
	//计算傅里叶变换尺寸
	int cRows = getOptimalDFTSize(nRows);
	int cCols = getOptimalDFTSize(nCols);
	Mat sizeConvMat;
	copyMakeBorder(srcGray, sizeConvMat, 0, cRows - nRows, 0, cCols - nCols, BORDER_CONSTANT, Scalar::all(0));

	Mat groupMats[] = { Mat_<float>(sizeConvMat), Mat::zeros(sizeConvMat.size(), CV_32F) };
	Mat mergeMat;
	//把两页合成一个2通道的mat  
	merge(groupMats, 2, mergeMat);
	//对上面合成的mat进行离散傅里叶变换，支持原地操作，傅里叶变换结果为复数，通道1存的是实部，通道2存的是虚部。  
	dft(mergeMat, mergeMat);
	//把变换的结果分割到各个数组的两页中，方便后续操作  
	split(mergeMat, groupMats);
	//求傅里叶变化各频率的幅值，幅值放在第一页中  
	magnitude(groupMats[0], groupMats[1], groupMats[0]);
	Mat magnitudeMat = groupMats[0].clone();
	//归一化操作，幅值加1
	magnitudeMat += Scalar::all(1);
	//傅里叶变换的幅度值范围大到不适合在屏幕上显示，高值在屏幕上显示为白点，而低值为黑点，  
	//高低值的变化无法有效分辨，为了在屏幕上凸显出高低的变化得连续性，我们可以用对数尺度来替换线性尺度 
	log(magnitudeMat, magnitudeMat);
	//归一化
	normalize(magnitudeMat, magnitudeMat, 0, 1, CV_MINMAX);
	magnitudeMat.convertTo(magnitudeMat, CV_8UC1, 255, 0);
	//imshow("magnitudeMat2", magnitudeMat);
	//重新分配象限，使(0,0)移动到图像中心，  
	//傅里叶变换之前要对源图像乘以(-1)^(x+y)，进行中心化  
	//这是对傅里叶变换结果进行中心化  
	int cx = magnitudeMat.cols / 2;
	int cy = magnitudeMat.rows / 2;
	Mat tmp;
	//Top-Left--为每一个象限创建ROI  
	Mat q0(magnitudeMat, Rect(0, 0, cx, cy));
	//Top-Right  
	Mat q1(magnitudeMat, Rect(cx, 0, cx, cy));
	//Bottom-Left  
	Mat q2(magnitudeMat, Rect(0, cy, cx, cy));
	//Bottom-Right  
	Mat q3(magnitudeMat, Rect(cx, cy, cx, cy));
	//交换象限，(Top-Left with Bottom-Right)  
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	//交换象限，（Top-Right with Bottom-Letf）  
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	Mat binaryMagnMat;
	threshold(magnitudeMat, binaryMagnMat, 155, 255, CV_THRESH_BINARY);
	vector<Vec2f> lines;
	binaryMagnMat.convertTo(binaryMagnMat, CV_8UC1, 255, 0);
	HoughLines(binaryMagnMat, lines, 1, CV_PI / 180, 100, 0, 0);
	cout << "lines.size:  " << lines.size() << endl;
	Mat houghMat(binaryMagnMat.size(), CV_8UC3);
	//绘制检测线
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		//坐标变换生成线表达式
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(houghMat, pt1, pt2, Scalar(0, 0, 255), 1, 8, 0);
	}
	imshow("houghMat", houghMat);
	float theta = 0;
	//检测线角度判断
	for (size_t i = 0; i < lines.size(); i++)
	{
		float thetaTemp = lines[i][1] * 180 / CV_PI;
		cout << lines[i][1] * 180 / CV_PI << endl;
		if (thetaTemp > 0 && thetaTemp < 180 && thetaTemp != 90)
		{
			theta = thetaTemp;
			break;
		}

	}

	//角度转换
	float angelT = nRows*tan(theta / 180. * CV_PI) / nCols;
	theta = atan(angelT) * 180 / CV_PI;
	cout << "theta: " << theta << endl;

	//取图像中心
	Point2f centerPoint = Point2f(nCols / 2, nRows / 2);
	double scale = 1;
	//计算旋转中心
	Mat warpMat = getRotationMatrix2D(centerPoint, theta, scale);
	//仿射变换
	Mat resultImage(srcGray.size(), srcGray.type());
	warpAffine(srcGray, resultImage, warpMat, resultImage.size());
	return resultImage;
}

OCR::~OCR()
{
}