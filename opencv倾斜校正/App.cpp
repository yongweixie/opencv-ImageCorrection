#include"OCR.hpp"
#define devView(i) imshow(#i,i)
#define PI 3.1415936
int main()
{
	Mat source = imread("imageText_02_R.jpg");
	//Mat output = RotateImg(source, -20);
	OCR img(source);
	while (waitKey(1)!=27)
	{
		devView(source);
		devView(img.Correction());
		//devView(output);
	}
	return 0;
}