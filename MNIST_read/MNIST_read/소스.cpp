#include <iostream>
#include <cstdio>
#include <cmath>
#include <assert.h>
#include <vector>				//vector 사용
#include <algorithm>			//max 사용을 위해서
#include <ctime>				//clk 측정
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

unsigned char image[60000][28][28];
unsigned char answer_value[60000];

double sigmoid(const double & x)
{
	return 1 / (1 + exp(-x));
}

double relu(const double &x)
{
	return max(x, 0.0);
}

//---------데이터를 읽어 오는 부분----------
int Data_extract(FILE * data_pass)
{
	/*
		MNIST 데이터가 처음 시작 16byte에 데이터에 관한 정보가 있어요 
		앞에 4byte는 뭔지 모르겠는데 뒤에 나머지 12byte는 의미가 있는데요
		5번부터 8번 byte는 데이터의 개수를 의미 해요
		9번부터 12번 byte는 데이터의 행을 의미하고
		13번부터 16번 byte는 데이터의 열을 의미합니다.

		어쩌면 행렬 순서가 바뀌었을지도.... ㅎㅎ
	*/
	int ch;
	int byte_value = 0;

	ch = fgetc(data_pass);
	ch = fgetc(data_pass);
	ch = fgetc(data_pass);
	byte_value = byte_value + (ch << 8);
	ch = fgetc(data_pass);
	byte_value = byte_value + (ch);
	
	printf("%d  ", byte_value);
	return byte_value;
}

int data_read()
{
	int ch;
	int row, colm;          //행렬
	int num_of_data;		//데이터 개수

	FILE * answer = fopen("train-labels.idx1-ubyte", "rb"); // "rt와 rb의 차이?"

	if (answer == NULL)
	{
		puts("answer 파일 오픈 실패!");
		return -1;
	}
	for (int i = 0; i < 8; i++)
	{
		ch = fgetc(answer);
		printf("%d", ch);
	}
	printf("\n\n");

	for (int i = 0; i < 60000; i++)
	{
		ch = fgetc(answer);
		answer_value[i] = ch;
		//printf("%d ", ch);
	}

	printf("\n-------answer file close -------\n");
	fclose(answer);
	 

	FILE * fp = fopen("train-images.idx3-ubyte", "rb");

	if (fp == NULL)
	{
		puts("images 파일 오픈 실패!");
		return -1;
	}

	
	Data_extract(fp);
	printf("\n\n");

	printf("num_of_data : ");
	num_of_data=Data_extract(fp);
	printf("row : ");
	row = Data_extract(fp);
	printf("colm : ");
	colm = Data_extract(fp);

	printf("\n\n");
	

	printf("\n-------imagefile open -------\n");


	for (int m = 0; m < 100; m++)
	{

		for (int i = 0; i < 28; i++)
		{
			for (int k = 0; k < 28; k++)
			{
				ch = fgetc(fp);


				if (ch == -1)
				{
					printf("\n\n----error----\n\n");
					while (1)
					{
						printf("plz ctrl + c");
					}
					
				}
				image[m][i][k] = ch;
				
				
				if (ch>200)
				{
					printf("#");
				}
				else if (ch>100)
				{
					printf("3");
				}
				else if (ch > 40)
				{
					printf("1");
				}
				else
				{
					printf(" ");
				}
				 //데이터 표시 부분
				//printf("%d	", ch);


			}
			printf("\n");
		}
		printf("%d \n",answer_value[m]);
		//printf("\n\n");
	}
	return 0;
}
//------------------------------------------

//--------읽어온 데이터 그림으로 표시하기 using OpenCV----

void show_me_the_data2828(unsigned char image[][28], int row, int col)
{
	cv::Mat test;

	test.create(row, col, CV_8UC1);
	

	for (int i = 0; i < test.rows; i++){
		for (int j = 0; j < test.cols; j++)
		{
			test.at<uchar>(i, j) = image[i][j];
		}
	}

	//cout << test.rows << endl;
	//cout << test.cols << endl;

	imshow("test value", test);
	waitKey(0);
}
void show_me_the_data1414(unsigned char image[][14], int row, int col)
{
	cv::Mat test;

	test.create(row, col, CV_8UC1);


	for (int i = 0; i < test.rows; i++){
		for (int j = 0; j < test.cols; j++)
		{
			test.at<uchar>(i, j) = image[i][j];
		}
	}

	//cout << test.rows << endl;
	//cout << test.cols << endl;

	imshow("test value", test);
	waitKey(0);
}


//---------Convolution을 수행 하는 layer-----
/*
1. 컨벌루션을 하기위한 입력? 28x28 이미지 데이터의 입력
2. 출력은 컨벌루션 이후의 데이터 5x5로 연산하면 가로세로 4씩 감소 24x24
3. 그럼 필터는 어떤걸 써야 하는가?
*/
void conv3x3(unsigned char image[][28], unsigned char filter[][3], int image_row, int image_col)
{
	for (int i = 0; i < image_row; i++)
	{

	}
}




int main()
{
	clock_t before;
	double result;

	before = clock();

	data_read();
	//show_me_the_data2828(image[0],28,28);
	
	result = (double)(clock() - before) / CLOCKS_PER_SEC;

	printf("걸린시간은 %5.2f초 입니다. \n", result);
	return 0;
}
