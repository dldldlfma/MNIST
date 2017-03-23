#include <iostream>
#include <cstdio>
#include <cmath>
#include <assert.h>
#include <vector>				//vector ���
#include <algorithm>			//max ����� ���ؼ�
#include <ctime>				//clk ����
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

//---------�����͸� �о� ���� �κ�----------
int Data_extract(FILE * data_pass)
{
	/*
		MNIST �����Ͱ� ó�� ���� 16byte�� �����Ϳ� ���� ������ �־�� 
		�տ� 4byte�� ���� �𸣰ڴµ� �ڿ� ������ 12byte�� �ǹ̰� �ִµ���
		5������ 8�� byte�� �������� ������ �ǹ� �ؿ�
		9������ 12�� byte�� �������� ���� �ǹ��ϰ�
		13������ 16�� byte�� �������� ���� �ǹ��մϴ�.

		��¼�� ��� ������ �ٲ��������.... ����
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
	int row, colm;          //���
	int num_of_data;		//������ ����

	FILE * answer = fopen("train-labels.idx1-ubyte", "rb"); // "rt�� rb�� ����?"

	if (answer == NULL)
	{
		puts("answer ���� ���� ����!");
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
		puts("images ���� ���� ����!");
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
				 //������ ǥ�� �κ�
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

//--------�о�� ������ �׸����� ǥ���ϱ� using OpenCV----

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


//---------Convolution�� ���� �ϴ� layer-----
/*
1. ��������� �ϱ����� �Է�? 28x28 �̹��� �������� �Է�
2. ����� ������� ������ ������ 5x5�� �����ϸ� ���μ��� 4�� ���� 24x24
3. �׷� ���ʹ� ��� ��� �ϴ°�?
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

	printf("�ɸ��ð��� %5.2f�� �Դϴ�. \n", result);
	return 0;
}
