
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cstdio>
#include <omp.h>
 
#define INPUT_IMAGE_DIMENSION 1
#define INPUT_IMAGE_SIZE 28
 
#define FINAL_OUTPUT 10
 
#define NUM_OF_FIRST_CONV_FILTER 5
#define NUM_OF_SECOND_CONV_FILTER 50
 
#define NUM_OF_FCNN_LAYER 2
#define NUM_OF_NEURON_IN_FIRST_FCNN_LAYER 100
#define NUM_OF_NEURON_IN_SECOND_FCNN_LAYER 10
#define FCNN_FIRST_INPUT_SIZE 800
#define FILTER_SIZE 5
 
#define LEARNING_RATE 0.0001
 
using namespace std;
 
#define MAX(a,b) ((a>b) ? a : b)
 
 
typedef vector<double> Mat1D;
typedef vector<Mat1D> Mat2D;
typedef vector<Mat2D> Mat3D;
typedef vector<Mat3D> Mat4D;
 
struct conv_val
{
	Mat2D unroll_input;
	Mat2D d_ReLU;
	Mat2D output;
	Mat2D delta_map_2D_from_delta;
	Mat2D delta_map_2D_now;
};
 
struct pool_val
{
	vector<Mat2D> input;
	vector<Mat2D> output;
	vector<Mat2D> delta;
	vector<Mat2D> pooling_position;
};
 
struct fcnn_layer_val
{
	Mat1D input;
	Mat1D output;
	Mat1D delta;
	Mat1D d_active;
};
 
double image[60000][28][28];
unsigned char answer_value[60000];
int answer_success = 0;
 
//---------데이터를 읽어 오는 부분----------7
int Data_extract(FILE * data_pass)
{
	/*
	MNIST 데이터가 처음 시작 16byte에 데이터에 관한 정보가 있어요
	앞에 4byte는 뭔지 모르겠는데 뒤에 나머지 12byte는 의미가 있는데요
	5번부터 8번 byte는 데이터의 개수를 의미 해요
	9번부터 12번 byte는 데이터의 행을 의미하고
	13번부터 16번 byte는 데이터의 열을 의미합니다.
 
	어쩌면 행렬 순서가 바뀌었을지도.... ㅎㅎ
	*/
	int ch;
	int byte_value = 0;
 
	ch = fgetc(data_pass);
	ch = fgetc(data_pass);
	ch = fgetc(data_pass);
	byte_value = byte_value + (ch << 8);
	ch = fgetc(data_pass);
	byte_value = byte_value + (ch);
 
	printf("%d  ", byte_value);
	return byte_value;
}
 
int data_read()
{
	int ch;
	int row, colm;          //행렬
	int num_of_data;		//데이터 개수
 
	FILE * answer = fopen("train-labels.idx1-ubyte", "rb"); // "rt와 rb의 차이? read binary & read text"
 
	if (answer == NULL)
	{
		puts("answer 파일 오픈 실패2!");
		return -1;
	}
	for (int i = 0; i < 8; i++)
	{
		ch = fgetc(answer);
		printf("%d", ch);
	}
	printf("\n\n");
 
	for (int i = 0; i < 60000; i++)
	{
		ch = fgetc(answer);
		answer_value[i] = ch;
		//printf("%d ", ch);
	}
 
	printf("\n-------Answer file close -------\n");
	fclose(answer);
 
 
	FILE * fp = fopen("train-images.idx3-ubyte", "rb");
 
	if (fp == NULL)
	{
		puts("images 파일 오픈 실패1!");
		return -1;
	}
 
 
	Data_extract(fp);
	printf("\n\n");
 
	printf("num_of_data : ");
	num_of_data = Data_extract(fp);
	printf("row : ");
	row = Data_extract(fp);
	printf("colm : ");
	colm = Data_extract(fp);
 
	printf("\n\n");
 
 
	printf("\n-------Image file open -------\n");
 
 
	for (int m = 0; m < 60000; m++)
	{
 
		for (int i = 0; i < 28; i++)
		{
			for (int k = 0; k < 28; k++)
			{
				ch = fgetc(fp);
 
 
				if (ch == -1)
				{
					printf("\n\n----error----\n\n");
					while (1)
					{
						printf("plz ctrl + c");
					}
 
				}
				image[m][i][k] = (double)ch;
 
				/*
				if (ch>200)
				{
				printf("#");
				}
				else if (ch>100)
				{
				printf("3");
				}
				else if (ch > 40)
				{
				printf("1");
				}
				else
				{
				printf(" ");
				}
				*/
				//데이터 표시 부분
				//printf("%d	", ch);
 
 
			}
			//printf("\n");
		}
		//printf("%d \n",answer_value[m]);
		//printf("\n\n");
	}
	printf("\n-------Image file close -------\n");
	return 0;
}
//------------------------------------------
 
Mat2D Mat_resize(int a)
{
	Mat2D Mat;
	Mat.resize(a);
	for (int i = 0; i < Mat.size(); i++)
	{
		Mat[i].resize(a);
	}
 
	for (int i = 0; i < Mat.size(); i++)
	{
		for (int j = 0; j < Mat[0].size(); j++)
		{
			Mat[i][j] = 0;
		}
	}
 
	return Mat;
}
 
 
void Mat1D_init(Mat1D &a)
{
#pragma omp parallel for
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = ((((double)(rand() / (double)(RAND_MAX)))-0.5) * 0.1);
	}
}
 
double MAX_func(double a, double b, double c, double d, int position[])
{
	double max;
 
	max = a;
 
	for (int i = 0; i<4; i++)
	{
		position[i] = 0;
	}
 
 
	if (b > max)
	{
		max = b;
	}
	if (c > max)
	{
		max = c;
	}
	if (d > max)
	{
		max = d;
	}
 
	if ((a == 0 && b == 0) && (c == 0 && d == 0))
	{
		return max;
	}
 
	if (a == max)
	{
		position[0] = 1;
	}
	if (b == max)
	{
		position[1] = 1;
	}
	if (c == max)
	{
		position[2] = 1;
	}
	if (d == max)
	{
		position[3] = 1;
	}
 
	return max;
}
 
int MAX_func(Mat1D a)
{
	double max;
	int answer;
	max = a[0];
	answer = 0;
 
	for (int i = 0; i<a.size(); i++)
	{
		if (a[i] > max)
		{
			max = a[i];
			answer = i;
		}
		
	}
	return answer;
}
 
void FCNN_forward_ReLU(fcnn_layer_val &fcnn, Mat2D weight)
{
	fcnn.output.resize(weight.size());
	fcnn.d_active.resize(weight.size());
	fcnn.delta.resize(weight.size());
 
	for (int i = 0; i < fcnn.output.size(); i++)
	{
		for (int j = 0; j < fcnn.input.size(); j++)
		{
			fcnn.output[i] += fcnn.input[j] * weight[i][j];
		}
		fcnn.output[i] = fcnn.output[i] + weight[i][fcnn.input.size()];
		if (fcnn.output[i]>0)
		{
			fcnn.d_active[i] = 1;
		}
		else
		{
			fcnn.output[i] = 0;
			fcnn.d_active[i] = 0;
		}
	}
}
 
void FCNN_forward_softmax(fcnn_layer_val &fcnn, Mat2D weight, double answer)
{
	Mat1D temp_output;
 
	temp_output.resize(weight.size());
 
	fcnn.output.resize(weight.size());
	fcnn.d_active.resize(weight.size());
	fcnn.delta.resize(weight.size());
 
	for (int i = 0; i < fcnn.output.size(); i++)
	{
		for (int j = 0; j < fcnn.input.size(); j++)
		{
			temp_output[i] += fcnn.input[j] * weight[i][j];
		}
		temp_output[i] = temp_output[i] + weight[i][fcnn.input.size()];
	}
 
	double sum = 0;
 
	for (int i = 0; i < fcnn.output.size(); i++)
	{
		temp_output[i] = exp(temp_output[i]);
		sum += temp_output[i];
	}
	//printf("\n sum : %f \n\n", sum);
 
	for (int i = 0; i < fcnn.output.size(); i++)
	{
		fcnn.output[i] = temp_output[i] / sum;
		if (i == (int)answer)
		{
			cout <<"O : "<<fcnn.output[i] <<"	answer : "<< answer<< endl;
			//if (fcnn.output[i]>0.5)
			//{
			//	answer_success += 1;
			//}
			fcnn.delta[i] = fcnn.output[i] - 1;
		}
		else
		{
			//cout << "X : " << fcnn.output[i] << "	answer : " << i << endl;
			fcnn.delta[i] = fcnn.output[i];
		}
	}
 
	if (answer == MAX_func(fcnn.output))
	{
		answer_success++;
	}
}
 
 
 
//----------- 이전 코드 --------------------------
 
void Mat_2D_transposed(Mat2D &input, Mat2D &output)
{
 
	output.resize(input[0].size());
	for (int i = 0; i < output.size(); i++)
	{
		output[i].resize(input.size());
	}
 
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input[0].size(); j++)
		{
			output[j][i] = input[i][j];
		}
	}
}
 
 
void Mat_2D_transposed(Mat2D &input)
{
 
	Mat2D output;
	output.resize(input[0].size());
	for (int i = 0; i < output.size(); i++)
	{
		output[i].resize(input.size());
	}
 
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input[0].size(); j++)
		{
			output[j][i] = input[i][j];
		}
	}
 
	input.resize(0);
	input.resize(output.size());
	for (int i = 0; i < output.size(); i++)
	{
		input[i].resize(output[0].size());
		for (int j = 0; j < output[0].size(); j++)
		{
			input[i][j] = output[i][j];
		}
	}
 
}
 
 
 
//input_A[i][j] x input_B[j][k] = output[i][k] 이렇게 해야하는데 
//input_A[i][j] x input_B[k][j] = output[i][k] 로 연산 할 것
//input_A[i][j] i행 j열
//input_B[j][k]가 맞음 그러나 구현은 input_B[k][j]로 함 연산 최적화를 위해서
//그러니 연산 형태는 다음과 같다.
//Input_A[행][열] x input_B[열][행] = output[input_A_행][input_B_행]
//example은 vector_test.sln
void Mat_mult(Mat2D input_A, Mat2D input_B, Mat2D &output, Mat2D &d_ReLU)
{
	double sum = 0;
	output.resize(input_A.size());
	d_ReLU.resize(input_A.size());
 
	
	for (int k = 0; k < input_B.size(); k++)
	{
 
		for (int i = 0; i < input_A.size(); i++)			//input_A[i][k] i는 행
		{
#pragma omp parallel for reduction(+:sum)
			for (int j = 0; j < input_A[0].size(); j++)
			{
				sum += input_A[i][j] * input_B[k][j];
			}
			if (sum>0)
			{
				output[i].push_back(sum);
				d_ReLU[i].push_back(1);
			}
			else
			{
				output[i].push_back(0);
				d_ReLU[i].push_back(0);
			}
			sum = 0;
 
		} 
 
	}
	
}
 
 
void Mat_mult(Mat2D input_A, Mat2D input_B, Mat2D &output)
{
	double sum = 0;
	output.resize(input_A.size());
 
	for (int k = 0; k < input_B.size(); k++)
	{
 
		for (int i = 0; i < input_A.size(); i++)			//input_A[i][k] i는 행
		{
#pragma omp parallel for reduction(+:sum)
			for (int j = 0; j < input_A[0].size(); j++)
			{
				sum += input_A[i][j] * input_B[k][j];
			}
			output[i].push_back(sum);
			sum = 0;
		}
	}
}
 
 
 
 
void Max_Pooling_2x2(Mat2D input, Mat2D & output, Mat2D & Max_position)
{
	int position[4] = { 0, };
	output = Mat_resize(input.size() / 2);
	Max_position = Mat_resize(input.size());
	for (int i = 0; i < output.size(); i++)
	{
		for (int j = 0; j < output.size(); j++)
		{
			output[i][j] = ((MAX_func(input[(2 * i)][(2 * j)], input[(2 * i)][(2 * j) + 1], input[(2 * i) + 1][(2 * j)], input[(2 * i) + 1][(2 * j) + 1], position)));
 
			Max_position[(2 * i)][(2 * j)] = position[0];
			Max_position[(2 * i)][(2 * j) + 1] = position[1];
			Max_position[(2 * i) + 1][(2 * j)] = position[2];
			Max_position[(2 * i) + 1][(2 * j) + 1] = position[3];
		}
	}
}
 
 
void Unrolling_2D_image(Mat3D input, Mat2D &output, int filter_size = 5)
{
 
	for (int C = 0; C < input.size(); C++)		//input.size() = C = thickness
	{
		int output_size = input[0].size() - (filter_size - 1);
		output.resize(output_size*output_size);
 
		for (int row = 0; row < output_size; row++)
		{
			for (int col = 0; col < output_size; col++)
			{
				for (int i = 0; i < filter_size; i++)
				{
					for (int j = 0; j < filter_size; j++)
					{
						output[(output_size*row) + col].push_back((double)input[C][i + row][j + col]);
					}
				}if (C == input.size() - 1) { output[(output_size*row) + col].push_back(1); }
			}
		}
	}
}
 
void Un_pooling_2x2(Mat2D input, Mat2D & Max_position)
{
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input.size(); j++)
		{
			Max_position[(2 * i)][(2 * j)] = Max_position[(2 * i)][(2 * j)] * input[i][j];
			Max_position[(2 * i)][(2 * j) + 1] = Max_position[(2 * i)][(2 * j) + 1] * input[i][j];
			Max_position[(2 * i) + 1][(2 * j)] = Max_position[(2 * i) + 1][(2 * j)] * input[i][j];
			Max_position[(2 * i) + 1][(2 * j) + 1] = Max_position[(2 * i) + 1][(2 * j) + 1] * input[i][j];
		}
	}
}
 
 
 
// filter를 4D(3D의 배열)로 받아서 3D(2D의 배열)로 만들어서 값을 뽑아냄
int conv(Mat3D input, conv_val &conv_value, Mat2D filter, int & real_mat_size)
{
	Mat2D temp_output;
	Mat2D temp_drelu;
	Unrolling_2D_image(input, conv_value.unroll_input);
 
	Mat_mult(conv_value.unroll_input, filter,temp_output, temp_drelu);
	
	Mat_2D_transposed(temp_output,conv_value.output);
	Mat_2D_transposed(temp_drelu,conv_value.d_ReLU);
 
	real_mat_size = real_mat_size - FILTER_SIZE + 1;
	return 0;
}
 
 
void pool(conv_val a, pool_val &pooling_value, int &mat_size)
{
	pooling_value.input.resize(a.output.size());
	pooling_value.output.resize(a.output.size());
	pooling_value.pooling_position.resize(a.output.size());
 
 
	for (int k = 0; k < a.output.size(); k++)
	{
		for (int i = 0; i < mat_size; i++)
		{
			pooling_value.input[k].resize(mat_size);
			for (int j = 0; j < mat_size; j++)
			{
				pooling_value.input[k][i].push_back(a.output[k][j + (mat_size * i)]);
			}
		}
	}
 
#pragma omp parallel for
	for (int i = 0; i < pooling_value.input.size(); i++)
	{
		Max_Pooling_2x2(pooling_value.input[i], pooling_value.output[i], pooling_value.pooling_position[i]);
	}
 
	mat_size = mat_size / 2;
 
}
 
 
//2D Matrix array를 1D array로 변경 
void convert_2D_to_1D(vector<Mat2D> input, Mat1D & output)
{
	output.resize(input.size() * input[0].size() * input[0][0].size());
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input[0].size(); j++)
		{
			for (int k = 0; k < input[0][0].size(); k++)
			{
				output[(16 * i) + (4 * j) + k] = input[i][j][k];
			}
		}
	}
 
}
 
void input_initial(Mat2D &input, double image[][INPUT_IMAGE_SIZE], double& answer, int i)
{
 
	for (int i = 0; i < input.size(); i++)
	{
		input = Mat_resize(INPUT_IMAGE_SIZE);
	}
 
 
	for (int j = 0; j < INPUT_IMAGE_SIZE; j++)
	{
		for (int k = 0; k < INPUT_IMAGE_SIZE; k++)
		{
			input[j][k] = image[j][k];
		}
	}
 
	answer = (double)answer_value[i];
 
 
}
 
 
 
// a에서 b로 delta의 전달
void FCNN_backprop(Mat1D& a, Mat1D& b, Mat2D weight)
{
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < b.size(); j++)
		{
			a[i] += b[j] * weight[j][i];
		}
	}
}
 
 
 
//1D Matrix array를 2D array로 변경
//2D로 변경할때 몇 by 몇의 Matrix로 변경할건지를
//Mat2D_size에 넣어주면 변환되서 나오는 X by X  Matrix의 개수는
//자동으로 계산된다.
void convert_1D_to_2D(Mat1D input, int Mat2D_row_size, int Mat2D_col_size, vector<Mat2D> & output)
{
	output.resize(input.size() / (Mat2D_row_size * Mat2D_col_size));
 
	for (int i = 0; i < output.size(); i++)
	{
		output[i].resize(Mat2D_row_size);
		for (int j = 0; j < Mat2D_row_size; j++)
		{
			output[i][j].resize(Mat2D_col_size);
		}
	}
 
 
	for (int i = 0; i < output.size(); i++)
	{
		for (int j = 0; j < output[0].size(); j++)
		{
			for (int k = 0; k < output[0][0].size(); k++)
			{
				output[i][j][k] = input[(Mat2D_row_size * Mat2D_col_size * i) + (Mat2D_col_size*j) + k];
			}
		}
	}
}
 
 
//fcnn->pool->conv
void Pooling_backprop(pool_val& pool, Mat1D delta, int unpool_row, int unpool_col)
{
	convert_1D_to_2D(delta, unpool_row, unpool_col, pool.delta);
 
#pragma omp parallel for
	for (int i = 0; i < pool.delta.size(); i++)
	{
		Un_pooling_2x2(pool.delta[i], pool.pooling_position[i]);
	}
 
}
 
//pool->conv
void Pooling_backprop(pool_val& pool)
{
#pragma omp parallel for
	for (int i = 0; i < pool.delta.size(); i++)
	{
		Un_pooling_2x2(pool.delta[i], pool.pooling_position[i]);
	}
 
}
 
//conv->pool
void conv_backprop(vector<Mat2D> delta_map, Mat2D filter, conv_val& conv, pool_val& pool,int original_conv_input_row_size, int original_conv_input_col_size, int filter_depth)
{
	Mat2D temp;
	temp.resize(delta_map.size());
	for (int i = 0; i < delta_map.size(); i++)
	{
		for (int j = 0; j < delta_map[0].size(); j++)
		{
			for (int k = 0; k < delta_map[0][0].size(); k++)
			{
				//d_ReLU를 곱해줌
				temp[i].push_back(delta_map[i][j][k] * conv.d_ReLU[i][delta_map[0].size()*j + k]);
			}
		}
	}
 
	// temp는 이따가 써야되서 빼내야 됨!!! delta_map_2D_now에 저장, weight update할때 씀!!
	conv.delta_map_2D_now = temp;
 
	Mat_2D_transposed(temp);
	Mat_2D_transposed(filter);
 
 
 
	Mat_mult(temp, filter, conv.delta_map_2D_from_delta);
 
 
 
	pool.delta.resize(pool.output.size());
	for (int i = 0; i < pool.delta.size(); i++)
	{
		pool.delta[i].resize(pool.output[0].size());
		for (int j = 0; j < 12; j++)
		{
			pool.delta[i][j].resize(pool.output[0][0].size());
		}
	}
 
	for (int row = 0; row < original_conv_input_row_size; row++)
	{
		for (int col = 0; col < original_conv_input_col_size; col++)
		{
			for (int i = 0; i < filter_depth; i++)
			{
				for (int j = 0; j < FILTER_SIZE; j++)
				{
					for (int k = 0; k < FILTER_SIZE; k++)
					{
						pool.delta[i][j + row][k + col] += conv.delta_map_2D_from_delta[(8 * row) + col][((25 * i) + (5 * j) + (k))];
					}
 
				}
			}
		}
	}
}
 
//conv->first_input
void conv_backprop(vector<Mat2D> delta_map,Mat2D filter, conv_val& conv)
{
	Mat2D temp;
 
	temp.resize(delta_map.size());
	for (int i = 0; i < delta_map.size(); i++)
	{
		for (int j = 0; j < delta_map[0].size(); j++)
		{
			for (int k = 0; k < delta_map[0][0].size(); k++)
			{
				//d_ReLU를 곱해줌
				temp[i].push_back(delta_map[i][j][k] * conv.d_ReLU[i][delta_map[0].size()*j + k]);
			}
		}
	}
 
	// temp는 이따가 써야됨 빼내야 됨!!! weight update할때 씀!!
 
 
	//temp 꺼냄
	conv.delta_map_2D_now = temp;
 
#pragma omp parallel
	{
#pragma omp sections
		{
#pragma omp section
			{
				Mat_2D_transposed(temp);
			}
#pragma omp section
			{
				Mat_2D_transposed(filter);
			}
			
 
		}
	}
	
 
	Mat_mult(temp, filter, conv.delta_map_2D_from_delta);
}
 
void final_value_initialize(conv_val &conv, pool_val & pool, fcnn_layer_val &fcnn)
{
	//------------------------initialize-----------------------------
	conv.delta_map_2D_from_delta.resize(0);
	conv.delta_map_2D_now.resize(0);
	conv.d_ReLU.resize(0);
	conv.output.resize(0);
	conv.unroll_input.resize(0);
	
	pool.delta.resize(0);
	pool.input.resize(0);
	pool.output.resize(0);
	pool.pooling_position.resize(0);
 
	fcnn.delta.resize(0);
	fcnn.d_active.resize(0);
	fcnn.input.resize(0);
	fcnn.output.resize(0);
}
 
int save_Weights(const string &path, Mat2D filter_1, Mat2D filter_2, Mat2D fcnn_weight_1, Mat2D fcnn_weight_2)
{
	FILE * fp = fopen(path.c_str(), "wb");
 
	for (int i = 0; i<filter_1.size(); i++)
	{
		for (int j = 0; j<filter_1[i].size(); j++)
		{
			fprintf(fp, "%f ", filter_1[i][j]);
		}
	}
 
	for (int i = 0;i< filter_2.size(); i++)
	{
		for (int j = 0;j< filter_2[i].size(); j++)
		{
			fprintf(fp, "%f ", filter_2[i][j]);
		}
	}
 
	for (int i = 0; i<fcnn_weight_1.size(); i++)
	{
		for (int j = 0; j<fcnn_weight_1[i].size(); j++)
		{
			fprintf(fp, "%f ", fcnn_weight_1[i][j]);
		}
	}
 
	for (int i = 0; i<fcnn_weight_2.size(); i++)
	{
		for (int j = 0; j<fcnn_weight_2[i].size(); j++)
		{
			fprintf(fp, "%f ", fcnn_weight_2[i][j]);
		}
	}
	fclose(fp);
	return 0;
 
}
 
int load_Weights(const string &path, Mat2D &filter_1, Mat2D &filter_2, Mat2D &fcnn_weight_1, Mat2D &fcnn_weight_2)
{
	FILE * fp = fopen(path.c_str(), "r");
	if (fp == NULL)
	{
		printf("i can't find weight file, plz check your folder \n");
		printf("i'll train by using random data");
		
		return -1;
	}
	else
	{
		for (int i = 0; i< filter_1.size(); i++)
		{
			for (int j = 0; j<filter_1[i].size(); j++)
			{
				fscanf(fp, "%lf ", &filter_1[i][j]);
			}
		}
 
		for (int i = 0; i<filter_2.size(); i++)
		{
			for (int j = 0; j<filter_2[i].size(); j++)
			{
				fscanf(fp, "%lf ", &filter_2[i][j]);
			}
		}
 
		for (int i = 0; i<fcnn_weight_1.size(); i++)
		{
			for (int j = 0; j<fcnn_weight_1[i].size(); j++)
			{
				fscanf(fp, "%lf ", &fcnn_weight_1[i][j]);
			}
		}
 
		for (int i = 0; i < fcnn_weight_2.size(); i++)
		{
			for (int j = 0; j < fcnn_weight_2[i].size(); j++)
			{
				fscanf(fp, "%lf ", &fcnn_weight_2[i][j]);
			}
		}
		fclose(fp);
		return 0;
	}
	
}
 
int main()
{
	omp_set_num_threads(2);
	int choice = 0;
	printf("기존에 weight를 사용하실 건가요?(0(미사용) or 1(사용) : ");
	scanf("%d", &choice);
	data_read();
 
	int error = 0;
 
	//-------------------------initialize start-------------------------------------------------
 
 
	Mat3D input_image(INPUT_IMAGE_DIMENSION);
	input_image[0].resize(INPUT_IMAGE_SIZE);
	Mat1D final_out;
	double answer = 0;
	double learning_rate = 0.0001;
 
	Mat1D output(10);
 
	final_out.resize(FINAL_OUTPUT);
 
 
	//conv_1.unroll_input.resize(NUM_OF_FIRST_CONV_FILTER);
	//conv_2.unroll_input.resize(NUM_OF_SECOND_CONV_FILTER);
 
	//filter.size() = num_of_filter
	//filter[0].size() = thickness of filter =C
	//filter[0][0].size() = row_size of filter
	//filter[0][0][0].size() = col_size of filter
 
	Mat2D filter_1(5);
	for (int i = 0; i < filter_1.size(); i++)
	{
		filter_1[i].resize(25 + 1);			//bias
		Mat1D_init(filter_1[i]);
	}
 
	Mat2D filter_2(50);
	for (int i = 0; i < filter_2.size(); i++)
	{
		filter_2[i].resize(125 + 1);			//bias
		Mat1D_init(filter_2[i]);
	}
 
	Mat2D fcnn_weight_1(100);
	for (int i = 0; i < fcnn_weight_1.size(); i++)
	{
		fcnn_weight_1[i].resize(FCNN_FIRST_INPUT_SIZE + 1);
		Mat1D_init(fcnn_weight_1[i]);
	}
 
	Mat2D fcnn_weight_2(10);
	for (int i = 0; i < fcnn_weight_2.size(); i++)
	{
		fcnn_weight_2[i].resize(fcnn_weight_1.size() + 1);
		Mat1D_init(fcnn_weight_2[i]);
	}
	Mat2D weight_update_value;
	Mat2D weight_update_value2;
 
	if (choice == 1)
	{
		load_Weights("train_weight_value.txt", filter_1, filter_2, fcnn_weight_1, fcnn_weight_2);
	}
 
	//----------------- start!------------------------------
	//-----------------forward----------------------------------------------
 
	for (int train_number = 6600; train_number < 6650; train_number++)
	{
 
		cout << "train_number : " << train_number<< "	";
		input_initial(input_image[0], image[train_number], answer, train_number);
		int Now_Mat_size = INPUT_IMAGE_SIZE;
 
		conv_val conv_1, conv_2;
		pool_val pool_1, pool_2;
		fcnn_layer_val fcnn_1, fcnn_2;
 
 
		conv(input_image, conv_1, filter_1, Now_Mat_size);
		pool(conv_1, pool_1, Now_Mat_size);
		conv(pool_1.output, conv_2, filter_2, Now_Mat_size);
		pool(conv_2, pool_2, Now_Mat_size);
 
		convert_2D_to_1D(pool_2.output, fcnn_1.input);
		FCNN_forward_ReLU(fcnn_1, fcnn_weight_1);
 
		fcnn_2.input = fcnn_1.output;
 
		FCNN_forward_softmax(fcnn_2, fcnn_weight_2, answer);
 
		if (train_number < 60000)
		{
			if (train_number>100)
			{
				learning_rate = 0.00001;
			}
			//-------------------backprop------------------------------------
			FCNN_backprop(fcnn_1.delta, fcnn_2.delta, fcnn_weight_2);
 
			Mat1D temp;
			temp.resize(FCNN_FIRST_INPUT_SIZE);
 
			FCNN_backprop(temp, fcnn_1.delta, fcnn_weight_1);
 
			Pooling_backprop(pool_2, temp, 4, 4);
 
			conv_backprop(pool_2.pooling_position, filter_2, conv_2, pool_1, 8, 8, 5);
 
			Pooling_backprop(pool_1);
 
			conv_backprop(pool_1.pooling_position, filter_1, conv_1);
 
 
			//---------------- weight update--------------------------------
 
#pragma omp parallel
			{
#pragma omp sections
				{
#pragma omp section
					{
						Mat_2D_transposed(conv_1.unroll_input);
 
						Mat_mult(conv_1.unroll_input, conv_1.delta_map_2D_now, weight_update_value);
 
						Mat_2D_transposed(weight_update_value);
 
 
 
						for (int i = 0; i < filter_1.size(); i++)
						{
							for (int j = 0; j < filter_1[0].size(); j++)
							{
								filter_1[i][j] = filter_1[i][j] - (learning_rate * weight_update_value[i][j]);
							}
						}
					}
 
 
#pragma omp section
					{
						Mat_2D_transposed(conv_2.unroll_input);
						Mat_mult(conv_2.unroll_input, conv_2.delta_map_2D_now, weight_update_value2);
 
						Mat_2D_transposed(weight_update_value2);
 
						for (int i = 0; i < filter_2.size(); i++)
						{
							for (int j = 0; j < filter_2[0].size(); j++)
							{
								filter_2[i][j] = filter_2[i][j] - (learning_rate * weight_update_value2[i][j]);
							}
						}
					}
					
				}
 
 
			}
			weight_update_value.resize(0);
			weight_update_value2.resize(0);
 
 
#pragma omp parallel
{
#pragma omp sections
	{
#pragma omp section
		{
						for (int i = 0; i < fcnn_weight_1.size(); i++)
						{
							for (int j = 0; j < fcnn_weight_1[0].size() - 1; j++)
							{
								fcnn_weight_1[i][j] = fcnn_weight_1[i][j] - (learning_rate * (fcnn_1.delta[i] * fcnn_1.input[j]));
							}
							fcnn_weight_1[i][(fcnn_weight_1[0].size() - 1)] -= (learning_rate * (fcnn_1.delta[i]));
						}
		}
 
#pragma omp section
		{
					for (int i = 0; i < fcnn_weight_2.size(); i++)
					{
						for (int j = 0; j < fcnn_weight_2[0].size() - 1; j++)
						{
							fcnn_weight_2[i][j] = fcnn_weight_2[i][j] - (LEARNING_RATE * (fcnn_2.delta[i] * fcnn_2.input[j]));
						}
						fcnn_weight_2[i][(fcnn_weight_2[0].size() - 1)] -= (LEARNING_RATE * (fcnn_2.delta[i]));
					}
		}
	}
}
 
 
		}
 
		else
		{
			//for (int num = 0; num < 10; num++)
			//{
				//cout << "answer : " << answer << endl;
				//cout << "output[" << num << "] : " << fcnn_2.output[num] << endl;
			//}
		}
 
		final_value_initialize(conv_1, pool_1, fcnn_1);
		final_value_initialize(conv_2, pool_2, fcnn_2);
 
		//cout << "answer : " << answer <<"   success : "<< answer_success << endl;
		
	}
	save_Weights("train_weight_value.txt", filter_1, filter_2, fcnn_weight_1, fcnn_weight_2);	
 
 
}
 
