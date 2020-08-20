

/*

Yuzhe Shang, Project 6, blur the dataset
ECE 350/450 Accelerated Computing for Deep Learning
17:00/May 08/2020
References:
https://github.com/csuldw/MachineLearning/blob/master/utils/data_util.py (Read and write the MNIST)
http://www.voidcn.com/article/p-rzninoud-bbs.html (Drag data from MNIST using C++)
https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%203.pdf (Image Blur as 2D kernal)
https://www.cnblogs.com/hjj-fighting/p/10429178.html (How to write the dataset to a file using C++)
timer.h and the timing method used in 01-nbody.cu.
01-nbody-gpu.cu
https://blog.csdn.net/u010579901/article/details/78852879


*/

/*First part is to read the MNIST dataset*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "timer.h"//All the head files below are from nbody-gpu.cu
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#define DimBlock 256 //Define the Dimblock
#define n 28 // Define the thread_per_block
using namespace std;
 
int ReverseInt(int i)  //Integer reverse, change the data into binary
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
 

 
void read_Mnist_Images(vector<vector <double> > &images) //Read dataset
{
	ifstream file("t10k-images.idx3-ubyte", ios::binary);  //Read the MNIST in binary
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
	
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

 		cout << "magic number = " << magic_number << endl; // show magic number
		cout << "number of images = " << number_of_images << endl; // show inmage numbers
		cout << "rows = " << n_rows << endl; //show number of rows
		cout << "cols = " << n_cols << endl; //show number of cols


		for (int i = 0; i < 10000; i++)   //number_of_images, read 10000 images
		{
			vector<double>tp;
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 1;
					file.read((char*)&image, sizeof(image));
					tp.push_back(image);
				}
			}
			images.push_back(tp);
		}
	}
}


/*GPU kernel code, initiated by CPU, cannot be called by other kernel*/
__global__
void blurkernel (unsigned char *in, unsigned char *out, int w, int h)
{

int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

if (col < h && row < w )
{

int pixVal = 0;
int pixels = 0; 
int BLUR_SIZE = 1; // Define the blur size as 1
for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE+1; ++blurcol){
 for (int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE+1; ++blurrow)
{
 int currow = row + blurrow;
 int curcol = col + blurcol;

  if (currow > -1 && currow < w && curcol > -1 && curcol < h)
{
 pixVal += in[curcol * w + currow];

 pixels++;   //Keep track of number of pixels in the accumulated total
     }
   }
 }

out[col * w + row] = (unsigned char)(pixVal / pixels);  //Write our new pixel value out

   }
                                           
}

/*use main function*/
int main()
{	
	vector< vector<double> > images;
	read_Mnist_Images(images);   //Read images


 ofstream outFile("train5.idx3-ubyte",  ios::binary);  //use ofstream to create a new file named train5.idx3-ubyte
   
 for (int i = 0; i < images.size(); i++)
	{
		for (int j = 0; j < images[0].size(); j++)   //for (auto iter = labels.begin(); iter != labels.end(); iter++)
		{ 
        	
		}
	}



/*Image Blur 10 times as a 2D Kernel*/

/*unsigned char in, unsigned char out, int w, int h*/


unsigned char *in; //Define the size of char in and out
unsigned char *out;
int w = images.size(); //Define the size of height and width
int h = images[0].size();


int size = 250000 *n * n * sizeof (unsigned char);// Changed the float to char, increase the memory
 
  cudaMallocManaged (&out, size);
  cudaMallocManaged (&in, size);

for(int col =0;col < w ;col++)
{
for(int row =0;row < h ;row++)
{
in[col*w+row] = images[col][row];                                      
out[col*w+row] = images[col][row]; 
}
}

/*code below are from the 01-nbody-gpu.cu*/
const int nIters = 1;//Blur one time
double totalTime = 0.0;

 for (int iter = 1; iter <= nIters; iter++) {
  StartTimer();


/*Blur the image*/
for(int col =0;col < w ;col++)
{
for(int row =0;row < h ;row++)
{

//in [col*w+ row] =  out [col*w+ row]    ;                         



}
}

  dim3 Dimblock (28, 28, 1); // 2D, n*n*1 
  dim3 Dimgrid ((n - 1 / Dimblock.x) + 1, (n - 1/ Dimblock.y) + 1, 1); //Total 3 dimension, I used 2 dims

 blurkernel <<<Dimblock, Dimblock>>> (in, out, 28, 28); //Write all parameters in the kernal

cudaDeviceSynchronize();// synchronized, Waiting for GPU kernel execution to end
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;

}


for(int col =0;col < w ;col++)
{
for(int row =0;row < h ;row++)
{

                               
outFile.write((char*)&out[col*w+ row], sizeof(out[col*w+ row]));


}
}

cout <<totalTime  << " ";
	return 0;
 

}
