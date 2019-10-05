#include "allocation.hpp"
#include <stdio.h>
int** alloc_2d_init(int r, int c)
{
	int** A = new int*[r];
	A[0] = new  int[r*c];
	for (int i = 1; i < r; ++i) 
		A[i] = A[i-1] + c;
	return A;
}

void data_init(int** A, int r, int c)
{
	for(int i=0; i<r; i++)
	{
		for(int j=0; j<c; j++)
		{
			A[i][j]=i*c+j;
		}
	}
}

void print_helper(int** A, int r, int c)
{
	
	for(int i=0; i<r; i++)
	{
		for(int j=0; j<c; j++)
		{
			printf("%d, ", A[i][j]);
		}
	printf("\n");
	}
}
