//
//  Matrix.c
//  Neural Net
//
//  Created by Mauricio Paulusma on 13/01/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#include <stdio.h>
#include "matrix.h"

/*
 Name function: give_matrix
 
 Description:
 This function returns the value of the given member (member: row, col) of the given matrix pm
 
 Usage:
 float *pm = pointer to the matrix
 int scol = number of collums in the matrix
 int row = the row in the matrix
 int col = the collum in the matrix
 
 Returns:
 value at the given place in the given matrix in floating point format
 
 */
float give_matrix(float *pm, int scol, int row, int col)
{
    return *(pm+(scol*row)+col);
}

/*
 Name function: set_matrix
 
 Description:
 This function set the given value to the given member (member: row, col) of the given matrix pm
 
 Usage:
 float *pm = pointer to the matrix
 int scol = number of collums in the matrix
 int row = the row in the matrix
 int col = the collum in the matrix
 float val = the value to set on the given place in the matrix
 
 */
void set_matrix(float *pm, int scol, int row, int col, float val)
{
    *(pm+(scol*row)+col) = val;
}

/*
 Name function: print_matrix
 
 Description:
 This function prints the content of the given matrix (block of memory should be declared as floats)
 
 Usage:
 *pm = pointer to matrix
 srow = number of rows
 scol = number of collums
 
 Returns: nothing
 
 */
void print_matrix(float *pm, int srow, int scol)
{
    for (int i = 0; i < srow; i++)
    {
        for (int j = 0; j < scol; j++)
        {
            printf("m[%d][%d] = %f\t", i, j, give_matrix(pm, scol, i, j));
        }
        printf("\n");
    }
}


/*
 Name function: matrix_mult
 
 Description:
 This function calculates the product of matrix pma and pmb (according to matrix multiplication: https://en.wikipedia.org/wiki/Matrix_multiplication) and stores the result in matrix pmc
 I.e.: *pmc = *pma * *pmb
 
 Usage:
 *pma = pointer matrix a
 *pmb = pointer matrix b
 *pmc = pointer matrix c
 srowa = number of rows in matrix a
 scola = number of collums in matrix a
 srowb = number of rows in matrix b
 scolb = number of collums in matrix b
 
 Returns: nothing
 
 */
void matrix_mult(float *pmc, float *pma, float *pmb, int srowa, int scola, int srowb, int scolb)
{
    int srowc = srowa; // assign them new names for clarification
    int scolc = scolb;
    for (int i = 0; i < srowc; i++) // this increments in the rows of matrix c
    {
        for (int j = 0; j < scolc; j++) // this increments in the collums of matrix c
        {
            float accum = 0;
            for (int k = 0; k < scola; k++) // This for loop calculates the the product of matrix a and matrix b
            {
                accum += give_matrix(pma, scola, i, k) * give_matrix(pmb, scolb, k, j); // calculate the product and accumulate
            }
            set_matrix(pmc, scolb, i, j, accum); // assign the accumulated value to matrix c
        }
    }
}

/*
 Name function: matrix_subt
 
 Description:
 This function calculates the result of the subtraction of matrix pma and pmb and stores the result in matrix pmc.
 I.e. *pmc = *pma - *pmb
 
 
 Usage:
 *pma = pointer to matrix a
 *pmb = pointer to matrix b
 *pmc = pointer to matrix c
 srow = number of rows of both matrices
 scol = number of collums of both matrices
 
 */
void matrix_subt(float *pmc, float *pma, float *pmb, int srow, int scol)
{
    for (int i = 0; i < srow; i++)
    {
        for (int j = 0; j < scol; j++)
        {
            float accum = give_matrix(pma, scol, i, j) - give_matrix(pmb, scol, i, j);
            set_matrix(pmc, scol, i, j, accum);
        }
    }
}


/*
 Name function: matrix_add
 
 Description:
 This function calculates the sum of matrix pma and pmb and stores the result in matrix pmc
 I.e.: *pmc = *pma + *pmb
 
 Usage:
 *pma = pointer to matrix a
 *pmb = pointer to matrix b
 *pmc = pointer to matrix c
 srow = number of rows of both matrices
 scol = number of collums of both matrices
 
 */
void matrix_add(float *pmc, float *pma, float *pmb, int srow, int scol)
{
    for (int i = 0; i < srow; i++)
    {
        for (int j = 0; j < scol; j++)
        {
            float accum = give_matrix(pma, scol, i, j) + give_matrix(pmb, scol, i, j);
            set_matrix(pmc, scol, i, j, accum);
        }
    }
}

/*
 Name function: matrix_hadamard
 
 Description:
 This function applies the Hadamard product to matrix a and b and stores te result in matrix c
 
 Usage
 *pma = pointer to matrix a
 *pma = pointer to matrix b
 *pma = pointer to matric c
 
 Returns: nothing
 
 */
void matrix_hadamard(float *pmc, float *pma, float *pmb, int s_row, int s_col)
{
    for (int i = 0; i < s_row; i++)
    {
        for (int j = 0; j < s_col; j++)
        {
            set_matrix(pmc, s_col, i, j, give_matrix(pmb, s_col, i, j)*give_matrix(pma, s_col, i, j));
        }
    }
}


/*
 Name function:  matrix_sigmoid_prime
 
 Description:
 This function applies the sigmoid_prime function to the given matrix.
 
 Usage:
 *pmz = pointer to the matrix with the weighted inputs (z)
 *pmr = pointer to the matrix where the result should be stored
 
 Returns: nothing
 
 */
void matrix_sigmoid_prime(float *pmz, float *pmr, int srow, int scol)
{
    for (int i = 0; i < srow; i++)
    {
        for (int j = 0; j < scol; j++)
        {
            float z = give_matrix(pmz, scol, i, j);
            float a = sigmoid_prime(z);
            set_matrix(pmr, scol, i, j, a);
        }
    }
}



/*
 Name function:  matrix_sigmoid
 
 Description:
 This function applies the sigmoid function to the given matrix.
 
 Usage:
 *pmz = pointer to the matrix with the weighted inputs (z)
 *pma = pointer to the matrix with the activation layer
 
 Returns: nothing
 
 */
void matrix_sigmoid(float *pmz, float *pma, int srow, int scol)
{
    for (int i = 0; i < srow; i++)
    {
        for (int j = 0; j < scol; j++)
        {
            float z = give_matrix(pmz, scol, i, j);
            float a = sigmoid(z);
            set_matrix(pma, scol, i, j, a);
        }
    }
}


void init_matrix(float *pm, int size_row, int size_col)
{
    printf("init_matrix function activated...");
    
    for (int i = 0; i < size_row; i++)
    {
        for (int j = 0; j < size_col; j++)
        {
            printf("type in value for[%d][%d]", i, j);
            char c = getchar();
            getchar();
            set_matrix(pm, size_col, i, j, c - 48);
        }
    }
    
    printf("the created matrix:\n");
    print_matrix(pm, size_row, size_col);
}


/*
 Name function: matrix_transpose
 
 Description
 This function calculates the transpose of matrix a and puts it into matrix b
 
 Returns: nothing
 
 */

void matrix_transpose(void *pma, void *pmb, int n_row, int n_col)
{
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            set_matrix(pmb, n_row, j, i, give_matrix(pma, n_col, i, j));
        }
    }
}
