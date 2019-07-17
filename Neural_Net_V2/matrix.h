//
//  matrix.h
//  Neural_Net_V2
//
//  Created by Mauricio Paulusma on 08/07/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#ifndef matrix_h
#define matrix_h

float give_matrix(float *pm, int scol, int row, int col);
void matrix_transpose(void *pma, void *pmb, int n_row, int n_col);
void init_matrix(float *pm, int size_row, int size_col);
void matrix_sigmoid(float *pma, float *pmz, int srow, int scol);
void matrix_sigmoid_prime(float *pmz, float *pmr, int srow, int scol);
void matrix_hadamard(float *pmc, float *pma, float *pmb, int s_row, int s_col);
void matrix_add(float *pmc, float *pma, float *pmb, int srow, int scol);
void matrix_subt(float *pmc, float *pma, float *pmb, int srow, int scol);
void matrix_mult(float *pmc, float *pma, float *pmb, int srowa, int scola, int srowb, int scolb);
void print_matrix(float *pm, int srow, int scol);
void set_matrix(float *pm, int scol, int row, int col, float val);
float give_matrix(float *pm, int scol, int row, int col);
float sigmoid_prime(float z);
float sigmoid(float z);
void memclear(void *pmem, int size);
int highest_index(void* pmatrix, int n_row);

#endif /* matrix_h */
