//
//  feedforward.c
//  Neural Net
//
//  Created by Mauricio Paulusma on 13/01/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#include "stdio.h"
#include "main.h"
#include "feedforward.h"
#include "matrix.h"

/*
 Name function: load_image_in_a
 
 Description:
 This function loads the image into activation layer 0
 
 Usage:
 float *pm = pointer to the activation layer 0
 struct image *pimage = pointer to the image
 
 */
void load_image_in_a(float *pma, struct image *pimage)
{
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLL; j++)
        {
            set_matrix(pma, 1, ((i)*COLL+j), 0, pimage->pixels[i][j]);
        }
    }
}


/*
 Name function: feedforward
 
 Description:
 This function feedforwards trough the neural network given by the pointers
 and according to the given image.
 
 Usage:
 *pimage = pointer to the structure holding the image data
 **pwl = pointer to the weight layers
 **pbl = pointer to the bias layers
 **pzl = pointer to the weighted input layers
 **pal = pointer to the activation layers
 
 returns: nothing
 */
void feedforward(struct image *pimage, float **pwl, float **pbl, float **pzl, float **pal, int* neurons, int FF_Evaluate)
{
    
#ifndef FF_CHECK
    load_image_in_a(*pal, pimage); // load al0 (input layer) with the image data
#ifdef DEBUG_FF
    printf("Input to neural net: %d", pimage->digit);
#endif
#else
    printf("feedforward function is in hard debugging mode\n");
#endif
    
    for (int i = 1; i < LAYERS; i++) // feedforward trough the network
    {
        
        matrix_mult(*(pzl+i), *(pwl+i), *(pal+i-1), *(neurons+i), *(neurons+i-1), *(neurons+i-1), 1); // zl[i] = wl[i] * al[i-1];
        
#ifdef DEBUG_FF
        printf("\n Weight matrix[%d] = \n", i);
        print_matrix(*(pwl+i), *(neurons+i), *(neurons+i-1));
        
        printf("\n 1. Activation matrix[%d] = \n", i-1);
        print_matrix(*(pal+(i-1)), *(neurons+i-1), 1);
        
        printf("\n Weighted input matrix[%d]\n", i);
        print_matrix(*(pzl+i), *(neurons+i), 1);
        
        printf("\n Bias matrix[%d]\n", i);
        print_matrix(*(pbl+i), *(neurons+i), 1);
#endif
        
        matrix_add(*(pzl+i), *(pzl+i), *(pbl+i),  *(neurons+i), 1); //   zl[i] = zl[i] + bl[i]
        matrix_sigmoid(*(pzl+i), *(pal+i),  *(neurons+i), 1); // apply the sigmoid function to the matrix zl[i] and store the result in pal[i]
        
#ifdef DEBUG_FF
        printf("\n Weighted input matrix after addition[%d]\n", i);
        print_matrix(*(pzl+i), *(neurons+i), 1);
        
        printf("\n 2. Activation matrix[%d]\n", i);
        print_matrix(*(pal+i), *(neurons+i), 1);
#endif
        
    }
    
    if (FF_Evaluate == 1)
    {
        printf("\nImage into FF_Function (input to network): %d\n", pimage->digit);
        printf("Output from FF_Function (output from network):\n");
        print_matrix(*(pal+(LAYERS-1)),  *(neurons+LAYERS-1), 1);
    }
    
}
