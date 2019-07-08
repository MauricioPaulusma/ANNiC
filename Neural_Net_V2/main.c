//
//  main.c
//  Neural_Net_V2
//
//  Created by Mauricio Paulusma on 07/07/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "main.h"
#include "feedforward.h"
#include "matrix.h"
#include "random.h"

int load_training_data(struct image *pstruct, int nr_of_img); // prototype for load_training_data function

int neurons [] = {784,10,10,10};

struct image training_data[TRAINING_SIZE]; // create structures for storing the training data

int main(int argc, const char * argv[])
{
    printf("Program started..\n");

    int ret = load_training_data(&training_data[0], TRAINING_SIZE);
    if(ret == 0)
        return 0;
    else
        printf("training data loaded succesfully\n");
    
    // Create the neural net
    // array of pointers for the memory given by malloc
    float *pwl[LAYERS];
    float *pbl[LAYERS];
    float *pzl[LAYERS];
    float *pal[LAYERS];
    
    // create the matrices for each layer and assign the start adress of each matrix to the pointers
    pal[0] = malloc(sizeof(float)*neurons[0]*1); // the first layer only has an activation layer
    for (int i = 1; i < LAYERS; i++)
    {
        pwl[i] = malloc(sizeof(float)* neurons[i] *neurons[i-1]);
        pbl[i] = malloc(sizeof(float)* neurons[i] * 1);
        pzl[i] = malloc(sizeof(float)* neurons[i] * 1);
        pal[i] = malloc(sizeof(float)* neurons[i] * 1);
    }
    
    for (int i = 1; i < LAYERS; i++) // initialize the weights and the biases with random values
    {
        init_rand(pwl[i], neurons[i]*neurons[i-1]);
        init_rand(pbl[i], neurons[i]);
    }
    
    for(int i = 0; i < TRAINING_SIZE; i++)
    {
        feedforward(&training_data[i], &pwl[0], &pbl[0], &pzl[0], &pal[0], &neurons[0], 1);
    }
    
    printf("quiting program...");
    return 1;
}

/*
 This function loads the training data into the given structures
 */
int load_training_data(struct image *pstruct, int nr_of_img)
{
    
    printf("loading training data....\n");
    
    FILE *fp;
    FILE *fpl;
    
    fp = fopen("train-images.idx3-ubyte", "r");
    fpl = fopen("train-labels.idx1-ubyte", "r");
    
    if(fp == NULL || fpl == NULL)
    {
        printf("Failed to open the file\n");
        return 0;
    }
    
    for (int i = 0; i < 4; i++)
    {
        int number = 0;
        number = number | (fgetc(fp) << 24);
        number = number | (fgetc(fp) << 16);
        number = number | (fgetc(fp) << 8);
        number = number | (fgetc(fp));
        printf("number = %d\n", number);
    }
    
    for (int i = 0; i < 2; i++)
    {
        int number = 0;
        number = number | (fgetc(fpl) << 24);
        number = number | (fgetc(fpl) << 16);
        number = number | (fgetc(fpl) << 8);
        number = number | (fgetc(fpl));
        printf("number = %d\n", number);
    }
    
    for (int i = 0; i < nr_of_img; i++, pstruct++)
    {
        pstruct->digit = fgetc(fpl);
        
#ifdef DEBUG_LTD
        printf("acquired digit = %d\n", pstruct->digit);
#endif
        
        for (int j = 0; j < ROWS; j++)
        {
            for (int k = 0; k < COLL; k++)
            {
                pstruct->pixels[j][k] = fgetc(fp);
            }
        }
    }
    return 1;
}
