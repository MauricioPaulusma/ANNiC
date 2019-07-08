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

/*
    Function prototypes
*/
int load_training_data(struct image *pstruct, int nr_of_img); // prototype for load_training_data function

/*
    Global Variables
*/
struct image training_data[TRAINING_SIZE]; // create structures for storing the training data
int neurons [] = {784,10,10};

/*
    Main
*/
int main(int argc, const char * argv[])
{
    printf("Program started..\n");

    int ret = load_training_data(&training_data[0], TRAINING_SIZE);
    if(ret == 0)
        return 0;
    else
        printf("training data loaded succesfully\n");
    
    struct neural_net ANN; // Creating a Neural Net
    
    ANN.nr_of_layers = LAYERS; // specify the amount of layers
    ANN.neurons = &neurons[0]; // assign the adress of the neurons array containing the size information of each layer
    
    // create the matrices for each layer and assign the start adress of each matrix to the pointers
    ANN.pal[0] = malloc(sizeof(float)*(*ANN.neurons+0)*1); // the first layer only has an activation layer
    for (int i = 1; i < ANN.nr_of_layers; i++)
    {
        ANN.pwl[i] = malloc(sizeof(float) * (*(ANN.neurons+i))*(*(ANN.neurons+i-1)));
        ANN.pbl[i] = malloc(sizeof(float) * (*(ANN.neurons+i)) * 1);
        ANN.pzl[i] = malloc(sizeof(float) * (*(ANN.neurons+i)) * 1);
        ANN.pal[i] = malloc(sizeof(float) * (*(ANN.neurons+i)) * 1);
    }
    
    for (int i = 1; i < ANN.nr_of_layers; i++) // initialize the weights and the biases with random values
    {
        init_rand(ANN.pwl[i], (*(ANN.neurons+i))*(*(ANN.neurons+i-1)));
        init_rand(ANN.pbl[i], (*(ANN.neurons+i)));
    }

#ifdef FF_CHECK
    for (int i = 1; i < ANN.nr_of_layers; i++)
    {
        printf("printing layer %d:\n", i);
        print_matrix(ANN.pwl[i], *(ANN.neurons+i), *(ANN.neurons+i-1));
        printf("\n");
        print_matrix(ANN.pbl[i], *(ANN.neurons+i), 1);
    }

    init_matrix(ANN.pal[0], *ANN.neurons+0, 1);
#endif
    feedforward2(&training_data[0], &ANN, 1);
//    for(int i = 0; i < TRAINING_SIZE; i++)
//    {
//        feedforward(&training_data[i], &pwl[0], &pbl[0], &pzl[0], &pal[0], &neurons[0], 1);
//    }
    
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
