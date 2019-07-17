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
#include "SGD.h"
#include "File_IO.h"

/*
    Function prototypes
*/
int load_training_data(struct image *pstruct, int nr_of_img);
void print_images(struct image *pstruct, int nr_of_img, int rows, int cols);
int load_test_data(struct image *pstruct, int nr_of_img);

/*
    Global Variables
*/
struct image training_data[TRAINING_SIZE]; // create structures for storing the training data
struct image test_data[TEST_SIZE]; // create structures for storing the test data

int neurons [] = {784,10,10,10};

/*
    Main
*/
int main(int argc, const char* argv[])
{
    printf("Program started..\n");

    int ret = load_training_data(&training_data[0], TRAINING_SIZE);
    if(ret == 0)
        return 0;
    else
        printf("training data loaded succesfully\n");

    //print_images(&training_data[0], 100, ROWS, COLL);
    
    ret = load_test_data(&test_data[0], TEST_SIZE);
    if(ret == 0)
        return 0;
    else
        printf("test data loaded succesfully\n");
    
    //print_images(&test_data[0], 100, ROWS, COLL);
    
    


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

    load_net(&ANN, "Neural");
    print_net(&ANN);
 
    
    //
//    for (int i = 1; i < ANN.nr_of_layers; i++) // initialize the weights and the biases with random values
//    {
//        init_rand(ANN.pwl[i], (*(ANN.neurons+i))*(*(ANN.neurons+i-1)));
//        init_rand(ANN.pbl[i], (*(ANN.neurons+i)));
//    }
//
//    for (int j = 0; j<EPOCH; j++)
//    {
//        for (int i = 0; i<TRAINING_SIZE/BATCH_SIZE; i++)
//        {
//            update_minibatch2(&training_data[i*BATCH_SIZE], &ANN, BATCH_SIZE, ETA);
//        }
//        printf("\nepoch done: %d/%d", j+1, EPOCH);
//
//        printf("\ncalculating accuracy\n");
//        int correct = 0;
//        for (int i = 0; i<TEST_SIZE; i++)
//        {
//            int ret = feedforward2(&test_data[i], &ANN, 1, 0);
//            correct = correct + ret;
//        }
//        printf("correct = %d/%d\n", correct, TEST_SIZE);
//
//        data_shuffle(&training_data[0], TRAINING_SIZE); // shuffle the training data
//    }
//    printf("\ndone training\n");
//
//    print_net(&ANN);
//
//    printf("\ntype in filename to save the net\n");
//    char filename[100];
//    scanf("%s", &filename[0]);
//    save_net(&ANN, &filename[0]);
    
    printf("quiting program...");
    return 1;
}

void print_images(struct image *pstruct, int nr_of_img, int rows, int cols)
{
    
    for (int i = 0; i< nr_of_img; i++, pstruct++)
    {
        printf("\nloaded image:%d\n", pstruct->digit);
        for (int j = 0; j<rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                if(pstruct->pixels[j][k] > 0)
                    printf("1");
                else
                    printf("0");
            }
            printf("\n");
        }
    }
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

/*
 This function loads the test data into the given structures
 */
int load_test_data(struct image *pstruct, int nr_of_img)
{
    
    printf("loading test data....\n");
    
    FILE *fp;
    FILE *fpl;
    
    fp = fopen("t10k-images.idx3-ubyte", "r");
    fpl = fopen("t10k-labels.idx1-ubyte", "r");
    
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


/*
 code for testing purposes
 */


//    float* pdCdW[LAYERS];
//    float* pdCdB[LAYERS];
//    for (int i = 1; i<LAYERS; i++)
//    {
//        pdCdW[i] = malloc(sizeof(float)*(*(ANN.neurons+i))*(*(ANN.neurons+i-1)));
//        pdCdB[i] = malloc(sizeof(float)*(*(ANN.neurons+i)));
//    }
//
//    backpropagation2(NULL, &pdCdW[0], &pdCdB[0], &ANN, 1);
//    printf("\nBACK IN MAIN\n");
//
//    for (int i = 1; i<LAYERS; i++)
//    {
//        printf("\ndCdB[%d]:\n",i);
//        print_matrix(pdCdB[i], (*(ANN.neurons+i)), 1);
//        printf("\ndCdW[%d]:\n",i);
//        print_matrix(pdCdW[i], (*(ANN.neurons+i)), (*(ANN.neurons+i-1)));
//    }
//

