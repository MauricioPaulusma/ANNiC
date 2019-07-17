//
//  File_IO.c
//  Neural_Net_V2
//
//  Created by Mauricio Paulusma on 17/07/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//


#include "main.h"
#include "File_IO.h"
#include <stdio.h>
#include "matrix.h"

int save_net(struct neural_net* ANN, char* file_name)
{
    FILE* fp;
    fp = fopen(file_name, "w");
    if(fp == NULL)
        return 0;
    
    int ret = fputc(ANN->nr_of_layers, fp); // first byte of the file is the number of layers of the neural net
    if(ret == 0)
        return 0;
    
    for (int i = 0; i<ANN->nr_of_layers; i++) // next nr_of_layers bytes define how much neurons each layers contains
    {
        int ret = fputc((*(ANN->neurons+i)), fp);
        if(ret == 0)
            return 0;
    }
    
    for (int i = 1; i<ANN->nr_of_layers; i++) // start at layer 1 because the input layer doesn't have any weights or biases
    {
        fwrite(ANN->pwl[i], sizeof(float), (*(ANN->neurons+i-1))*(*(ANN->neurons+i)), fp); // write the weights from wl[i] to the file pointed by fp
        fwrite(ANN->pbl[i], sizeof(float), (*(ANN->neurons+i)), fp); // write the biases from bl[i] to the file pointed by fp
    }
    return 1;
}

int load_net(struct neural_net* ANN, char* file_name)
{
    FILE* fp;
    fp = fopen(file_name, "r");
    if(fp == NULL)
        return 0;
    
    ANN->nr_of_layers = fgetc(fp); // acquire the number of layers
    
    for (int i = 0; i < ANN->nr_of_layers; i++)
    {
        *(ANN->neurons+i) = fgetc(fp); // acquire the amount of neurons of each layer
    }
    
    for (int i = 1; i < ANN->nr_of_layers; i++)
    {
        fread(ANN->pwl[i], sizeof(float), (*(ANN->neurons+i-1))*(*(ANN->neurons+i)), fp); // read the weights from the file pointed by fp and save it to wl[i]
        fread(ANN->pbl[i], sizeof(float), (*(ANN->neurons+i)), fp); // read the biases from the file pointed by fp and save it to bl[i]
    }
    
    return 1;
}

void print_net(struct neural_net* ANN)
{
    printf("\n Number of layers: %d\n", ANN->nr_of_layers);
    printf("layers = {");
    for (int i = 0; i<ANN->nr_of_layers; i++)
    {
        printf(" %d,", *(ANN->neurons+i));
    }
    printf("}\n");
    
    for (int i = 1; i < ANN->nr_of_layers; i++)
    {
        printf("\nWL[%d]:\n", i);
        print_matrix(ANN->pwl[i], *(ANN->neurons+i), *(ANN->neurons+i-1));
        printf("\nBL[%d]:\n", i);
        print_matrix(ANN->pbl[i], *(ANN->neurons+i), 1);
    }
}
