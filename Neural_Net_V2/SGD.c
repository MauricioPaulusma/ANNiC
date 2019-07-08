//
//  SGD.c
//  Neural Net
//
//  Created by Mauricio Paulusma on 13/01/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#include "SGD.h"
#include "matrix.h"
#include "feedforward.h"
#include "sigmoid.h"
#include "main.h"

/*
 Name function: SGD
 
 Description:
 This function applies the Stochastic Gradient Descent (SGD) Algorithm to the Neural Network.
 
 Usage:
 struct image* pimage = pointer to the beginning of the image dataset
 float **pwl = pointer to the array with the addresses of the weight layer memory
 float **pbl = pointer to the array with the addresses of the bias layer memory
 float **pzl = pointer to the array with the addresses of the weighted input layer memory
 float **pal = pointer to the array with the addresses of the activation layer memory
 
 */
void SGD(struct image* pimage, float **pwl, float **pbl, float **pzl, float **pal)
{
    
    for (int i = 0; i < EPOCH; i++) // for loop for the epochs (times the net is trained with the entire training data set)
    {
        data_shuffle(pimage, TRAINING_SIZE); // shuffle the training data
#ifdef Evaluate_FF
        feedforward(pimage, pwl, pbl, pzl, pal, 1);
#endif
        struct image* pimage_buf = pimage; // make a buffer so the pointer can be set back after all the minibatches have been used for training
        for (int j = 0; j < TRAINING_SIZE/BATCH_SIZE; j++, pimage += BATCH_SIZE) // split the training data into mini batches
        {
            update_minibatch2(pimage, pwl, pbl, pzl, pal, BATCH_SIZE, ETA); // apply SGD to a mini batch
        }
        pimage = pimage_buf; // place the pointer back to the start adress
    }
}

/*
 Name function: update_minibatch2
 
 Description:
 This function updates the Weights and the Biases in the neural net according to the given learning rate (eta).
 It calculates the dC/dW and the dC/dB over a number of images set by argument batch_size.
 
 Usage:
 struct image *pimage = pointer to the begin of the images
 float **pwl = pointer to the weight layers
 float **pbl = pointer to the bias layers
 float **pzl = pointer to the weighted input layers
 float **pal = pointer to activation layers
 int batch_size = number of images where the dC/dW and the dC/dB will be averaged over.
 float eta = learning rate
 
 */
void update_minibatch2(struct image *pimage, float **pwl, float **pbl, float **pzl, float **pal, int batch_size, float eta)
{
    
    float *dCdW[LAYERS]; // dCdW for backpropagation function
    float *dCdB[LAYERS]; // dCdB for backpropagation function
    float *Acc_dCdW[LAYERS]; // dCdW for accumulating the dCdW values
    float *Acc_dCdB[LAYERS]; // dCdB for accumulating the dCdB values
    
    for (int i = 1; i < LAYERS; i++)
    {
        dCdW[i] = malloc(sizeof(float) * neurons[i-1]*neurons[i]);
        dCdB[i] = malloc(sizeof(float) * neurons[i]);
        memclear(dCdW[i], sizeof(float) * neurons[i-1]*neurons[i]); // initialize the memory to 0
        memclear(dCdB[i], sizeof(float) * neurons[i]); // initialize the memory to 0
        
        Acc_dCdW[i] = malloc(sizeof(float) * neurons[i-1]*neurons[i]);
        Acc_dCdB[i] = malloc(sizeof(float) * neurons[i]);
        memclear(Acc_dCdW[i], sizeof(float) * neurons[i-1]*neurons[i]); // initialize the memory to 0
        memclear(Acc_dCdB[i], sizeof(float) * neurons[i]); // initialize the memory to 0
    }
    
    for (int i = 0; i < batch_size; i++, pimage++)
    {
        backpropagation2(pimage, &dCdW[0], &dCdB[0], pwl, pbl, pzl, pal);
        for (int j = 1; j < LAYERS; j++)
        {
            matrix_add(Acc_dCdW[j], Acc_dCdW[j], dCdW[j], neurons[j], neurons[j-1]);
            matrix_add(Acc_dCdB[j], Acc_dCdB[j], dCdB[j], neurons[j], 1);
        }
    }
    
    // this block of code calculates the average dCdB and dCdW
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            // calculate the average dCdB per neuron
            float valb = give_matrix(Acc_dCdB[i], 1, j, 0);
#ifdef DEBUG_UM
            printf("dC/dB(accum)[%d][%d] = %f\n", i, j, valb);
#endif
            valb = valb / batch_size*1.0;
#ifdef DEBUG_UM
            printf("dC/dB|(mean)[%d][%d] = %f\n", i, j, valb);
#endif
            set_matrix(Acc_dCdB[i], 1, j, 0, valb);
            
            // calculate the average dCdW for each weight of the neuron
            for (int k = 0; k < neurons[i-1]; k++)
            {
                float valw = give_matrix(Acc_dCdW[i], neurons[i-1], j, k);
#ifdef DEBUG_UM
                printf("dC/dW(accum)[%d][%d][%d] = %f\n", i, j, k, valw);
#endif
                valw = valw / batch_size*1.0;
#ifdef DEBUG_UM
                printf("dC/dW(mean)[%d][%d][%d] = %f\n", i, j, k, valw);
#endif
                set_matrix(Acc_dCdW[i], neurons[i-1], j, k, valw);
            }
        }
    }
    
#ifdef Evaluate_UM
    printf("\nEvaluating dC/dB...\n");
    for (int i = 0; i < neurons[LAYERS-1]; i++)
    {
        printf("dC/dB[%d][%d] = %f\n", LAYERS-1, i, *(dCdB[LAYERS-1]+i));
    }
#endif
    
    // This block of code adjusts the weights and biases according to the calculated dC/dW and dC/dB
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            set_matrix(*(pbl+i), 1, j, 0, give_matrix(*(pbl+i), 1, j, 0) + (*(Acc_dCdB[i]+j) * (-eta))); // takes the current value of the Bias and accumulates it with dC/dB*(-eta). I.e. Bias = Bias + (dC/dB*(-eta))
            for (int k = 0; k < neurons[i-1]; k++)
            {
                set_matrix(*(pwl+i), neurons[i-1], j, k, give_matrix(*(pwl+i), neurons[i-1], j, k) + give_matrix(Acc_dCdW[i], neurons[i-1], j, k) * (-eta)); // takes the current value of the weight and accumulates it with dC/dW*(-eta). I.e. Weight = Weight + (dC/dW*(-eta))
            }
        }
    }
}


/*
 Name function: update_minibatch
 
 Description:
 This function updates the Weights and the Biases in the neural net according to the given learning rate (eta).
 It calculates the dC/dW and the dC/dB over a number of images set by argument batch_size.
 
 Usage:
 struct image *pimage = pointer to the begin of the images
 float **pwl = pointer to the weight layers
 float **pbl = pointer to the bias layers
 float **pzl = pointer to the weighted input layers
 float **pal = pointer to activation layers
 int batch_size = number of images where the dC/dW and the dC/dB will be averaged over.
 float eta = learning rate
 
 */
void update_minibatch(struct image *pimage, float **pwl, float **pbl, float **pzl, float **pal, int batch_size, float eta)
{
    
    float *dCdW[LAYERS];
    float *dCdB[LAYERS];
    
    for (int i = 1; i < LAYERS; i++)
    {
        dCdW[i] = malloc(sizeof(float) * neurons[i-1]*neurons[i]);
        dCdB[i] = malloc(sizeof(float) * neurons[i]);
        memclear(dCdW[i], sizeof(float) * neurons[i-1]*neurons[i]); // initialize the memory to 0
        memclear(dCdB[i], sizeof(float) * neurons[i]); // initialize the memory to 0
    }
    
    for (int i = 0; i < batch_size; i++, pimage++)
    {
        backpropagation(pimage, &dCdW[0], &dCdB[0], pwl, pbl, pzl, pal);
    }
    
    // this block of code calculates the average dCdB and dCdW
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            // calculate the average dCdB per neuron
            float valb = give_matrix(dCdB[i], 1, j, 0);
#ifdef DEBUG_UM
            printf("dC/dB(accum)[%d][%d] = %f\n", i, j, valb);
#endif
            valb = valb / batch_size;
#ifdef DEBUG_UM
            printf("dC/dB|(mean)[%d][%d] = %f\n", i, j, valb);
#endif
            set_matrix(dCdB[i], 1, j, 0, valb);
            
            // calculate the average dCdW for each weight of the neuron
            for (int k = 0; k < neurons[i-1]; k++)
            {
                float valw = give_matrix(dCdW[i], neurons[i-1], j, k);
#ifdef DEBUG_UM
                printf("dC/dW(accum)[%d][%d][%d] = %f\n", i, j, k, valw);
#endif
                valw = valw / batch_size;
#ifdef DEBUG_UM
                printf("dC/dW(mean)[%d][%d][%d] = %f\n", i, j, k, valw);
#endif
                set_matrix(dCdW[i], neurons[i-1], j, k, valw);
            }
        }
    }
    
#ifdef Evaluate_UM
    printf("\nEvaluating dC/dB...\n");
    for (int i = 0; i < neurons[LAYERS-1]; i++)
    {
        printf("dC/dB[%d][%d] = %f\n", LAYERS-1, i, *(dCdB[LAYERS-1]+i));
    }
#endif
    
    // This block of code adjusts the weights and biases according to the calculated dC/dW and dC/dB
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            set_matrix(*(pbl+i), 1, j, 0, give_matrix(*(pbl+i), 1, j, 0) + (*(dCdB[i]+j) * (-eta))); // takes the current value of the Bias and accumulates it with dC/dB*(-eta). I.e. Bias = Bias + (dC/dB*(-eta))
            for (int k = 0; k < neurons[i-1]; k++)
            {
                set_matrix(*(pwl+i), neurons[i-1], j, k, give_matrix(*(pwl+i), neurons[i-1], j, k) + give_matrix(dCdW[i], neurons[i-1], j, k) * (-eta)); // takes the current value of the weight and accumulates it with dC/dW*(-eta). I.e. Weight = Weight + (dC/dW*(-eta))
            }
        }
    }
}


void backpropagation2(struct image *pimage, float **pdCdW, float **pdCdB, float **pwl, float **pbl, float **pzl, float **pal)
{
    
    feedforward(pimage, pwl, pbl, pzl, pal, 0); // feedforward through the network
    
    float *pdelta[LAYERS]; // make pointers for the vectors of the errors of each layer
    float *presultSPZL[LAYERS]; // make pointers for the vectors of the result of sigmoid_prima(zl)
    float *wT[LAYERS]; // buffer for the transpose of the weight layer
    
    for (int i = 1; i < LAYERS ; i++)
    {
        wT[i] = malloc(neurons[i]*neurons[i-1]*sizeof(float));
    }
    
    for (int i = 0; i < LAYERS; i++)
    {
        pdelta[i] = malloc(sizeof(float)*neurons[i]); // make memory for the vectors of the errors of each layer and assign the adress to pError
        presultSPZL[i] = malloc(sizeof(float)*neurons[i]); // make memory for the vectors of the result of sigmoid_prima(zl)
    }
    
    // Make the Y vector
    float *pY = make_matrix(neurons[LAYERS-1], 1); // make the Y vector
    memclear(pY, (neurons[LAYERS-1]*sizeof(float))); // clear the Y vector, i.e. set all places to 0
    set_matrix(pY, 1, ((pimage->digit)), 0, 1); // set the appropriate neuron to 1 (dependent on the input image)
    
    // Calculate delta_L
    matrix_subt(pdelta[LAYERS-1], *(pal+(LAYERS-1)), pY, neurons[LAYERS-1], 1); // (AL-Y)
    matrix_sigmoid_prime(*(pzl+(LAYERS-1)), presultSPZL[LAYERS-1], neurons[LAYERS-1], 1); // sigmoid_prime(ZL)
    matrix_hadamard(pdelta[LAYERS-1], pdelta[LAYERS-1], presultSPZL[LAYERS-1], neurons[LAYERS-1], 1);
    
    
#ifdef DEBUG_BP2
    printf("The Y vector =\n");
    print_matrix(pY, neurons[LAYERS-1], 1);
    printf("The result of sigmoid_prima(ZL) = \n");
    print_matrix(presultSPZL[LAYERS-1], neurons[LAYERS-1], 1);
    printf("calculated deltaL = \n");
    print_matrix(pdelta[LAYERS-1], neurons[LAYERS-1], 1);
#endif
    
    // Calculate delta_l
    for (int i = LAYERS-2; i > 0; i--)
    {
        matrix_transpose((*(pwl+i+1)), wT[i+1], neurons[i+1], neurons[i]); // calculate transpose of the weight layer and put it in the buffer wT
        matrix_mult(pdelta[i], wT[i+1], pdelta[i+1], neurons[i], neurons[i+1], neurons[i+1], 1); // multiply the transpose of w(l+1) with delta(l+1)
        matrix_sigmoid_prime(*(pzl+(i)), presultSPZL[i], neurons[i], 1);
        
        matrix_hadamard(pdelta[i], pdelta[i], presultSPZL[i], neurons[i], 1); // calculate the hadamard product of deltal
        //        printf("delta_layer[%d] =\n", i);
        //        print_matrix(pdelta[i], neurons[i], 1);
    }
    
    // Calculate dCdW
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            for (int k = 0; k < neurons[i-1]; k++)
            {
                float temp = give_matrix(pdelta[i], 1, j, 0)*give_matrix(*(pal+i-1), 1, k, 0);
                set_matrix(*(pdCdW+i), neurons[i-1], j, k, temp);
            }
        }
    }
    
    // Calculate dCdB
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            set_matrix(*(pdCdB+i), 1, j, 0, give_matrix(pdelta[i], 1, j, 0));
        }
    }
    
}

/*
 Name function: backpropagation
 
 Description:
 This function feedforwards trough the neural network given by the pointers
 and according to the given image. After this it will calculate the dC/dW and the
 dc/dB of the network.
 
 Usage:
 *pimage = pointer to the structure holding the image data
 **pwl = pointer to the weight layers
 **pbl = pointer to the bias layers
 **pzl = pointer to the weighted input layers
 **pal = pointer to the activation layers
 
 Returns: nothing
 
 */
void backpropagation(struct image *pimage, float **pdCdW, float **pdCdB, float **pwl, float **pbl, float **pzl, float **pal)
{
#ifdef DEBUG_BP
    printf("input digit = %d\n", pimage->digit);
#endif
    
    feedforward(pimage, pwl, pbl, pzl, pal, 0);
    
    // Code for calculating the delta's of each layer
    
    // this block of code allocates memory for the deltal vector of every layer
    float *pdelta_l[LAYERS]; // this line allocates a floating point array for the addresses of the deltal memory of each layer.
    for (int i = 0; i < LAYERS; i++)
    {
        pdelta_l[i] = malloc(sizeof(float)*neurons[i]); // this line of code allocates the actual memory that each layer needs for the amount of neurons in that layer
    }
    
    // this block of code calculates the delta's of the last layer (deltaL)
    for (int i = 0; i < neurons[LAYERS-1]; i++)
    {
        // dirty way of generating the y vector
        int y;
        if(pimage->digit == i)
        {
            //printf("in backpropagation function, digit = %d, i = %d\n", pimage->digit, i);
            y = 1;
        }
        else
            y = 0;
        
        *(pdelta_l[LAYERS-1]+i) = (give_matrix(*(pal+(LAYERS-1)), 1, i, 0) - y) * sigmoid_prime(give_matrix(*(pzl+(LAYERS-1)), 1, i, 0)); // this line of code calculates the delta value for each neurons according to: (aL-y)*sigmoid(zL)'
        
#ifdef DEBUG_BP
        printf("calculated delta[%d][%d] = %f\n", LAYERS-1, i, *(pdelta_l[LAYERS-1]+i)); // debug output
#endif
    }
    
    // This block of code will calculate the delta's of the other layers
    for (int i = LAYERS-2; i > 0; i--) // This for loop starts at the last layer -1. I.e. the second-last layer and will then go backwards and select each next layer until the second layer.
    {
        for (int j = 0; j < neurons[i]; j++) // This for loop will select every neuron in each layer.
        {
            for (int k = 0; k < neurons[i+1]; k++) // This for loop will calculate the delta of each neuron
            {
                *(pdelta_l[i]+j) += *(pdelta_l[i+1]+k) * give_matrix(*(pwl+i+1), neurons[i], k, j) * sigmoid_prime(give_matrix(*(pzl+i), 1, j, 0));
            }
#ifdef DEBUG_BP
            printf("calculated delta[%d][%d] = %f\n", i, j, *(pdelta_l[i]+j)); // debug output
#endif
        }
    }
    
    // This block of code will assign the deltal to the pdCdB (i.e. dC/dB = dC/dZ)
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            set_matrix(*(pdCdB+i), 1, j, 0, *(pdelta_l[i]+j)+give_matrix(*(pdCdB+i), 1, j, 0)); // this line of code will accumulate the delta's over the batch size
        }
    }
    
    // This block of code will calculate the dCdW (delta_l*a_l-1) and assigns it to pdCdW
    for (int i = 1; i < LAYERS; i++)
    {
        for (int j = 0; j < neurons[i]; j++)
        {
            for (int k = 0; k < neurons[i-1]; k++)
            {
                float val = *(pdelta_l[i]+j) * give_matrix(*(pal+(i-1)), 1, k, 0); // calculates dC/dWljk = delta_lj * a_l-1k
#ifdef DEBUG_BP
                printf("Calculated dCdW[%d][%d][%d] = %f\n", i, j, k, val);
#endif
                set_matrix(*(pdCdW+i), neurons[i-1], j, k, val + give_matrix(*(pdCdW+i), neurons[i-1], j, k)); // this line of code accumulates the calculated dC/dWljk with the dC/dWljk of the other calculations from the batch
            }
        }
    }
}



/*
 Name function: data_shuffle
 
 Description:
 This function shuffles the given dataset.
 
 Usage:
 struct image *pdata = pointer to the beginning of the image dataset
 int data_size = size of the dataset
 */
void data_shuffle(struct image *pdata, int data_size)
{
    int shuffle[data_size]; // integer array used for shuffling the training data
    init_shuffle(&shuffle[0], data_size); // initialize the shuffle array with values 0 to data_size
    shuffle_array(&shuffle[0], data_size); // shuffle the array
    
    struct image training_data_buf[data_size]; // create a buffer for the training data (this is needed to shuffle the training data)
    
    for (int i = 0; i < data_size; i++)
    {
        training_data_buf[i] = *(pdata+i); // make a copy of the whole training data set
    }
    for (int i = 0; i < data_size; i++)
    {
        *(pdata+i) = training_data_buf[shuffle[i]]; // shuffle the data set with the shuffled array as index
    }
    
}

/*
 This function initializes the shuffle array with numbers. This is needed because it needs initial values in order for them to be shuffled.
 */
void init_shuffle(int *shuffle, int data_size)
{
    for (int i = 0; i < data_size; i++, shuffle++)
    {
        *shuffle = i;
    }
}

/*
 Arrange the N elements of ARRAY in random order.
 Only effective if N is much smaller than RAND_MAX;
 if this may not be the case, use a better random
 number generator.
 */
void shuffle_array(int *array, int n)
{
    if (n > 1)
    {
        int i;
        for (i = 0; i < n - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
