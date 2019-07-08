//
//  sigmoid.c
//  Neural Net
//
//  Created by Mauricio Paulusma on 13/01/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#include <math.h>
#include <stdio.h>


/*
 Name function: sigmoid
 
 Description:
 This function applies the sigmoid function to the given float value
 
 Usage:
 float z = the floating point value as input for the sigmoid function
 
 Returns:
 The output of the sigmoid function in floating point format
 
 */
float sigmoid(float z)
{
    float c = (1.0/(1.0+exp(-z)));
#ifdef DEBUG_sigmoid
    printf("[%d]given value = %f\n", teller++, z);
    printf("calcultated value = %f\n", c);
#endif
    return c;
}

/*
 Name function: sigmoid_prime
 
 Description:
 This function computes the derivative of the sigmoid function at a given value for z
 
 Usage:
 float z = the input for the derivative of the sigmoid function
 */
float sigmoid_prime(float z)
{
    return sigmoid(z)*(1-sigmoid(z));
}
