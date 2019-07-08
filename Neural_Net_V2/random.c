//
//  random.c
//  Neural Net
//
//  Created by Mauricio Paulusma on 13/01/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#include "random.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/*
 Random generator with a gaussian distribution. Mean = 0, Variance = 1
 */
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
    
    if(phase == 0)
    {
        do
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    
    phase = 1 - phase;
    
    return X;
}


/*
 This function initializes a float array with random gaussian distributed numbers
 */
void init_rand(float *p, int size)
{
    for (int i = 0; i < size; i++, p++)
    {
        *p = (float)gaussrand();
        
#ifdef DEBUG_RAND_INIT
        printf("%d:*p = %f\n", i, *p);
#endif
    }
}
