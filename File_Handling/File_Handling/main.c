//
//  main.c
//  File_Handling
//
//  Created by Mauricio Paulusma on 17/07/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#include <stdio.h>

int main(int argc, const char * argv[])
{
    float test[] = {0.1, 0.2, 0.3, 0.4, 0.5};

    FILE *fp;
    fp = fopen("Weights_Biases_test", "w");
    
    fwrite(&test[0], sizeof(float), sizeof(test)/sizeof(float), fp);
    
    fclose(fp);
    
    fp = fopen("Weights_Biases_test", "r");

    float read [5];
    fread(&read[0], sizeof(float), 5, fp);
    
    printf("\nread = {");
    for (int i = 0; i < 4; i++)
    {
        printf("%f, ", read[i]);
    }
    printf(" %f}\n", read[4]);

}
