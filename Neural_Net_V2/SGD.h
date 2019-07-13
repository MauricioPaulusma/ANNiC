//
//  SGD.h
//  Neural_Net_V2
//
//  Created by Mauricio Paulusma on 08/07/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#ifndef SGD_h
#define SGD_h

void SGD(struct image* pimage, struct neural_net* ANN, int epoch, int batch_size, float eta);
int update_minibatch2(struct image *pimage, struct neural_net* ANN, int batch_size, float eta);
int backpropagation2(struct image* pimage, float **pdCdW, float **pdCdB, struct neural_net* ANN, int debug);
void data_shuffle(struct image *pdata, int data_size);
void init_shuffle(int *shuffle, int data_size);
void shuffle_array(int *array, int n);


#endif /* SGD_h */
