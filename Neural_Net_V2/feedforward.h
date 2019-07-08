//
//  feedforward.h
//  Neural_Net_V2
//
//  Created by Mauricio Paulusma on 07/07/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#ifndef feedforward_h
#define feedforward_h

void feedforward(struct image *pimage, float **pwl, float **pbl, float **pzl, float **pal, int* neurons, int FF_Evaluate);
void feedforward2(struct image *pimage, struct neural_net *ANN, int FF_Evaluate);
#endif /* feedforward_h */
