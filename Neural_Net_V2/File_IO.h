//
//  File_IO.h
//  Neural_Net_V2
//
//  Created by Mauricio Paulusma on 17/07/2019.
//  Copyright © 2019 Mauricio Paulusma. All rights reserved.
//

#ifndef File_IO_h
#define File_IO_h

#include <stdio.h>

int save_net(struct neural_net* ANN, char* file_name);
int load_net(struct neural_net* ANN, char* file_name);
void print_net(struct neural_net* ANN);

#endif /* File_IO_h */
