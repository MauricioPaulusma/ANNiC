/*
 Debug defines
*/
//#define FF_CHECK
//#define DEBUG_FF
#define Evaluate_SGD
//#define DEBUG_UM
/*
 Other defines
*/
#define TRAINING_SIZE 50000 // size of the training data set
#define TEST_SIZE 10000 // size of the test data set
#define BATCH_SIZE 200 // mini-batch size
#define ETA 0.1 // learning rate
#define EPOCH 2
#define LAYERS 4

#define ROWS 28
#define COLL 28



/*
 Structure for holding the training data
 */
struct image
{
    unsigned char pixels[ROWS][COLL]; // this multidimensional array stores the pixel values
    unsigned char digit; // this char holds the value of the digit
};

/*
 Structure for holding the neural net
 */
struct neural_net
{
    float *pwl[LAYERS];
    float *pbl[LAYERS];
    float *pzl[LAYERS];
    float *pal[LAYERS];
    int nr_of_layers;
    int* neurons;
};

