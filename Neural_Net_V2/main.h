/*
 Debug defines
*/
//#define FF_CHECK
//#define DEBUG_FF
/*
 Other defines
*/
#define TRAINING_SIZE 50000
#define ROWS 28
#define COLL 28
#define LAYERS 3

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

