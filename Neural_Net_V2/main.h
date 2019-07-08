#define TRAINING_SIZE 50000
#define ROWS 28
#define COLL 28
#define LAYERS 4

/*
 Structure for holding the training data
 */
struct image
{
    unsigned char pixels[ROWS][COLL]; // this multidimensional array stores the pixel values
    unsigned char digit; // this char holds the value of the digit
};

