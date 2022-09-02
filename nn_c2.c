/*
*   nn_c2.c - Colin Luoma
*
*   A simple Neural Network designed to learn on MNIST data.
*   Code was implemented using formulas found at:
*   (http://neuralnetworksanddeeplearning.com/chap2.html)
*
*   LEARN_RATE, NUM_LAYERS, and the inner sizes can be changed.
*
*   Read data in CSV form. Find data: (https://pjreddie.com/projects/mnist-in-csv/)
*
*   No License. Free to use for any purposes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>

#define SAMPLE_SIZE 60000

// NN vars
#define LEARN_RATE 0.1
#define LMBDA 5.0
#define MINIBATCH_SIZE 10  // Must divide into SAMPLE_SIZE evenly
#define NUM_LAYERS 4
int sizes[] = { 784, 100, 30, 10 };

// Training data
int train_labels[SAMPLE_SIZE];
double train_values[SAMPLE_SIZE][784];
int test_labels[10000];
double test_values[10000][784];

// Weights and biases
gsl_vector *biases[NUM_LAYERS - 1];
gsl_matrix *weights[NUM_LAYERS - 1];

// Intermediate feedforward results
gsl_matrix *z[NUM_LAYERS - 1];
gsl_matrix *a[NUM_LAYERS - 1];
gsl_matrix *delta[NUM_LAYERS - 1];

// Read in MNIST data from CSVs
void
load_data(const char *file, int labels[], double values[][784])
{
    FILE *mnist_train = fopen(file, "r");
    char line[10000];
    int row = 0;
    int col = 0;
    int line_c = 0;
    while( fgets( line, 9999, mnist_train ) != NULL && line_c < SAMPLE_SIZE)
    {
        char *token = strtok( line, "," );
        if( token == NULL ) continue;

        labels[row] = atoi( token );

        while( (token = strtok( NULL, "," )) != NULL )
        {
            float val = atof(token);
            if ( val > 50) {
                val = 255;
            } else {
                val = 0;
            }
            values[row][col] = val / 255.0;
            col++;
        }
        row++;
        col = 0;

        line_c++;
    }
}

double
sigmoid(double x)
{
    return ( 1.0 / ( 1.0 + pow( M_E, (-1.0) * x ) ) );
}

double
sigmoid_prime(double x)
{
    return ( sigmoid( x ) * ( 1 - sigmoid( x ) ) );
}

void
elementwise_sigmoid(gsl_matrix *m)
{
    for( int i = 0; i < m->size1; i++ )
    {
        for( int j = 0; j < m->size2; j++ )
        {
            gsl_matrix_set(m, i, j, sigmoid(gsl_matrix_get(m, i, j)));
        }
    }
}

void
elementwise_sigmoid_prime(gsl_matrix *m)
{
    for( int i = 0; i < m->size1; i++ )
    {
        for( int j = 0; j < m->size2; j++ )
        {
            gsl_matrix_set(m, i, j, sigmoid_prime(gsl_matrix_get(m, i, j)));
        }
    }
}

// Does a single feed forward and counts the correct
// predictions from test data
void
eval_network(gsl_matrix *data, int samples)
{
    // Feed forward
    gsl_matrix *curr = data;
    for( int layer = 1; layer < NUM_LAYERS; layer++ )
    {
        gsl_matrix *result = gsl_matrix_calloc( sizes[layer], samples );
        // Multiply all observations by weights
        gsl_blas_dgemm( CblasNoTrans, CblasNoTrans,
                        1.0, weights[layer - 1], curr,
                        0.0, result);

        // Add bias and apply sigmoid
        for( int i = 0; i < result->size1; i++ )
        {
            for( int j = 0; j < result->size2; j++ )
            {
                gsl_matrix_set( result, i, j,
                                sigmoid(
                                        gsl_matrix_get( result, i, j ) +
                                        gsl_vector_get( biases[layer - 1], i ) ) );
            }
        }
        // Save a
        a[layer-1] = gsl_matrix_alloc( result->size1, result->size2 );
        gsl_matrix_memcpy( a[layer-1], result );

        gsl_matrix_free(result);

        curr = a[layer-1];
    }

    // Count correct guesses
    int guesses[samples];
    size_t correct = 0;
    for( int i = 0; i < samples; i++ )
    {
        int max = 0;
        double cur_guess = gsl_matrix_get(curr, 0, i);
        for( int j = 1; j < 10; j++ )
        {
            if( gsl_matrix_get(curr, j, i) > cur_guess )
            {
                max = j;
                cur_guess = gsl_matrix_get(curr, j, i);
            }
        }
        if( max == test_labels[i] )
            correct++;
    }

    // Free intermediate calculations
    for( int layer = NUM_LAYERS-2; layer >=0; layer-- )
    {
        gsl_matrix_free(a[layer]);
    }
    printf("Correct: %d %%%f\n", (int)correct, (correct / (double)samples)*100.0);
}

// Does a feed forward, and a back propogation
// Updates weights and biases during back propogation
void
update_coefficients(gsl_matrix *data, gsl_matrix *labels, int m)
{
    // Feed forward
    gsl_matrix *curr = data;
    for( int layer = 1; layer < NUM_LAYERS; layer++ )
    {
        gsl_matrix *result = gsl_matrix_calloc( sizes[layer], m );
        // Multiply all observations by weights
        gsl_blas_dgemm( CblasNoTrans, CblasNoTrans,
                        1.0, weights[layer - 1], curr,
                        0.0, result);

        // Add bias
        for( int i = 0; i < result->size1; i++ )
        {
            for( int j = 0; j < result->size2; j++ )
            {
                gsl_matrix_set( result, i, j,
                                gsl_matrix_get( result, i, j ) +
                                gsl_vector_get( biases[layer - 1], i ) );
            }
        }
        // Save z
        z[layer-1] = gsl_matrix_alloc( result->size1, result->size2 );
        gsl_matrix_memcpy( z[layer-1], result );

        // Do sigmoid transformation
        elementwise_sigmoid( result );

        // Save a
        a[layer-1] = gsl_matrix_alloc( result->size1, result->size2 );
        gsl_matrix_memcpy( a[layer-1], result );

        gsl_matrix_free( result );

        curr = a[layer-1];
    }

    // Outermost layer error
    delta[NUM_LAYERS-2] = gsl_matrix_alloc( 10, m );
    gsl_matrix_memcpy( delta[NUM_LAYERS-2], a[NUM_LAYERS-2] );
    gsl_matrix_sub( delta[NUM_LAYERS-2], labels );
//    // Do inverse sigmoid
//    elementwise_sigmoid_prime( z[NUM_LAYERS-2] );
//    // Multiply
//    gsl_matrix_mul_elements( delta[NUM_LAYERS-2], z[NUM_LAYERS-2] );

    // Back propogation
    for( int layer = NUM_LAYERS-2; layer > 0; layer-- )
    {
        delta[layer-1] = gsl_matrix_alloc( sizes[layer], m );

        // Do inverse sigmoid
        elementwise_sigmoid_prime( z[layer-1] );

        gsl_blas_dgemm( CblasTrans, CblasNoTrans,
                        1.0, weights[layer], delta[layer],
                        0.0, delta[layer-1]);

        // Multiply
        gsl_matrix_mul_elements( delta[layer-1], z[layer-1] );
    }

    // Weights
    for( int layer = NUM_LAYERS-1; layer > 0; layer--)
    {
        gsl_matrix *temp_m = gsl_matrix_alloc( sizes[layer], sizes[layer-1] );

        if( layer > 1 )
        {
            gsl_blas_dgemm( CblasNoTrans, CblasTrans,
                            1.0, delta[layer-1], a[layer-2],
                            0.0, temp_m);
            gsl_matrix_scale( temp_m, LEARN_RATE / m );
            gsl_matrix_scale( weights[layer-1], (1 - (LEARN_RATE) * (LMBDA / SAMPLE_SIZE)) );
//            (1-eta*(lmbda/n))
            gsl_matrix_sub( weights[layer-1], temp_m );
        }
        else
        {
            gsl_blas_dgemm( CblasNoTrans, CblasTrans,
                            1.0, delta[0], data,
                            0.0, temp_m );
            gsl_matrix_scale( temp_m, LEARN_RATE / m );
            gsl_matrix_scale( weights[0], 1 - (LEARN_RATE * (LMBDA / SAMPLE_SIZE)) );
            gsl_matrix_sub( weights[0], temp_m );
        }
        gsl_matrix_free(temp_m);
    }

    // Biases
    for( int layer = NUM_LAYERS-1; layer > 0; layer--)
    {
        gsl_matrix *temp_b = gsl_matrix_alloc( m, 1 );
        for(int i = 0; i < m; i++ )
        {
            gsl_matrix_set(temp_b, i, 0, 1.0);
        }

        gsl_matrix *temp_b2 = gsl_matrix_calloc( sizes[layer], 1 );
        gsl_blas_dgemm( CblasNoTrans, CblasNoTrans,
                        1.0, delta[layer-1], temp_b,
                        0.0, temp_b2);
        gsl_matrix_scale( temp_b2, LEARN_RATE / m );
        for(int i = 0; i < biases[layer-1]->size; i++ )
        {
            gsl_vector_set( biases[layer-1], i, gsl_vector_get(biases[layer-1], i) - gsl_matrix_get(temp_b2, i, 0) );
        }
        gsl_matrix_free(temp_b);
        gsl_matrix_free(temp_b2);
    }

    // Free intermediate calculations
    for( int layer = NUM_LAYERS-2; layer >=0; layer-- )
    {
        gsl_matrix_free(a[layer]);
        gsl_matrix_free(z[layer]);
        gsl_matrix_free(delta[layer]);
    }
}

int
main()
{

    // Load data from CSV
    load_data("mnist_train.csv", train_labels, train_values);
    load_data("mnist_test.csv", test_labels, test_values);

    // Create normal random generator
    gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);

    // Move training data into GSL matrices
    gsl_matrix *train_data_m = gsl_matrix_alloc( 784, SAMPLE_SIZE );
    gsl_matrix *train_labels_m = gsl_matrix_calloc( 10, SAMPLE_SIZE );
    for( int i = 0; i < SAMPLE_SIZE; i++ )
    {
        gsl_matrix_set( train_labels_m, (int)train_labels[i], i, 1.0 );
        for( int j = 0; j < 784; j++ )
        {
            gsl_matrix_set( train_data_m, j, i, train_values[i][j] );
        }
    }
    // Move test data into GSL matrices
    gsl_matrix *test_data_m = gsl_matrix_alloc( 784, 10000 );
    gsl_matrix *test_labels_m = gsl_matrix_calloc( 10, 10000 );
    for( int i = 0; i < 10000; i++ )
    {
        gsl_matrix_set( test_labels_m, (int)test_labels[i], i, 1.0 );

        for( int j = 0; j < 784; j++ )
        {
            gsl_matrix_set( test_data_m, j, i, test_values[i][j] );
        }
    }

    // Initialize biases
    for( int i = 0; i < NUM_LAYERS - 1; i++ )
    {
        biases[i] = gsl_vector_alloc( sizes[i+1] );
        for( int j = 0; j < sizes[i+1]; j++ )
        {
            gsl_vector_set( biases[i], j, gsl_ran_gaussian( r, 1 ) );
        }
    }
    // Initialize weights
    for( int k = 0; k < NUM_LAYERS - 1; k++ )
    {
        weights[k] = gsl_matrix_alloc(sizes[k+1], sizes[k]);
        for( int i = 0; i < sizes[k+1]; i++ )
        {
            for( int j = 0; j < sizes[k]; j++ )
            {
                gsl_matrix_set( weights[k], i, j, gsl_ran_gaussian( r, 1 ) );
            }
        }
    }

    // Main epoch loop
    for( int l = 0; l < 300; l++ )
    {
        // Minibatches, create an array or minibatch groups and shuffle
        int batches = SAMPLE_SIZE / MINIBATCH_SIZE;
        int order[SAMPLE_SIZE];
        for( int i = 0; i < SAMPLE_SIZE; i += batches )
        {
            for( int j = 0; j < batches; j++ )
            {
                order[i+j] = j;
            }
        }
        gsl_ran_shuffle( r, order, SAMPLE_SIZE, sizeof(int) );

        // Do a feedforward and a backprop for each minibatch
        for( int i = 0; i < batches; i++ )
        {
            gsl_matrix *train_batch_v = gsl_matrix_alloc( 784, MINIBATCH_SIZE );
            gsl_matrix *train_batch_l = gsl_matrix_alloc( 10, MINIBATCH_SIZE );

            int count = 0;
            for( int j = 0; j < SAMPLE_SIZE; j++ )
            {
                if( order[j] == i )
                {
                    for( int k = 0; k < 784; k++ )
                    {
                        gsl_matrix_set( train_batch_v, k, count, gsl_matrix_get( train_data_m, k, j ) );
                    }
                    for( int k = 0; k < MINIBATCH_SIZE; k++ )
                    {
                        gsl_matrix_set( train_batch_l, k, count, gsl_matrix_get( train_labels_m, k, j ) );
                    }
                    count++;
                }
            }
            update_coefficients(train_batch_v, train_batch_l, MINIBATCH_SIZE);

            gsl_matrix_free( train_batch_v );
            gsl_matrix_free( train_batch_l );
        }

        // Show accuracy at the end of the epoch, using test data
        printf("%d: ", l);
        eval_network( test_data_m, (int)test_data_m->size2 );
    }

    // save model to disk
    // use to_json.R to turn weight and bias files into json
    char biasfilename[100];
    char weightfilename[100];
    for (int layer = 0; layer < NUM_LAYERS-1; layer++)
    {
        sprintf(biasfilename, "biases%d.csv", layer);
        sprintf(weightfilename, "weights%d.csv", layer);
        FILE *bias_file = fopen(biasfilename, "w+");
        FILE *weight_file = fopen(weightfilename, "w+");

        gsl_vector_fprintf(bias_file, biases[layer], "%f");
        gsl_matrix_fprintf(weight_file, weights[layer], "%f");

        fclose(bias_file);
        fclose(weight_file);
    }

    return 0;
}