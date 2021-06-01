/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2017/2018
 *
 * Version: 2.0
 *
 * OpenMP code.
 *
 * (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<omp.h>

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}


#define THRESHOLD    0.001f

/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *posval; // Positions and values
} Storm;

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
//TODO: temos de alterar esta função, o vetor layer[k] é acedido por varios porocessos ao mesmo tempo
void update( float *layer, int layer_size, int k, int pos, float energy ) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
    
}


/* ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched */
/* DEBUG function: Prints the layer status */
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms ) {
    int i,k;
    /* Only print for array size up to 35 (change it for bigger sizes if needed) */
    if ( layer_size <= 35 ) {
        /* Traverse layer */
        for( k=0; k<layer_size; k++ ) {
            /* Print the energy value of the current cell */
            printf("%10.4f |", layer[k] );

            /* Compute the number of characters. 
               This number is normalized, the maximum level is depicted with 60 characters */
            int ticks = (int)( 60 * layer[k] / maximum[num_storms-1] );

            /* Print all characters except the last one */
            for (i=0; i<ticks-1; i++ ) printf("o");

            /* If the cell is a local maximum print a special trailing character */
            if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
                printf("x");
            else
                printf("o");

            /* If the cell is the maximum of any storm, print the storm mark */
            for (i=0; i<num_storms; i++) 
                if ( positions[i] == k ) printf(" M%d", i );

            /* Line feed */
            printf("\n");
        }
    }
}

/*
 * Function: Read data of particle storms from a file
 */
Storm read_storm_file( char *fname ) {
    FILE *fstorm = fopen( fname, "r" );
    if ( fstorm == NULL ) {
        fprintf(stderr,"Error: Opening storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    Storm storm;    
    int ok = fscanf(fstorm, "%d", &(storm.size) );
    if ( ok != 1 ) {
        fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
    if ( storm.posval == NULL ) {
        fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
        exit( EXIT_FAILURE );
    }
    
    
    
    //TODO: Talvez isto pode ser paralelizado:> ver melhor por causa do scanf, pois será preciso fazer seek() ou assim
    // passei a declaracao do elem para dentro do for e o ok privatizado localmente
    /*
     * Confirmar isto, nao sei bem como o programa ira comportar se com o fscanf e se vale a pena paralelizar
     * Mas nao tem dependencias no vetor, cada particao ira aceder a sua parte do vetor
     * Sera que vale a pena parallelizar, nao vao estar numa corrida pelo ficheiro?
     */
    //#pragma omp parallel for
    for ( int elem=0; elem<storm.size; elem++ ) {
        int ok = fscanf(fstorm, "%d %d\n",
                    &(storm.posval[elem*2]),
                    &(storm.posval[elem*2+1]) );
        if ( ok != 2 ) {
            fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
            exit( EXIT_FAILURE );
        }
    }
    fclose( fstorm );

    return storm;
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i,j,k;
    
    printf("Inicio main");

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    Storm storms[ num_storms ];

    /* 1.2. Read storms information */
    for( i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    for (i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
    }

    /* 2. Begin time measurement */
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */

    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_size );
    float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    //TODO: Estes 2 for podem ser paralelizados provavelmente, ver mlehor se tem dependencias, inicializacao do vetor
    /*
     * confirmar isto e ver se divide o trabalho como e suposto ou nao
     * sera que vale a pena fazer a paralelização so para a inicializacao dos vetores
     *
     * retirei os dois for e juntei as instrucoes num so
     */

    //TODO: verificar os valores de layer_size
    //#pragma omp parallel for
    for( k=0; k<layer_size; k++ ) {
        layer[k] = 0.0f;
        layer_copy[k] = 0.0f;
    }
    
    
    printf("Antes dos 3 nested fors");
    
    //TODO: Esta parte faz o trabalho todo da simulacao, retirar dependencias e paralelizar o codigo
    /* 4. Storms simulation */
    /*
     *
     * Verificar como o openMP faz handle de nested loops, devo usar um #omp parallel for dentro do loop
     * ou devo usar o #pragma omp parallel for collapse(2)
     * verificar como e onde e melhor usar parallel for
     *
     */
    //provavelmente melhora o tempo
    //o resultado pode estar errado com este pragma falta testes, o vetor layer e partilhado entre eles
    //talvez a solucao seja calcular o resultado separadamente e depois juntar as varias storms
    
    //!!!! Nao sei se isto é parallelizado, talvez é mas é preciso manter a ordem
    
    
    //#pragma opm parallel for
    for( i=0; i<num_storms; i++ ) {

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        //este pragma faz com que o resultado fique errado, provavelmente tem dependencias
        //o resultado fica errado porque esta a ser partilhado o vetor soma das energias para cada particao da superficie
        //e necessario proteger a variavel ou computar de outa maneira o resultado
        //TODO: grande parte do trabalho esta aqui, vale a pena paralelizar
        printf("Aqui\n");
        int max_thread_num = omp_get_max_threads();
        printf("Max thread num: %d thread(s)\n", max_thread_num);
        int individualLayer_size = layer_size * max_thread_num;
        printf("Layer size: %d\nIndividualLayerSize:%d\n", layer_size, individualLayer_size);
        float *individualLayer = (float *)malloc( sizeof(float) * individualLayer_size ) ;
        
        //init array
        for( k=0; k<individualLayer_size; k++ ) {
        	individualLayer[k] = 0.0f;
    	}
        
        omp_set_num_threads(1);
        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            printf("ThreadNum: %d\n", thread_num);
            
            float *privateLayer;
            int storm_size;
            int p_layer_size;
            
            #pragma omp critical // TODO: talvez tirar isto tava so a testar
            {
		    privateLayer = individualLayer + layer_size * thread_num;
		    //printf("Private layer: %p\n", privateLayer);
		    //printf("%p\n", individualLayer);
		    storm_size = storms[i].size; // nao sei se vale a pena mas e para nao aceder todas as vezes que faz o loop
		    p_layer_size = layer_size;
            }

            #pragma omp for
            for (j = 0; j < storm_size; j++) {
                /* Get impact energy (expressed in thousandths) */
                float energy = (float) storms[i].posval[j * 2 + 1] * 1000;
                /* Get impact position */
                int position = storms[i].posval[j * 2];

                /* For each cell in the layer */
//TODO: Verificar quantos pontos e que este loop percorre normalmente e ver se e sufuciente para parallelizar
                //os tempos com este pragma ficam piores, provavelmente tem dependencias
                //os valores de k só variam de 1 ate layer_size, sendo o valor de layer_size pequeno siginifica
                //quantas particoes da superficie temos. A menos que a superficie seja grande, nao vale a pena
                //#pragma omp parallel 
                
                // !!!!! preciso por esta variavel local, para nao se parallelizada!!!!!
                for (int k = 0; k < layer_size; k++) {
                    /* Update the energy value for the cell */
                    update(privateLayer, layer_size, k, position, energy);
                }
	     }
	}

        //juntar resultados
        //#pragma omp parallel for
        //ver se istp vale a pena
        // se usar menos threads que o maximo ira estar a somar zeros, 
        //preciso arranjar outra forma de ver quantas threads foram usadas
        for (k = 0; k < layer_size; k++) {
            for (int l = 0; l < max_thread_num; l++) { 
                layer[k] += individualLayer[k + layer_size * l];
            }
        }

        
        free(individualLayer);

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        //#pragma omp parallel for
        //TODO: verificar os valores de layer_size
        for( k=0; k<layer_size; k++ ) 
            layer_copy[k] = layer[k];

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
//TODO: Nesta parte, esta a ser lido de um vetor e escrito noutro por isso nao ha dependencias entre as iteracoes
        //#pragma omp parallel for
        //TODO: verificar os valores de layer_size
        for( k=1; k<layer_size-1; k++ )
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

        /* 4.3. Locate the maximum value in the layer, and its position */
//TODO: verificar se nao ha dependencias entre as iteracoes
//  mas acho que nao tem, ver isto melhor
        //#pragma omp parallel for
        //em principio os resultados ficam ok com este pragma mas temos de verificar os valore de layer_size para ver se vale a pena
        //TODO: verificar os valores de layer_siz
        for( k=1; k<layer_size-1; k++ ) {
            /* Check it only if it is a local maximum */
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                if ( layer[k] > maximum[i] ) {
                    maximum[i] = layer[k];
                    positions[i] = k;
                }
            }
        }

    }

    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    ttotal = cp_Wtime() - ttotal;

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG
    debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
    /* 7.2. Print the maximum levels */
    printf("Result:");
    for (i=0; i<num_storms; i++)
        printf(" %d %f", positions[i], maximum[i] );
    printf("\n");

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );

    /* 9. Program ended successfully */
    return 0;
}

