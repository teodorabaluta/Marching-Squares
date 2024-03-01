// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }
// definirea structurii pentru resursele care vor fi folosite in thread-uri
typedef struct {
    ppm_image *image; // imaginea originala
    ppm_image *new_image; // imaginea redimensionata
    ppm_image *img; // imaginea cu care vor lucra thread-urile
    unsigned char **grid;
    int p;
    int q;
    int step_x;
    int step_y;
    int sigma;
    int n; // numarul de thread-uri
    ppm_image **contour_map;
    pthread_barrier_t barrier; // bariera pentru sincronizare
} thread_resource;

typedef struct {
    thread_resource *resource; 
    int id; // id-ul thread-ului
} thread_arg;


// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "../checker/contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
unsigned char **sample_grid(ppm_image *image, int step_x, int step_y, unsigned char sigma) {
    int p = image->x / step_x;
    int q = image->y / step_y;

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = 0; i < p; i++) {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }
    for (int j = 0; j < q; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }

    return grid;
}

// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(ppm_image *image, unsigned char **grid, ppm_image **contour_map, int step_x, int step_y) {
    int p = image->x / step_x;
    int q = image->y / step_y;

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_map[k], i * step_x, j * step_y);
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

ppm_image *rescale_image(ppm_image *image) {
    uint8_t sample[3];

    // we only rescale downwards
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        return image;
    }

    // alloc memory for image
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    new_image->x = RESCALE_X;
    new_image->y = RESCALE_Y;

    new_image->data = (ppm_pixel*)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    // use bicubic interpolation for scaling
    for (int i = 0; i < new_image->x; i++) {
        for (int j = 0; j < new_image->y; j++) {
            float u = (float)i / (float)(new_image->x - 1);
            float v = (float)j / (float)(new_image->y - 1);
            sample_bicubic(image, u, v, sample);

            new_image->data[i * new_image->y + j].red = sample[0];
            new_image->data[i * new_image->y + j].green = sample[1];
            new_image->data[i * new_image->y + j].blue = sample[2];
        }
    }

    free(image->data);
    free(image);

    return new_image;
}


int min(int x, int y) {
    if (x < y) {
        return x;
    } else {
        return y;
    }
}
void *parallel_function(void *arg) {
    
    // extragem argumentele pentru thread
    thread_arg *arguments = (thread_arg *) arg;
    thread_resource *info = arguments->resource;
    int id = arguments->id;
    int N = info->n;

    // extragem imaginile cu care lucram si de intrare
    ppm_image *img = info->img;
    ppm_image *image = info->image;
    ppm_image *new_image = info->new_image;
    

    /////////////// FUNCTIA RESCALE_IMAGE ///////////////

    // verificam daca este necesara redimensionarea
    if(img == NULL) {

        uint8_t sample[3];

        // use bicubic interpolation for scaling
        int start = id * (double) new_image->x / N;
        int end = min((id + 1) * (double)new_image->x / N, new_image->x);

        for (int i = start; i < end; i++) {
            for (int j = 0; j < new_image->y; j++) {
                float u = (float)i / (float)(new_image->x - 1);
                float v = (float)j / (float)(new_image->y - 1);
                sample_bicubic(image, u, v, sample);

                new_image->data[i * new_image->y + j].red = sample[0];
                new_image->data[i * new_image->y + j].green = sample[1];
                new_image->data[i * new_image->y + j].blue = sample[2];
            }
        }

        // asteapta ca toate thread-urile sa termine
        // redimensionarea inainte de a continua
        pthread_barrier_wait(&info->barrier);

        // actualizeaza img ul cu noua imagine redimensionata
        if(!id) {
            info->img = new_image;
            free(info->image->data);
            free(info->image);
        }
    }

    // asteapta ca toate thread-urile sa ajunga
    // la bariera inainte de a continua
    pthread_barrier_wait(&info->barrier);
    
    /////////////// FUNCTIA SAMPLE_GRID ///////////////

    int p = info->p;
    int q = info->q;
    int step_x = info->step_x;
    int step_y = info->step_y;
    int sigma = info->sigma;
    unsigned char **grid = info->grid;

    int start_1 = id * (double) p / N;
    int end_1 = min((id + 1) * (double)p / N, p);

    // esantionarea grid-ului prin paralelizare
    for (int i = start_1; i < end_1; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = info->img->data[i * step_x * info->img->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    for (int i = start_1; i < end_1; i++) {
        ppm_pixel curr_pixel = info->img->data[i * step_x * info->img->y + info->img->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }

    int start_2 = id * (double) q / N;
    int end_2 = min((id + 1) * (double)q / N, q);
    for (int j = start_2; j < end_2; j++) {
        ppm_pixel curr_pixel = info->img->data[(info->img->x - 1) * info->img->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }
    
    // asteapta ca toate thread-urile sa ajunga la aceasta 
    // bariera inainte de a continua
    pthread_barrier_wait(&info->barrier);

    // actualizeaza grid-ul cu ce obtinut dupa esantionare
    if(!id) {
        info->grid = grid;
    }
    pthread_barrier_wait(&info->barrier);
    grid = info->grid;

    /////////////// FUNCTIA MARCH ///////////////

    int start_3 = id * (double) p / N;
    int end_3 = min((id + 1) * (double)p / N, p);
    for (int i = start_3; i < end_3; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(info->img, info->contour_map[k], i * step_x, j * step_y);
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    // setarea numarului de thread-uri dat in linia de comanda
    int N = atoi(argv[3]);

    // declararea variabilelor si a resurselor necesare
    pthread_t threads[N];
    thread_resource resource;
    thread_arg arguments[N];
    pthread_barrier_init(&resource.barrier, NULL, N);


    // citirea imaginii de intrare din primul
    // argument dat in linia de comanda
    ppm_image *image = read_ppm(argv[1]);
    int step_x = STEP;
    int step_y = STEP;
    void *status;


    // 1. Rescale the image 
    // ppm_image *scaled_image = rescale_image(image);


    // initializarea resurselor thread-ului 
    resource.image = image;
    resource.new_image = NULL;
    resource.img = NULL;
    resource.n = N;
    resource.step_x = step_x;
    resource.step_y = step_y;
    resource.sigma = SIGMA;
    resource.contour_map = init_contour_map();

    int p,q;

    // verificare daca dimensiunile imaginii initiale sunt potrivite
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        resource.img = image;
        p = image->x / resource.step_x;
        q = image->y / resource.step_y;
    } else {
        resource.new_image = malloc(sizeof(ppm_image));
        resource.new_image->x = RESCALE_X;
        resource.new_image->y = RESCALE_Y;
        p = resource.new_image->x / resource.step_x;
        q = resource.new_image->y / resource.step_y;
        resource.new_image->data = malloc(RESCALE_X * RESCALE_Y * sizeof(ppm_pixel));
        resource.img = NULL;
    }
    
    // setarea dimensiunilor gridului in resursele thread-ului
    resource.p = p;
    resource.q = q;

    // alocare memorie grid
    resource.grid = malloc((p + 1) * sizeof(unsigned char*));
    for (int i = 0; i <= p; i++)
        resource.grid[i] = malloc((q + 1) * sizeof(unsigned char));
    
    //creearea si executarea thread-ului
    for (int i = 0; i < N; i++) {
        arguments[i].id = i;
        arguments[i].resource = &resource;
        int r = pthread_create(&threads[i], NULL, parallel_function,  &arguments[i]);
 
        if (r) {
            printf("Eroare la crearea thread-ului %d\n", i);
            exit(-1);
        }
    }
 
    for (int i = 0; i < N; i++) {
        int r = pthread_join(threads[i], &status);

        if (r) {
            printf("Eroare la asteptarea thread-ului %d\n", i);
            exit(-1);
        }
    }
    

    printf("resource : %p arg 0 resource %p arg 1 resource %p\n", &resource.img, arguments[0].resource->img, arguments[0].resource->img);

    // 2. Sample the grid
    // unsigned char **grid = sample_grid(arguments[0].resource->img, step_x, step_y, SIGMA);

    // 3. March the squares
    // march(arguments[0].resource->img, arguments[0].resource->grid, contour_map, step_x, step_y);

    // 4. Write output
    write_ppm(resource.img, argv[2]);

    free_resources(resource.img, resource.contour_map, resource.grid, step_x);

    // distrugerea barierei
    pthread_barrier_destroy(&resource.barrier);

    return 0;
}
