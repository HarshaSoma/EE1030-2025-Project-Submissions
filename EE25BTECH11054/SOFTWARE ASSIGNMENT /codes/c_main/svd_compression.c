#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    double **d;
    int r, c;
} Mat;

Mat make_mat(int r, int c) {
    Mat m;
    m.r = r; m.c = c;
    m.d = malloc(r * sizeof(double *));
    for (int i = 0; i < r; i++) {
        m.d[i] = calloc(c, sizeof(double));
    }
    return m;
}

void free_mat(Mat *m) {
    for (int i = 0; i < m->r; i++) free(m->d[i]);
    free(m->d);
}

Mat transpose_mat(Mat m) {
    Mat t = make_mat(m.c, m.r);
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            t.d[j][i] = m.d[i][j];
    return t;
}

Mat mul_mat(Mat a, Mat b) {
    if (a.c != b.r) {
        printf("Matrix dimension error\n");
        exit(1);
    }
    
    Mat c = make_mat(a.r, b.c);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < b.c; j++)
            for (int k = 0; k < a.c; k++)
                c.d[i][j] += a.d[i][k] * b.d[k][j];
    return c;
}

double vec_len(double *v, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrt(sum);
}

void norm_vec(double *v, int n) {
    double len = vec_len(v, n);
    if (len > 1e-10)
        for (int i = 0; i < n; i++) v[i] /= len;
}

void find_svd(Mat A, Mat *U, double **S, Mat *V, int k) {
    int m = A.r, n = A.c;
    Mat Ac = make_mat(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            Ac.d[i][j] = A.d[i][j];

    *U = make_mat(m, k);
    *V = make_mat(n, k);
    *S = calloc(k, sizeof(double));

    Mat AT = transpose_mat(Ac);
    Mat ATA = mul_mat(AT, Ac);

    for (int iter = 0; iter < k; iter++) {
        double *v = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) v[i] = (double)rand() / RAND_MAX;

        for (int p = 0; p < 100; p++) {
            double *v_new = calloc(n, sizeof(double));
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    v_new[i] += ATA.d[i][j] * v[j];
            norm_vec(v_new, n);
            for (int i = 0; i < n; i++) v[i] = v_new[i];
            free(v_new);
        }

        for (int i = 0; i < n; i++) V->d[i][iter] = v[i];

        double *u = calloc(m, sizeof(double));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                u[i] += Ac.d[i][j] * v[j];

        double s = vec_len(u, m);
        (*S)[iter] = s;

        if (s > 1e-10)
            for (int i = 0; i < m; i++) u[i] /= s;

        for (int i = 0; i < m; i++) U->d[i][iter] = u[i];

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                Ac.d[i][j] -= s * u[i] * v[j];

        free_mat(&AT);
        free_mat(&ATA);
        AT = transpose_mat(Ac);
        ATA = mul_mat(AT, Ac);
        free(v);
        free(u);

        printf("Computed singular value %d: %.2f\n", iter + 1, s);
    }

    free_mat(&Ac);
    free_mat(&AT);
    free_mat(&ATA);
}

Mat rebuild_img(Mat U, double *S, Mat V, int k) {
    Mat out = make_mat(U.r, V.r);
    for (int i = 0; i < U.r; i++)
        for (int j = 0; j < V.r; j++)
            for (int r = 0; r < k; r++)
                out.d[i][j] += U.d[i][r] * S[r] * V.d[j][r];
    return out;
}

void save_img(Mat img, const char *name, int w, int h) {
    unsigned char *px = malloc(w * h);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double val = img.d[i][j];
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            px[i * w + j] = (unsigned char)val;
        }
    }
    stbi_write_png(name, w, h, 1, px, w);
    printf("Saved: %s\n", name);
    free(px);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_filename>\n", argv[0]);
        return 1;
    }

    int w, h, ch;
    unsigned char *img = stbi_load(argv[1], &w, &h, &ch, 1);
    if (!img) {
        printf("Error loading image\n");
        return 1;
    }

    printf("Image loaded: %dx%d\n", w, h);

    Mat A = make_mat(h, w);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            A.d[i][j] = (double)img[i * w + j];

    stbi_image_free(img);

    int max_k = 100;
    if (max_k > w) max_k = w;
    if (max_k > h) max_k = h;

    printf("\nComputing SVD with k=%d...\n", max_k);
    Mat U, V;
    double *S;
    find_svd(A, &U, &S, &V, max_k);

    int k_vals[] = {5, 20, 50, 100};
    int num_k = 4;

    printf("\nReconstructing images:\n");
    printf("k\tError\n");
    printf("----------------\n");

    for (int i = 0; i < num_k; i++) {
        int k = k_vals[i];
        if (k > max_k) continue;

        Mat Ak = rebuild_img(U, S, V, k);

        double err = 0;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                double diff = A.d[i][j] - Ak.d[i][j];
                err += diff * diff;
            }
        err = sqrt(err);

        printf("%d\t%.2f\n", k, err);

        char name[100];
        sprintf(name, "reconstructed_k%d.png", k);
        save_img(Ak, name, w, h);

        free_mat(&Ak);
    }

    free_mat(&A);
    free_mat(&U);
    free_mat(&V);
    free(S);

    printf("\nCompression complete!\n");
    return 0;
}