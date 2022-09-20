#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <cstring>


namespace py = pybind11;


float *softmax(float *data, size_t rows, size_t cols)
{
    float *result= (float *)malloc((rows * cols) * sizeof(float)); 
    // apply exp to all data 
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++) 
        {
            result[i * cols + j] = exp(data[i * cols + j]);
        }
    }

    // average by row
    for (int i = 0; i < rows; i++) {
        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += result[i * cols + j];
        }

        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = result[i * cols + j] / sum;
        }
    }
    return result;
}

float *dot(float *a, float *b, size_t a_batch_start, size_t a_batch_end, size_t a_cols, size_t b_rows, size_t b_cols) {
    assert(a_cols == b_rows && "Cannot perform dot product operation");
    float *result= (float *)malloc(((a_batch_end - a_batch_start) * b_cols) * sizeof(float));
    for (int i = a_batch_start; i < a_batch_end; i++) {
        for (int j = 0; j < b_cols; j++) {
            for (int k = 0; k < a_cols; k++) {
                result[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
            }
            printf("%f\t", result[i * b_cols + j]);
        }
        printf("\n");
    }
    return result;
}

float *zeros(size_t rows, size_t cols)
{
    return (float*)calloc((rows * cols), sizeof(float));
}

// verified
float *set_ones(float *a, float *b, size_t a_rows, size_t a_cols, size_t b_start, size_t b_end, size_t b_cols)
{
    assert(a_rows == (b_end - b_start) && "Cannot perform dot set_ones operation");
    // a -> I_y, b-> batch_y (label)
    for (int i = b_start; i < b_end; i++) {
        for (int j = 0; j < b_cols; j++) {
            int idx = b[i * b_cols + j];
            a[(i - b_start) * a_cols + idx] = 1;
        }
    }
    return a;
}

// verified
float *subtract(float *a, float *b, size_t row_start, size_t row_end, size_t col_start, size_t col_end)
{
    size_t rows = row_end - row_start;
    size_t cols = col_end - col_start;
    float *result = (float*)calloc((rows * cols), sizeof(float));
    for (int i = row_start; i < row_end; i++) {
        for (int j = col_start; j < col_end; j++) {
            result[i * (cols) + j] = a[i * (cols) + j] - b[i * (cols) + j];
        }
    }
    return result;
}

float *transpose(float *data, size_t batch_start, size_t batch_end, size_t cols)
{
    // input: (batch, cols)
    float *result = (float*)calloc((cols * (batch_end - batch_start)), sizeof(float));

    for (int i = batch_start; i < batch_end; i++) {
        for (int j = 0; j < cols; j++) {
            int curr_idx = i * cols + j;
            int next_idx = j * (batch_end - batch_start) + i;
            result[next_idx] = data[curr_idx];
        }
    }
    return result;
    // output: (cols, batch)
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int i = 0; i < m / batch; i++) {
        printf("--------- %d iterations ---------", i);
        size_t batch_start = i * batch;
        size_t batch_end = (i + 1) * batch;
        // float *batch_Z = softmax(dot(X, theta, batch_start, batch_end, n, n, k), batch_end - batch_start, k);
        // float *I_y = zeros(batch_end - batch_start, k);
        // set_ones(I_y, y, batch_end - batch_start, k, batch_start, batch_end, 1);
        // float *sub = subtract(batch_Z, I_y, 0, batch, 0, k);
        // float *g = dot(transpose(X, batch_start, batch_end, n), subtract, batch_start, batch_end, )


    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
