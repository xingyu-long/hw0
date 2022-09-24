#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <cstring>
#include <vector>
using namespace std;


namespace py = pybind11;


vector<float> softmax(vector<float> data, size_t rows, size_t cols)
{
    vector<float> result(rows * cols, 0.0);
    // apply exp to all data 
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++) 
        {
            result[i * cols + j] = exp(data[i * cols + j]);
            // result[i * cols + j] = exp(data[i * cols + j]);
        }
    }

    // average by row
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0;
        for (size_t j = 0; j < cols; j++) {
            sum += result[i * cols + j];
        }

        for (size_t j = 0; j < cols; j++) {
            result[i * cols + j] /= sum;
        }
    }
    return result;
}


vector<float> new_dot(vector<float> a, vector<float> b, size_t a_row_start, size_t a_row_end, size_t a_col_start, size_t a_col_end,
                size_t b_row_start, size_t b_row_end, size_t b_col_start, size_t b_col_end)
{
    size_t a_rows = a_row_end - a_row_start;
    size_t a_cols = a_col_end - a_col_start;
    size_t b_rows = b_row_end - b_row_start;
    size_t b_cols = b_col_end - b_col_start;
    // printf("shape for 1nd input (%d, %d)\n", a_rows, a_cols);
    // printf("shape for 2nd input (%d, %d)\n", b_rows, b_cols);
    // vector<float> dot_res = new_dot(X_vector, theta_vector, batch_start, batch_end, 0, n, 0, n, 0, k); // m' * k
    // vector<float> g = new_dot(transpose_res, sub_res, 0, n, batch_start, batch_end, batch_start, batch_end, 0, k); 
    vector<float> result(a_rows * b_cols, 0.0);
    for (size_t i = a_row_start; i < a_row_end; i++) {
        for (size_t j = b_col_start; j < b_col_end; j++) {
            // prod[i][j] = prod[i][j] + a[i][k] * b[k][j];   
            // printf("for i = %d, j = %d\n", i, j);
            float sum = 0.0;
            for (size_t k = b_row_start; k < b_row_end; k++) {
                // NOTE: a belongs large size, so we don't need to subtract from row_start or col_start
                sum += a[i * a_cols + k] * b[k * b_cols + j];
                // printf("a[%d][%d](%f) * b[%d][%d](%f) \t", i, k, a[i * a_cols + k], k, j, b[k * b_cols + j]);
            }
            result[(i - a_row_start) * b_cols + (j - b_col_start)] = sum;
            // printf("\n");
            // printf("%f \n", result[i * b_cols + j]);  
        }
        // printf("\n");
    }
    return result;
}


vector<float> set_ones(vector<float> a, vector<int> b, size_t a_rows, size_t a_cols, size_t b_start, size_t b_end, size_t b_cols)
{
    assert(a_rows == (b_end - b_start) && "Cannot perform dot set_ones operation");
    // a -> I_y, b-> batch_y (label)
    for (size_t i = b_start; i < b_end; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            int idx = b[i * b_cols + j];
            a[(i - b_start) * a_cols + idx] = 1;
        }
    }
    return a;
}

vector<float> subtract(vector<float> a, vector<float> b, size_t row_start, size_t row_end, size_t col_start, size_t col_end)
{
    size_t rows = row_end - row_start;
    size_t cols = col_end - col_start;
    vector<float> result(rows * cols, 0.0);
    for (size_t i = row_start; i < row_end; i++) {
        for (size_t j = col_start; j < col_end; j++) {
            result[(i - row_start) * (cols) + (j - col_start)] = a[(i - row_start) * (cols) + (j - col_start)] - b[(i - row_start) * (cols) + (j - col_start)];
        }
    }
    return result;
}

void subtract_in_place(float *a, vector<float> b, size_t row_start, size_t row_end, size_t col_start, size_t col_end)
{
    size_t rows = row_end - row_start;
    size_t cols = col_end - col_start;
    for (size_t i = row_start; i < row_end; i++) {
        for (size_t j = col_start; j < col_end; j++) {
            a[i * (cols) + j] -= b[(i - row_start) * (cols) + (j - col_start)];
        }
    }
}

vector<float> transpose(vector<float> data, size_t batch_start, size_t batch_end, size_t cols)
{
    // input: (batch, cols)
    vector<float> result(cols * (batch_end - batch_start), 0.0);

    for (size_t i = batch_start; i < batch_end; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t curr_idx = i * cols + j;
            size_t next_idx = j * (batch_end - batch_start) + (i - batch_start);
            result[next_idx] = data[curr_idx];
        }
    }
    return result;
    // output: (cols, batch)
}

vector<float> multiply(vector<float> data, float lr, size_t rows, size_t cols)
{
    vector<float> result(rows * cols, 0.0);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i * cols + j] = data[i * cols + j] * lr;
        }
    }
    return result;
}

vector<float> divide(vector<float> data, size_t div, size_t rows, size_t cols)
{
    vector<float> result(rows * cols, 0.0);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i * cols + j] = data[i * cols + j] / div;
        }
    }
    return result;
}

void print_helper(float *data, size_t r_start, size_t r_end, size_t c_start, size_t c_end)
{
    for (size_t i = r_start; i < r_end; i++) {
        for (size_t j = c_start; j < c_end; j++) {
            printf("%f \t", data[i * (c_end - c_start) + j]);
        }
        printf("\n");
    }
}

void print_vector(vector<float> data, size_t r_start, size_t r_end, size_t c_start, size_t c_end) 
{
    for (size_t i = r_start; i < r_end; i++) {
        for (size_t j = c_start; j < c_end; j++) {
            printf("%f \t", data[(i - r_start) * (c_end - c_start) + (j - c_start)]);
        }
        printf("\n");
    }
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

    vector<float> X_vector (X, X + (m * n));
    vector<int> y_vector (y, y + m);
    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m / batch; i++) {
        // printf("--------- %zu iterations ---------\n", i);
        size_t batch_start = i * batch;
        size_t batch_end = (i + 1) * batch;
        // printf("------- X ------\n");
        // print_helper((float *)X, batch_start, batch_end, 0, n);
        vector<float> theta_vector(theta, theta + (n * k));
        // printf("------- theta ------\n");
        // print_vector(theta_vector, 0, n, 0, k);
        vector<float> dot_res = new_dot(X_vector, theta_vector, batch_start, batch_end, 0, n, 0, n, 0, k); // m' * k

        // printf("------- dot result ------\n");
        // print_vector(dot_res, 0, batch, 0, k);
        vector<float> batch_Z = softmax(dot_res, batch_end - batch_start, k); // m' * k
        // printf("------- softmax result ------\n");
        // print_vector(batch_Z, 0, batch, 0, k);
        vector<float> I_y((batch_end - batch_start) * k, 0.0);
        I_y = set_ones(I_y, y_vector, batch_end - batch_start, k, batch_start, batch_end, 1);
        // printf("------- set_ones result ------\n");
        // print_vector(I_y, 0, batch, 0, k);
        vector<float> sub_res = subtract(batch_Z, I_y, batch_start, batch_end, 0, k); // m' * k
        // printf("------- subtract result ------\n");
        // print_vector(sub_res, 0, batch, 0, k);
        vector<float> transpose_res = transpose(X_vector, batch_start, batch_end, n); // n * m'
        // printf("------- transpose result ------\n");
        // print_vector(transpose_res, 0, n, 0, batch);
        vector<float> g = new_dot(transpose_res, sub_res, 0, n, 0, batch, 0, batch, 0, k); // (n * m') dot (m' * k) -> n * k
        // printf("------- g result before divide ------\n");
        // print_vector(g, 0, n, 0, k);
        vector<float> div = divide(g, batch, n, k);
        // printf("------- g result ------\n");
        // print_vector(g, 0, n, 0, k);
        vector<float> mul = multiply(div, lr, n, k);
        // printf("------- multiply result ------\n");
        // print_vector(mul, 0, n, 0, k);
        subtract_in_place(theta, mul, 0, n, 0, k);
        // printf("------- theta result ------\n");
        // print_helper(theta, 0, n, 0, k);

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
