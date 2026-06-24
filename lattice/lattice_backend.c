#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#ifdef _MSC_VER
#define PRAGMA_OMP_SIMD __pragma(omp parallel for simd)
#else
#define PRAGMA_OMP_SIMD _Pragma("omp parallel for simd")
#endif

// Helper macros for math functions to handle float/double
#define UNARY_OP_LOOP(T, out, in, size, func) \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = func(in[i]); \
    }

#define BINARY_OP_LOOP(T, out, in1, in2, size, op) \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = in1[i] op in2[i]; \
    }

// --- Kernel Implementations ---
// We create a macro to stamp out both float and double versions
#define DECLARE_KERNELS(T, SUFFIX, FUNC_SUFFIX) \
static void kernel_add_##SUFFIX(T *out, T *in1, T *in2, npy_intp size) { \
    BINARY_OP_LOOP(T, out, in1, in2, size, +); \
} \
static void kernel_sub_##SUFFIX(T *out, T *in1, T *in2, npy_intp size) { \
    BINARY_OP_LOOP(T, out, in1, in2, size, -); \
} \
static void kernel_mul_##SUFFIX(T *out, T *in1, T *in2, npy_intp size) { \
    BINARY_OP_LOOP(T, out, in1, in2, size, *); \
} \
static void kernel_div_##SUFFIX(T *out, T *in1, T *in2, npy_intp size) { \
    BINARY_OP_LOOP(T, out, in1, in2, size, /); \
} \
static void kernel_pow_##SUFFIX(T *out, T *in1, T *in2, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = pow##FUNC_SUFFIX(in1[i], in2[i]); \
    } \
} \
static void kernel_sin_##SUFFIX(T *out, T *in, npy_intp size) { \
    UNARY_OP_LOOP(T, out, in, size, sin##FUNC_SUFFIX); \
} \
static void kernel_cos_##SUFFIX(T *out, T *in, npy_intp size) { \
    UNARY_OP_LOOP(T, out, in, size, cos##FUNC_SUFFIX); \
} \
static void kernel_tan_##SUFFIX(T *out, T *in, npy_intp size) { \
    UNARY_OP_LOOP(T, out, in, size, tan##FUNC_SUFFIX); \
} \
static void kernel_exp_##SUFFIX(T *out, T *in, npy_intp size) { \
    UNARY_OP_LOOP(T, out, in, size, exp##FUNC_SUFFIX); \
} \
static void kernel_log_##SUFFIX(T *out, T *in, npy_intp size) { \
    UNARY_OP_LOOP(T, out, in, size, log##FUNC_SUFFIX); \
} \
static void kernel_sqrt_##SUFFIX(T *out, T *in, npy_intp size) { \
    UNARY_OP_LOOP(T, out, in, size, sqrt##FUNC_SUFFIX); \
} \
static void kernel_abs_##SUFFIX(T *out, T *in, npy_intp size) { \
    UNARY_OP_LOOP(T, out, in, size, fabs##FUNC_SUFFIX); \
} \
static void kernel_neg_##SUFFIX(T *out, T *in, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = -in[i]; \
    } \
} \
static void kernel_constant_##SUFFIX(T *out, T val, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = val; \
    } \
} \
static void kernel_fma_##SUFFIX(T *out, T *in1, T *in2, T *in3, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = (in1[i] * in2[i]) + in3[i]; \
    } \
} \
static void kernel_fused_hypot_##SUFFIX(T *out, T *in1, T *in2, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = sqrt##FUNC_SUFFIX((in1[i] * in1[i]) + (in2[i] * in2[i])); \
    } \
} \
static void kernel_fused_sin2cos2_##SUFFIX(T *out, T *in1, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        T s = sin##FUNC_SUFFIX(in1[i]); \
        T c = cos##FUNC_SUFFIX(in1[i]); \
        out[i] = s*s + c*c; \
    } \
} \
static void kernel_fused_exp_decay_##SUFFIX(T *out, T *A, T *lambda, T *t, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = A[i] * exp##FUNC_SUFFIX(-lambda[i] * t[i]); \
    } \
} \
static void kernel_fused_gaussian_##SUFFIX(T *out, T *A, T *mu, T *sigma, T *x, npy_intp size) { \
    PRAGMA_OMP_SIMD \
    for (npy_intp i = 0; i < size; i++) { \
        T z = (x[i] - mu[i]) / sigma[i]; \
        out[i] = A[i] * exp##FUNC_SUFFIX(-0.5 * z * z); \
    } \
}

DECLARE_KERNELS(double, f64, )
DECLARE_KERNELS(float, f32, f)

// --- Python Interface ---

static PyObject* lattice_execute(PyObject *self, PyObject *args) {
    PyObject *op_name_obj;
    PyObject *inputs_list;
    PyObject *constants_obj = NULL;
    npy_intp size;
    
    // Parse arguments: op_name, inputs_list, size, constants(optional)
    if (!PyArg_ParseTuple(args, "OO!n|O", &op_name_obj, &PyList_Type, &inputs_list, &size, &constants_obj)) {
        return NULL;
    }

    const char *op_name = PyUnicode_AsUTF8(op_name_obj);
    if (!op_name) return NULL;

    // Detect Input Data Type
    int dtype = NPY_FLOAT64;
    int num_inputs = PyList_Size(inputs_list);
    if (num_inputs > 0) {
        PyObject *first_item = PyList_GetItem(inputs_list, 0);
        if (PyArray_Check(first_item)) {
            dtype = PyArray_TYPE((PyArrayObject *)first_item);
        }
    }

    // We only optimize for NPY_FLOAT32 and NPY_FLOAT64
    if (dtype != NPY_FLOAT32 && dtype != NPY_FLOAT64) {
        dtype = NPY_FLOAT64; // Default fallback to casting to double
    }

    // Prepare Input Arrays
    PyArrayObject *arr_in1 = NULL, *arr_in2 = NULL, *arr_in3 = NULL, *arr_in4 = NULL;
    PyArrayObject *arr_out = NULL;
    
    if (num_inputs > 0) arr_in1 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 0), dtype, NPY_ARRAY_IN_ARRAY);
    if (num_inputs > 1) arr_in2 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 1), dtype, NPY_ARRAY_IN_ARRAY);
    if (num_inputs > 2) arr_in3 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 2), dtype, NPY_ARRAY_IN_ARRAY);
    if (num_inputs > 3) arr_in4 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 3), dtype, NPY_ARRAY_IN_ARRAY);

    if (!arr_in1 && num_inputs > 0) goto error;
    if (!arr_in2 && num_inputs > 1) goto error;
    if (!arr_in3 && num_inputs > 2) goto error;
    if (!arr_in4 && num_inputs > 3) goto error;

    // Create Output Array
    npy_intp dims[1] = {size};
    arr_out = (PyArrayObject*)PyArray_SimpleNew(1, dims, dtype);
    if (!arr_out) goto error;

    // Dispatch to Typed C Kernel
    if (dtype == NPY_FLOAT32) {
        float *out_data = (float*)PyArray_DATA(arr_out);
        float *in1_data = arr_in1 ? (float*)PyArray_DATA(arr_in1) : NULL;
        float *in2_data = arr_in2 ? (float*)PyArray_DATA(arr_in2) : NULL;
        float *in3_data = arr_in3 ? (float*)PyArray_DATA(arr_in3) : NULL;
        float *in4_data = arr_in4 ? (float*)PyArray_DATA(arr_in4) : NULL;

        if (strcmp(op_name, "add") == 0 && in1_data && in2_data) kernel_add_f32(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "sub") == 0 && in1_data && in2_data) kernel_sub_f32(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "mul") == 0 && in1_data && in2_data) kernel_mul_f32(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "div") == 0 && in1_data && in2_data) kernel_div_f32(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "pow") == 0 && in1_data && in2_data) kernel_pow_f32(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "sin") == 0 && in1_data) kernel_sin_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "cos") == 0 && in1_data) kernel_cos_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "tan") == 0 && in1_data) kernel_tan_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "exp") == 0 && in1_data) kernel_exp_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "log") == 0 && in1_data) kernel_log_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "sqrt") == 0 && in1_data) kernel_sqrt_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "abs") == 0 && in1_data) kernel_abs_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "neg") == 0 && in1_data) kernel_neg_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "constant") == 0) {
            float val = 0.0f;
            if (constants_obj) val = (float)PyFloat_AsDouble(constants_obj);
            kernel_constant_f32(out_data, val, size);
        }
        else if (strcmp(op_name, "fma") == 0 && in1_data && in2_data && in3_data) kernel_fma_f32(out_data, in1_data, in2_data, in3_data, size);
        else if (strcmp(op_name, "fused_hypot") == 0 && in1_data && in2_data) kernel_fused_hypot_f32(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "fused_sin2cos2") == 0 && in1_data) kernel_fused_sin2cos2_f32(out_data, in1_data, size);
        else if (strcmp(op_name, "fused_exp_decay") == 0 && in1_data && in2_data && in3_data) kernel_fused_exp_decay_f32(out_data, in1_data, in2_data, in3_data, size);
        else if (strcmp(op_name, "fused_gaussian") == 0 && in1_data && in2_data && in3_data && in4_data) kernel_fused_gaussian_f32(out_data, in1_data, in2_data, in3_data, in4_data, size);
        else {
            PyErr_SetString(PyExc_ValueError, "Unsupported operation or missing arguments for C backend");
            goto error;
        }
    } else {
        double *out_data = (double*)PyArray_DATA(arr_out);
        double *in1_data = arr_in1 ? (double*)PyArray_DATA(arr_in1) : NULL;
        double *in2_data = arr_in2 ? (double*)PyArray_DATA(arr_in2) : NULL;
        double *in3_data = arr_in3 ? (double*)PyArray_DATA(arr_in3) : NULL;
        double *in4_data = arr_in4 ? (double*)PyArray_DATA(arr_in4) : NULL;

        if (strcmp(op_name, "add") == 0 && in1_data && in2_data) kernel_add_f64(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "sub") == 0 && in1_data && in2_data) kernel_sub_f64(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "mul") == 0 && in1_data && in2_data) kernel_mul_f64(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "div") == 0 && in1_data && in2_data) kernel_div_f64(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "pow") == 0 && in1_data && in2_data) kernel_pow_f64(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "sin") == 0 && in1_data) kernel_sin_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "cos") == 0 && in1_data) kernel_cos_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "tan") == 0 && in1_data) kernel_tan_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "exp") == 0 && in1_data) kernel_exp_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "log") == 0 && in1_data) kernel_log_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "sqrt") == 0 && in1_data) kernel_sqrt_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "abs") == 0 && in1_data) kernel_abs_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "neg") == 0 && in1_data) kernel_neg_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "constant") == 0) {
            double val = 0.0;
            if (constants_obj) val = PyFloat_AsDouble(constants_obj);
            kernel_constant_f64(out_data, val, size);
        }
        else if (strcmp(op_name, "fma") == 0 && in1_data && in2_data && in3_data) kernel_fma_f64(out_data, in1_data, in2_data, in3_data, size);
        else if (strcmp(op_name, "fused_hypot") == 0 && in1_data && in2_data) kernel_fused_hypot_f64(out_data, in1_data, in2_data, size);
        else if (strcmp(op_name, "fused_sin2cos2") == 0 && in1_data) kernel_fused_sin2cos2_f64(out_data, in1_data, size);
        else if (strcmp(op_name, "fused_exp_decay") == 0 && in1_data && in2_data && in3_data) kernel_fused_exp_decay_f64(out_data, in1_data, in2_data, in3_data, size);
        else if (strcmp(op_name, "fused_gaussian") == 0 && in1_data && in2_data && in3_data && in4_data) kernel_fused_gaussian_f64(out_data, in1_data, in2_data, in3_data, in4_data, size);
        else {
            PyErr_SetString(PyExc_ValueError, "Unsupported operation or missing arguments for C backend");
            goto error;
        }
    }

    // Cleanup inputs
    Py_XDECREF(arr_in1);
    Py_XDECREF(arr_in2);
    Py_XDECREF(arr_in3);
    Py_XDECREF(arr_in4);

    return (PyObject*)arr_out;

error:
    Py_XDECREF(arr_in1);
    Py_XDECREF(arr_in2);
    Py_XDECREF(arr_in3);
    Py_XDECREF(arr_in4);
    Py_XDECREF(arr_out);
    return NULL;
}

static PyMethodDef LatticeMethods[] = {
    {"execute", lattice_execute, METH_VARARGS, "Execute math kernels in C"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef latticemodule = {
    PyModuleDef_HEAD_INIT,
    "lattice_backend",
    "High-performance C backend for Formula2ONNX",
    -1,
    LatticeMethods
};

PyMODINIT_FUNC PyInit_lattice_backend(void) {
    import_array(); // Required for NumPy C API
    return PyModule_Create(&latticemodule);
}

#ifndef _MSC_VER
// Ensure math library symbols are properly linked (GCC/Clang only)
__attribute__((constructor))
void init_math_symbols(void) {
    // Force linking of math functions
    volatile double (*fp)(double) = &tan;
    (void)fp;
}
#endif
