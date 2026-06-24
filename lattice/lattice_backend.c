#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

// Helper macros for math functions to handle float/double
#define UNARY_OP_LOOP(out, in, size, func) \
    _Pragma("omp parallel for simd") \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = func(in[i]); \
    }

#define BINARY_OP_LOOP(out, in1, in2, size, op) \
    _Pragma("omp parallel for simd") \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = in1[i] op in2[i]; \
    }

#define TERNARY_OP_LOOP(out, in1, in2, in3, size, expr) \
    _Pragma("omp parallel for simd") \
    for (npy_intp i = 0; i < size; i++) { \
        out[i] = expr; \
    }

// --- Kernel Implementations ---

static void kernel_add(double *out, double *in1, double *in2, npy_intp size) {
    BINARY_OP_LOOP(out, in1, in2, size, +);
}

static void kernel_sub(double *out, double *in1, double *in2, npy_intp size) {
    BINARY_OP_LOOP(out, in1, in2, size, -);
}

static void kernel_mul(double *out, double *in1, double *in2, npy_intp size) {
    BINARY_OP_LOOP(out, in1, in2, size, *);
}

static void kernel_div(double *out, double *in1, double *in2, npy_intp size) {
    BINARY_OP_LOOP(out, in1, in2, size, /);
}

static void kernel_pow(double *out, double *in1, double *in2, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        out[i] = pow(in1[i], in2[i]);
    }
}

static void kernel_sin(double *out, double *in, npy_intp size) {
    UNARY_OP_LOOP(out, in, size, sin);
}

static void kernel_cos(double *out, double *in, npy_intp size) {
    UNARY_OP_LOOP(out, in, size, cos);
}

static void kernel_tan(double *out, double *in, npy_intp size) {
    UNARY_OP_LOOP(out, in, size, tan);
}

static void kernel_exp(double *out, double *in, npy_intp size) {
    UNARY_OP_LOOP(out, in, size, exp);
}

static void kernel_log(double *out, double *in, npy_intp size) {
    UNARY_OP_LOOP(out, in, size, log);
}

static void kernel_sqrt(double *out, double *in, npy_intp size) {
    UNARY_OP_LOOP(out, in, size, sqrt);
}

static void kernel_abs(double *out, double *in, npy_intp size) {
    UNARY_OP_LOOP(out, in, size, fabs);
}

static void kernel_neg(double *out, double *in, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        out[i] = -in[i];
    }
}

static void kernel_constant(double *out, double val, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        out[i] = val;
    }
}

// --- Fused Kernels (The Secret Sauce for Speed) ---
// Example: Fused Multiply-Add (a * b + c)
static void kernel_fma(double *out, double *in1, double *in2, double *in3, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        out[i] = (in1[i] * in2[i]) + in3[i];
    }
}

// Example: Fused Activation (sin(x * w) + b)
static void kernel_fused_sin_linear(double *out, double *in, double *w, double *b, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        out[i] = sin(in[i] * w[i]) + b[i];
    }
}

static void kernel_fused_hypot(double *out, double *in1, double *in2, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        out[i] = sqrt((in1[i] * in1[i]) + (in2[i] * in2[i]));
    }
}

static void kernel_fused_sin2cos2(double *out, double *in1, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        double s = sin(in1[i]);
        double c = cos(in1[i]);
        out[i] = s*s + c*c;
    }
}

static void kernel_fused_exp_decay(double *out, double *A, double *lambda, double *t, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        out[i] = A[i] * exp(-lambda[i] * t[i]);
    }
}

static void kernel_fused_gaussian(double *out, double *A, double *mu, double *sigma, double *x, npy_intp size) {
    _Pragma("omp parallel for simd")
    for (npy_intp i = 0; i < size; i++) {
        double z = (x[i] - mu[i]) / sigma[i];
        out[i] = A[i] * exp(-0.5 * z * z);
    }
}

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

    // Prepare Input Arrays
    PyArrayObject *arr_in1 = NULL, *arr_in2 = NULL, *arr_in3 = NULL, *arr_in4 = NULL;
    PyArrayObject *arr_out = NULL;
    
    int num_inputs = PyList_Size(inputs_list);
    if (num_inputs > 0) arr_in1 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 0), NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (num_inputs > 1) arr_in2 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 1), NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (num_inputs > 2) arr_in3 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 2), NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (num_inputs > 3) arr_in4 = (PyArrayObject*)PyArray_FROM_OTF(PyList_GetItem(inputs_list, 3), NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (!arr_in1 && num_inputs > 0) goto error;
    if (!arr_in2 && num_inputs > 1) goto error;
    if (!arr_in3 && num_inputs > 2) goto error;
    if (!arr_in4 && num_inputs > 3) goto error;

    // Create Output Array
    npy_intp dims[1] = {size};
    arr_out = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!arr_out) goto error;

    double *out_data = (double*)PyArray_DATA(arr_out);
    double *in1_data = arr_in1 ? (double*)PyArray_DATA(arr_in1) : NULL;
    double *in2_data = arr_in2 ? (double*)PyArray_DATA(arr_in2) : NULL;
    double *in3_data = arr_in3 ? (double*)PyArray_DATA(arr_in3) : NULL;
    double *in4_data = arr_in4 ? (double*)PyArray_DATA(arr_in4) : NULL;

    // Dispatch to C Kernel
    if (strcmp(op_name, "add") == 0) {
        kernel_add(out_data, in1_data, in2_data, size);
    } else if (strcmp(op_name, "sub") == 0) {
        kernel_sub(out_data, in1_data, in2_data, size);
    } else if (strcmp(op_name, "mul") == 0) {
        kernel_mul(out_data, in1_data, in2_data, size);
    } else if (strcmp(op_name, "div") == 0) {
        kernel_div(out_data, in1_data, in2_data, size);
    } else if (strcmp(op_name, "pow") == 0) {
        kernel_pow(out_data, in1_data, in2_data, size);
    } else if (strcmp(op_name, "sin") == 0) {
        kernel_sin(out_data, in1_data, size);
    } else if (strcmp(op_name, "cos") == 0) {
        kernel_cos(out_data, in1_data, size);
    } else if (strcmp(op_name, "tan") == 0) {
        kernel_tan(out_data, in1_data, size);
    } else if (strcmp(op_name, "exp") == 0) {
        kernel_exp(out_data, in1_data, size);
    } else if (strcmp(op_name, "log") == 0) {
        kernel_log(out_data, in1_data, size);
    } else if (strcmp(op_name, "sqrt") == 0) {
        kernel_sqrt(out_data, in1_data, size);
    } else if (strcmp(op_name, "abs") == 0) {
        kernel_abs(out_data, in1_data, size);
    } else if (strcmp(op_name, "neg") == 0) {
        kernel_neg(out_data, in1_data, size);
    } else if (strcmp(op_name, "constant") == 0) {
        double val = 0.0;
        if (constants_obj) val = PyFloat_AsDouble(constants_obj);
        kernel_constant(out_data, val, size);
    } 
    // Fused kernels example
    else if (strcmp(op_name, "fma") == 0) {
        kernel_fma(out_data, in1_data, in2_data, in3_data, size);
    } else if (strcmp(op_name, "fused_hypot") == 0) {
        kernel_fused_hypot(out_data, in1_data, in2_data, size);
    } else if (strcmp(op_name, "fused_sin2cos2") == 0) {
        kernel_fused_sin2cos2(out_data, in1_data, size);
    } else if (strcmp(op_name, "fused_exp_decay") == 0) {
        kernel_fused_exp_decay(out_data, in1_data, in2_data, in3_data, size);
    } else if (strcmp(op_name, "fused_gaussian") == 0) {
        kernel_fused_gaussian(out_data, in1_data, in2_data, in3_data, in4_data, size);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Unsupported operation for C backend");
        goto error;
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

// Ensure math library symbols are properly linked
__attribute__((constructor))
void init_math_symbols(void) {
    // Force linking of math functions
    volatile double (*fp)(double) = &tan;
    (void)fp;
}
