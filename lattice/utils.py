import numpy as np
from onnx import TensorProto


DTYPE_MAP = {
    'float32': TensorProto.FLOAT,
    'float': TensorProto.FLOAT,
    'float64': TensorProto.DOUBLE,
    'int32': TensorProto.INT32,
    'int': TensorProto.INT64,
    'int64': TensorProto.INT64,
    'bool': TensorProto.BOOL,
    'string': TensorProto.STRING,
}

DTYPE_NAMES = {
    1: 'FLOAT',
    2: 'UINT8',
    3: 'INT8',
    5: 'INT16',
    6: 'INT32',
    7: 'INT64',
    9: 'BOOL',
    10: 'FLOAT16',
    11: 'DOUBLE',
}


def infer_shape(value):
    if isinstance(value, (int, float)):
        return [1]
    if isinstance(value, list):
        return [len(value)]
    if isinstance(value, np.ndarray):
        return list(value.shape)
    return [1]


def dtype_mapping(value):
    if isinstance(value, float):
        return TensorProto.FLOAT
    if isinstance(value, int):
        return TensorProto.INT64
    if isinstance(value, bool):
        return TensorProto.BOOL
    if isinstance(value, np.ndarray):
        return DTYPE_MAP.get(str(value.dtype), TensorProto.FLOAT)
    if isinstance(value, str):
        dtype_str = value.lower()
        if dtype_str in DTYPE_MAP:
            return DTYPE_MAP[dtype_str]
    return TensorProto.FLOAT


def format_shape(tensor_info):
    if not tensor_info.type.HasField('tensor_type'):
        return 'unknown'

    tensor_type = tensor_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        return 'dynamic'

    shape = tensor_type.shape
    if shape is None or len(shape.dim) == 0:
        return 'dynamic'

    parts = []
    for dim in shape.dim:
        if dim.HasField('dim_param'):
            parts.append(dim.dim_param)
        elif dim.HasField('dim_value'):
            parts.append(str(dim.dim_value))
        else:
            parts.append('?')

    return '[' + ', '.join(parts) + ']'


def pretty_print(model):
    graph = model.graph

    print("=" * 50)
    print("ONNX MODEL STRUCTURE")
    print("=" * 50)

    print(f"\nOpset: {model.opset_import[0].version}")

    print(f"\nInputs ({len(graph.input)}):")
    for inp in graph.input:
        shape_str = format_shape(inp)
        dtype = DTYPE_NAMES.get(inp.type.tensor_type.elem_type,
                                str(inp.type.tensor_type.elem_type))
        dynamic = " DYNAMIC" if 'dynamic' in shape_str.lower() or '?' in shape_str else ""
        print(f"  {inp.name:15s}  shape={shape_str:15s}  dtype={dtype}{dynamic}")

    print(f"\nInitializers ({len(graph.initializer)}):")
    for init in graph.initializer:
        if init.float_data:
            vals = list(init.float_data)
        elif init.int64_data:
            vals = list(init.int64_data)
        elif init.int32_data:
            vals = list(init.int32_data)
        else:
            vals = []
        print(f"  {init.name:15s}  value={vals}")

    print(f"\nNodes ({len(graph.node)}):")
    for i, node in enumerate(graph.node):
        inputs = ", ".join(node.input)
        outputs = ", ".join(node.output)
        attrs = ""
        if node.attribute:
            parts = []
            for attr in node.attribute:
                parts.append(f"{attr.name}={attr.i if attr.i else attr.f}")
            attrs = f"  [{', '.join(parts)}]"
        print(f"  [{i:>3d}] {node.op_type:20s}({inputs}) → {outputs}{attrs}")

    print(f"\nOutputs ({len(graph.output)}):")
    for out in graph.output:
        shape_str = format_shape(out)
        dtype = DTYPE_NAMES.get(out.type.tensor_type.elem_type,
                                str(out.type.tensor_type.elem_type))
        print(f"  {out.name:15s}  shape={shape_str:15s}  dtype={dtype}")

    print("=" * 50)
