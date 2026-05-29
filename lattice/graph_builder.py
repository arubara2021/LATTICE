import onnx
from onnx import helper, TensorProto
from .ast_nodes import *
from .op_registry import (
    get_binary_op, get_unary_op, get_comparison_op,
    get_function, get_constant, is_special_function,
    is_constant, get_opset_version
)

COMPARISON_OPS = {'<', '>', '<=', '>=', '==', '!='}


class GraphBuilder:
    def __init__(self, shapes=None, dynamic=False):
        self.shapes = shapes or {}
        self.dynamic = dynamic
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.variables_seen = set()
        self.constants_seen = {}
        self.counter = 0
        self.output_shape = [1]

    def get_name(self, prefix):
        name = f"{prefix}_{self.counter}"
        self.counter += 1
        return name

    def build(self, ast):
        output_name = self.visit(ast)

        if isinstance(ast, BinaryOp) and ast.op in COMPARISON_OPS:
            output_type = TensorProto.BOOL
        else:
            output_type = TensorProto.FLOAT

        self.create_output(output_name, output_type)
        graph = helper.make_graph(
            self.nodes,
            "formula_graph",
            self.inputs,
            self.outputs,
            initializer=self.initializers
        )
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", get_opset_version())
        ])
        return model

    def visit(self, node):
        if isinstance(node, BinaryOp):
            return self.visit_binary_op(node)
        if isinstance(node, UnaryOp):
            return self.visit_unary_op(node)
        if isinstance(node, FunctionCall):
            return self.visit_function_call(node)
        if isinstance(node, Variable):
            return self.visit_variable(node)
        if isinstance(node, Constant):
            return self.visit_constant(node)
        raise Exception(f"Unknown node type: {type(node)}")

    def visit_binary_op(self, node):
        left_name = self.visit(node.left)
        right_name = self.visit(node.right)

        if node.op == '!=':
            eq_name = self.get_name("eq")
            self.nodes.append(helper.make_node(
                "Equal", [left_name, right_name], [eq_name]
            ))
            not_name = self.get_name("not")
            self.nodes.append(helper.make_node(
                "Not", [eq_name], [not_name]
            ))
            return not_name

        if node.op == '%':
            output_name = self.get_name("mod")
            self.nodes.append(helper.make_node(
                "Mod", [left_name, right_name], [output_name],
                fmod=1
            ))
            return output_name

        try:
            onnx_op = get_binary_op(node.op)
        except Exception:
            onnx_op = get_comparison_op(node.op)

        output_name = self.get_name(onnx_op.lower())
        self.nodes.append(helper.make_node(
            onnx_op, [left_name, right_name], [output_name]
        ))
        return output_name

    def visit_unary_op(self, node):
        operand_name = self.visit(node.operand)
        onnx_op = get_unary_op(node.op)
        if onnx_op is None:
            return operand_name
        output_name = self.get_name("neg")
        self.nodes.append(helper.make_node(
            onnx_op, [operand_name], [output_name]
        ))
        return output_name

    def visit_function_call(self, node):
        name = node.name
        if name == 'log2':
            return self.visit_log_base(node, 2.0)
        if name == 'log10':
            return self.visit_log_base(node, 10.0)
        if name == 'cbrt':
            return self.visit_cbrt(node)
        if name == 'deg':
            return self.visit_deg(node)
        if name == 'rad':
            return self.visit_rad(node)
        if name == 'factorial':
            arg_name = self.visit(node.args[0])
            return self.visit_factorial(arg_name)

        arg_names = [self.visit(arg) for arg in node.args]
        onnx_op = get_function(name)
        output_name = self.get_name(onnx_op.lower())
        self.nodes.append(helper.make_node(
            onnx_op, arg_names, [output_name]
        ))
        return output_name

    def visit_variable(self, node):
        name = node.name

        if is_constant(name):
            const_value = get_constant(name)
            if const_value in self.constants_seen:
                return self.constants_seen[const_value]
            const_name = self.get_name("const")
            self.constants_seen[const_value] = const_name
            self.initializers.append(
                helper.make_tensor(
                    const_name, TensorProto.FLOAT, [1], [float(const_value)]
                )
            )
            return const_name

        if name not in self.variables_seen:
            self.variables_seen.add(name)

            if self.dynamic:
                shape = []
            elif name in self.shapes:
                shape = self.shapes[name]
                self.output_shape = shape
            else:
                shape = [1]

            self.inputs.append(
                helper.make_tensor_value_info(
                    name, TensorProto.FLOAT, shape
                )
            )
        return name

    def visit_constant(self, node):
        value = float(node.value)
        if value in self.constants_seen:
            return self.constants_seen[value]
        name = self.get_name("const")
        self.constants_seen[value] = name
        self.initializers.append(
            helper.make_tensor(
                name, TensorProto.FLOAT, [1], [value]
            )
        )
        return name

    def visit_log_base(self, node, base):
        arg_name = self.visit(node.args[0])
        if base in self.constants_seen:
            base_name = self.constants_seen[base]
        else:
            base_name = self.get_name("const")
            self.constants_seen[base] = base_name
            self.initializers.append(
                helper.make_tensor(
                    base_name, TensorProto.FLOAT, [1], [base]
                )
            )
        log_arg_name = self.get_name("log")
        self.nodes.append(helper.make_node(
            "Log", [arg_name], [log_arg_name]
        ))
        log_base_name = self.get_name("log")
        self.nodes.append(helper.make_node(
            "Log", [base_name], [log_base_name]
        ))
        output_name = self.get_name("div")
        self.nodes.append(helper.make_node(
            "Div", [log_arg_name, log_base_name], [output_name]
        ))
        return output_name

    def visit_cbrt(self, node):
        arg_name = self.visit(node.args[0])
        third_value = 1.0 / 3.0
        if third_value in self.constants_seen:
            third_name = self.constants_seen[third_value]
        else:
            third_name = self.get_name("const")
            self.constants_seen[third_value] = third_name
            self.initializers.append(
                helper.make_tensor(
                    third_name, TensorProto.FLOAT, [1], [third_value]
                )
            )
        output_name = self.get_name("pow")
        self.nodes.append(helper.make_node(
            "Pow", [arg_name, third_name], [output_name]
        ))
        return output_name

    def visit_deg(self, node):
        arg_name = self.visit(node.args[0])
        factor_value = 3.141592653589793 / 180.0
        if factor_value in self.constants_seen:
            factor_name = self.constants_seen[factor_value]
        else:
            factor_name = self.get_name("const")
            self.constants_seen[factor_value] = factor_name
            self.initializers.append(
                helper.make_tensor(
                    factor_name, TensorProto.FLOAT, [1], [factor_value]
                )
            )
        output_name = self.get_name("mul")
        self.nodes.append(helper.make_node(
            "Mul", [arg_name, factor_name], [output_name]
        ))
        return output_name

    def visit_rad(self, node):
        arg_name = self.visit(node.args[0])
        factor_value = 180.0 / 3.141592653589793
        if factor_value in self.constants_seen:
            factor_name = self.constants_seen[factor_value]
        else:
            factor_name = self.get_name("const")
            self.constants_seen[factor_value] = factor_name
            self.initializers.append(
                helper.make_tensor(
                    factor_name, TensorProto.FLOAT, [1], [factor_value]
                )
            )
        output_name = self.get_name("mul")
        self.nodes.append(helper.make_node(
            "Mul", [arg_name, factor_name], [output_name]
        ))
        return output_name

    def visit_factorial(self, operand_name):
        two_pi_value = 6.283185307179586
        if two_pi_value in self.constants_seen:
            two_pi_name = self.constants_seen[two_pi_value]
        else:
            two_pi_name = self.get_name("const")
            self.constants_seen[two_pi_value] = two_pi_name
            self.initializers.append(
                helper.make_tensor(
                    two_pi_name, TensorProto.FLOAT, [1], [two_pi_value]
                )
            )
        e_value = 2.718281828459045
        if e_value in self.constants_seen:
            e_name = self.constants_seen[e_value]
        else:
            e_name = self.get_name("const")
            self.constants_seen[e_value] = e_name
            self.initializers.append(
                helper.make_tensor(
                    e_name, TensorProto.FLOAT, [1], [e_value]
                )
            )
        two_pi_n_name = self.get_name("mul")
        self.nodes.append(helper.make_node(
            "Mul", [two_pi_name, operand_name], [two_pi_n_name]
        ))
        sqrt_name = self.get_name("sqrt")
        self.nodes.append(helper.make_node(
            "Sqrt", [two_pi_n_name], [sqrt_name]
        ))
        n_over_e_name = self.get_name("div")
        self.nodes.append(helper.make_node(
            "Div", [operand_name, e_name], [n_over_e_name]
        ))
        pow_name = self.get_name("pow")
        self.nodes.append(helper.make_node(
            "Pow", [n_over_e_name, operand_name], [pow_name]
        ))
        output_name = self.get_name("mul")
        self.nodes.append(helper.make_node(
            "Mul", [sqrt_name, pow_name], [output_name]
        ))
        return output_name

    def create_output(self, output_name, output_type=TensorProto.FLOAT):
        if self.dynamic:
            output_shape = []
        else:
            output_shape = self.output_shape

        self.outputs = [
            helper.make_tensor_value_info(
                output_name, output_type, output_shape
            )
        ]
