import onnx


class Exporter:
    def export_to_file(self, model, filepath):
        onnx.save(model, filepath)
        return filepath

    def export_to_bytes(self, model):
        return model.SerializeToString()

    def validate(self, model, strict=True):
        if strict:
            onnx.checker.check_model(model)
        else:
            try:
                onnx.checker.check_model(model)
            except onnx.checker.ValidationError:
                pass
        return True
