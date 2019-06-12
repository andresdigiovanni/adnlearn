import onnxmltools
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from skl2onnx import convert_sklearn
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType, SequenceType
from skl2onnx.operator_converters.RandomForest import convert_sklearn_random_forest_classifier
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from onnxmltools.convert.common.shape_calculator import calculate_linear_classifier_output_shapes

def _register_converters():
    onnxmltools.convert.common.data_types.Int64TensorType = Int64TensorType
    onnxmltools.convert.common.data_types.StringTensorType = StringTensorType
    onnxmltools.convert.common.data_types.FloatTensorType = FloatTensorType
    onnxmltools.convert.common.data_types.DictionaryType = DictionaryType
    onnxmltools.convert.common.data_types.SequenceType = SequenceType

    update_registered_converter(SGDClassifier, 'SklearnLinearClassifier',
                                calculate_linear_classifier_output_shapes,
                                convert_sklearn_random_forest_classifier)

    update_registered_converter(LGBMClassifier, 'LightGbmLGBMClassifier',
                                calculate_linear_classifier_output_shapes,
                                convert_lightgbm)


def _convert_dataframe_schema(df, drop=None):
    inputs = []
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([1, 1])
        elif v == 'float64':
            t = FloatTensorType([1, 1])
        else:
            t = StringTensorType([1, 1])
        inputs.append((k, t))
    return inputs


def save_model_onnx(model, df, filename):
    _register_converters()
    inputs = _convert_dataframe_schema(df)
    onx = convert_sklearn(model, initial_types=inputs)
    
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
