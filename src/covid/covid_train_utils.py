from typing import List
import keras_tuner
import tensorflow as tf
import tfx.v1 as tfx
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma

from tensorflow import keras
from tfx_bsl.public import tfxio

_DENSE_FLOAT_FEATURES = ['density', 
                        'demographic', 
                        'population_p65', 
                        'population_p14', 
                        'gdp', 
                        'area',
                        'school', 
                        'school_5days',
                        'school_10days',
                        'school_15days',
                        'school_30days',
                        'public_transport', 
                        'public_transport_5days',
                        'public_transport_10days',
                        'public_transport_15days',
                        'public_transport_30days',
                        'international_transport',
                        'international_transport_5days',
                        'international_transport_10days',
                        'international_transport_15days',
                        'international_transport_30days',
                        'workplace',
                        'workplace_5days',
                        'workplace_10days',
                        'workplace_15days',
                        'workplace_30days',
                        'residential',
                        'residential_5days',
                        'residential_10days',
                        'residential_15days',
                        'residential_30days',
                        'retail/recreation',
                        'retail/recreation_5days', 
                        'retail/recreation_10days',                           
                        'retail/recreation_15days',
                        'retail/recreation_30days',
                        'grocery/pharmacy',
                        'grocery/pharmacy_5days',
                        'grocery/pharmacy_10days',
                        'grocery/pharmacy_15days',
                        'grocery/pharmacy_30days',
                        'parks',
                        'parks_5days',
                        'parks_10days',
                        'parks_15days',
                        'parks_30days',
                        'transit_stations',
                        'transit_stations_5days',
                        'transit_stations_10days',
                        'transit_stations_15days',
                        'transit_stations_30days']


ONE_HOT_FEATURES = {
"region": 11
}

_LABEL_KEY = 'R'

_SLICING_FEATURE_KEYS = ['region']

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

def input_fn(file_pattern: List[str], data_accessor: tfx.components.DataAccessor, tf_transform_output: tft.TFTransformOutput, batch_size: int) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema).repeat()


def _get_hyperparameters() -> keras_tuner.HyperParameters:
    hp = keras_tuner.HyperParameters()
    hp.Choice('hidden_layers', [
                "1000, 50",
                "50, 100, 50",
                "50, 100, 100",
                "50, 500, 50",
            ])

    hp.Choice('learning_rate', [0.0001, 0.05])

    return hp


def _transformed_name(key):
      return key + '_xf'


def _fill_in_missing(x):
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def _make_keras_model(hyperparameters: keras_tuner.HyperParameters) -> tf.keras.Model:
    input_features = []
    
    for feature, dim in ONE_HOT_FEATURES.items():
        input_features.append(tf.keras.Input(shape=(dim + 1,), name=_transformed_name(feature)))
    
    for feature in _DENSE_FLOAT_FEATURES:
        input_features.append(tf.keras.Input(shape=(1,), name=_transformed_name(feature)))

    d = keras.layers.concatenate(input_features)

    for l in hyperparameters.get('hidden_layers').split(","):
        d = keras.layers.Dense(int(l.strip()), activation='relu')(d)
    
    outputs = keras.layers.Dense(1)(d)

    model = keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(hyperparameters.get('learning_rate')),
        loss='mean_squared_error',
        metrics=[keras.metrics.MeanSquaredError()])

    return model


def _make_serving_signatures(model, tf_transform_output: tft.TFTransformOutput):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(_LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)

        outputs = model(transformed_features)
        # TODO(b/154085620): Convert the predicted labels from the model using a
        # reverse-lookup (opposite of transform.py).
        return {'outputs': outputs}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)
        return transformed_features

    return {
        'serving_default': serve_tf_examples_fn,
        'transform_features': transform_features_fn
    }


def _convert_num_to_one_hot(label_tensor, num_labels=2):
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    outputs = {}
    for key in ONE_HOT_FEATURES.keys():
        dim = ONE_HOT_FEATURES[key]
        index = tft.compute_and_apply_vocabulary(_fill_in_missing(inputs[key]), top_k=dim + 1)
        outputs[_transformed_name(key)] = _convert_num_to_one_hot(index, num_labels=dim + 1)
    for key in _DENSE_FLOAT_FEATURES:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))
    outputs[_transformed_name(_LABEL_KEY)] = tft.scale_to_z_score(_fill_in_missing(inputs[_LABEL_KEY]))

    return outputs


def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:  
    tuner = keras_tuner.RandomSearch(
        _make_keras_model,
        max_trials=8,
        hyperparameters=_get_hyperparameters(),
        allow_new_entries=False,
        objective=keras_tuner.Objective('mean_squared_error', 'min'),
        directory=fn_args.working_dir,
        project_name='covid_tuning')

    transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        transform_graph,
        _TRAIN_BATCH_SIZE)

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        transform_graph,
        _EVAL_BATCH_SIZE)

    return tfx.components.TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        })


def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        _TRAIN_BATCH_SIZE)

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        _EVAL_BATCH_SIZE)

    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hparams = _get_hyperparameters()

    model = _make_keras_model(hparams)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    signatures = _make_serving_signatures(model, tf_transform_output)
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


eval_config = tfma.EvalConfig(
    model_specs = [
        #tfma.ModelSpec(signature_name='eval'),
        tfma.ModelSpec(label_key='R')
    ],
    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=_SLICING_FEATURE_KEYS)
    ],
    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='MeanSquaredError',
                 threshold=tfma.MetricThreshold(
                     value_threshold=tfma.GenericValueThreshold(
                         upper_bound={'value': 1000}),
                     change_threshold=tfma.GenericChangeThreshold(
                         direction=tfma.MetricDirection.LOWER_IS_BETTER,
                         absolute={'value': -1e-10})))
            ],
        )
    ]
)