import datetime
import os

from covid.covid_train_utils import eval_config

from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Evaluator, Pusher
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.orchestration import pipeline, metadata
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner, AirflowPipelineConfig
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.dsl.components.common import resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

from airflow import DAG

_pipeline_name = 'covid-train'

_airflow_root = os.path.join(os.environ['HOME'], 'airflow')
_data_root = os.path.join(_airflow_root, 'data', 'covid')
_module_file = os.path.join(_airflow_root, 'dags', 'covid', 'covid_train_utils.py')

_tfx_root = os.path.join(_airflow_root, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name, 'metadata.db')
_serving_model_dir = os.path.join(_tfx_root, 'serving_model', _pipeline_name)

_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2022,1,1)
}


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str, module_file: str, metadata_path: str, serving_model_dir: str) -> pipeline.Pipeline:
    example_gen = CsvExampleGen(
        input_base=os.path.join(data_root, 'merged')
    )
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    infer_schema = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False
    )
    validate_stats = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=infer_schema.outputs['schema']
    )
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=infer_schema.outputs['schema'],
        module_file=module_file
    )
    tuner = Tuner(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=20),
        eval_args=trainer_pb2.EvalArgs(num_steps=5)
    )
    trainer = Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=infer_schema.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000)
    )
    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(
            type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )
    pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=serving_model_dir)
        )
    )
    
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            infer_schema,
            validate_stats,
            transform,
            tuner,
            trainer,
            model_resolver,
            evaluator,
            pusher
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        )
    )

DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        metadata_path=_metadata_path,
        serving_model_dir=_serving_model_dir,
    )
)