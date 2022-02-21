from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from covid.prepare_data_utils import merge

with DAG('prepare-data', description='Prepare data for the training',
          schedule_interval=None,
          start_date=datetime(2022, 1, 1), catchup=False) as dag:

    hello_operator = PythonOperator(task_id='merge', python_callable=merge, dag=dag)

    hello_operator