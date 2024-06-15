import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Param, Metric, RunTag
from dotenv import load_dotenv
from typing import Optional, List, Union, Any, Dict

from utils import get_current_time_millis

class MLflowClientWrapper:
    def __init__(self,
        prod: bool,
    ):
        
        if prod:
            uri = os.getenv('MLFLOW_DEV_URI')
        else:
            uri = os.getenv('MLFLOW_PROD_URI')
            
        self.client = MlflowClient(uri)
        self.experiment_name = None
        self.experiment_id = None
        self.run_client = None
        self.run_name = None
        self.run_id = None
    
    def set_experiment(self,
        experiment_name: str
    ):
        
        experiment_names = self._list_experiments(
            client=self.client)
        
        if experiment_name in experiment_names:
            print(f"Experiment '{experiment_name}' already exists. Setting the experiment to '{experiment_name}'")
            self.experiment_id = self.client.get_experiment_by_name(
                name=experiment_name).experiment_id
        else:
            print(f"Creating new experiment with '{experiment_name}'")
            self.experiment_id = self.client.create_experiment(
                name=experiment_name)
    
    def set_run(self,
        run_name: str,
        tags: Optional[Dict] = None
    ):
        
        assert self.experiment_id is not None, f"experiment_id or experiment_name is None. Finish 'set_experiment()' first"
        
        # Design Choice - throws an error if run name exists
        if self._get_run_id() is not None:
            raise ValueError(f"A run with the name '{run_name}' in '{self.experiment_name}' already exists. Please choose a different run name")
        
        
        # Add OS Uname
        if tags is None:
            tags = {}
            
        if "Run Name" not in tags.keys():
            tags['Run Name'] = run_name
            
        if "Owner" not in tags.keys():
            tags['Owner'] = os.uname()[1].split("-")[1]
        
        self.run_client = self.client.create_run(
            experiment_id = self.experiment_id,
            tags = tags,
            run_name = run_name
        )
        
        self.run_id = self.run_client.info.run_id
        self.run_name = run_name
    
    def log_param(self,
        key: str,
        value: Any,
        synchronous: Optional[bool] = None
    ):
        
        self.client.log_param(
            self.run_id, key=key, value=value, synchronous=synchronous)
    
    def log_params(self,
        params: Dict[str, Any],
        synchronous: Optional[bool] = None
    ):
        
        params_arr = [Param(key, str(value)) for key, value in params.items()]
        self.client.log_batch(
            run_id = self.run_id,
            metrics = [],
            params = params_arr,
            tags = [],
            synchronous = synchronous
        )

    def log_metric(self,
        key: str,
        value: float,
        synchronous: Optional[bool] = None
    ):
        
        self.client.log_metric(
            self.run_id, key=key, value=value, synchronous=synchronous)

    def log_metrics(self,
        metrics: Dict[str, float],
        synchronous: Optional[bool] = None
    ):
        
        timestamp = get_current_time_millis()
        metrics_arr = [Metric(key, value, timestamp, 0) for key, value in metrics.items()]
        self.client.log_batch(
            run_id = self.run_id,
            metrics = metrics_arr,
            params = [],
            tags = [],
            synchronous = synchronous
        )
    
    def log_tag(self,
        key: str,
        value: Any,
        synchronous: Optional[bool] = None
    ):
        
        self.client.set_tag(
            self.run_id, key=key, value=value, synchronous=synchronous)

    def log_tags(self,
        tags: Dict[str, Any],
        synchronous: Optional[bool] = None
    ):
        tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
        self.client.log_batch(
            run_id = self.run_id,
            metrics = [],
            params = [],
            tags = tags_arr,
            synchronous = synchronous
        )
    
    def log_model_sklearn(self,
        sk_model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ):
        
        model_logger = mlflow.models.Model(run_id=self.run_id)
        model_logger.log(
            sk_model = sk_model,                                # User-defined
            artifact_path = artifact_path,                      # User-defined
            registered_model_name = registered_model_name,      # User-defined
            flavor = mlflow.sklearn,                            # Model-defined
            code_paths = None,                                  # the rest default values
            serialization_format = 'cloudpickle',
            signature = None,
            input_example = None,
            await_registration_for = 300,
            pip_requirements = None,
            extra_pip_requirements = None,
            pyfunc_predict_fn = "predict",
            metadata = None,
        )
    
    @staticmethod
    def _list_experiments(
        client: MlflowClient,
    ) -> List[str]:

        experiments = client.list_experiments()
        experiment_names = [e.name for e in experiments]
        
        return experiment_names
        
    @staticmethod
    def _get_run_id(
        client: MlflowClient,
        experiment_name: str,
        run_name
    ) -> Union[str, None]:
        
        experiemnt_id = client.get_experiment_by_name(experiment_name).experiment_id
        df = mlflow.search_runs(
            experiment_ids=experiemnt_id, filter_string=f'tags."Run Name"="{run_name}"')

        if len(df) > 0:
            run_id = df.loc[0, "run_id"]
        else:
            run_id = None
            
        return run_id
        
    def __str__(self):
        string = f'experiment_name={self.experiment_name}, experiment_id={self.experiment_id}\n, \
            run_client={self.run_client}, run_name={self.run_name}, run_id={self.run_id}'
        return string