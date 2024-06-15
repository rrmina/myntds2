import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Param, Metric, RunTag
from mlflow.utils.file_utils import TempDir
from dotenv import load_dotenv
from typing import Optional, List, Union, Any, Dict

from .utils import get_current_time_millis

class MLflowClientWrapper:
    def __init__(self,
        prod: bool,
    ):
        
        load_dotenv()
        if prod:
            uri = os.getenv('MLFLOW_PROD_URI')
        else:
            uri = os.getenv('MLFLOW_DEV_URI')
            
        self.client = MlflowClient(uri)
        self.experiment_name = None
        self.experiment_id = None
        self.run_object = None
        self.run_name = None
        self.run_id = None
    
    def create_experiment(self,
        experiment_name: str
    ):
        
        experiment_names = self._list_experiments(client=self.client)
        
        if experiment_name in experiment_names:
            raise ValueError(f"Experiment '{experiment_name}' already exists. Please create a new experiment using a different name or use set_experiment().")
        else:
            print(f"Creating new experiment with '{experiment_name}'")
            self.client.create_experiment(name=experiment_name)
    
    def set_experiment(self,
        experiment_name: str
    ):
        
        experiment_names = self._list_experiments(client=self.client)
        
        if experiment_name in experiment_names:
            print(f"Experiment '{experiment_name}' already exists. Setting the experiment to '{experiment_name}'")
            self.experiment_id = self.client.get_experiment_by_name(name=experiment_name).experiment_id
            self.experiment_name = experiment_name
        else:
            raise ValueError(f"Experiment '{experiment_name}' does not exist. Please create it with create_experiment()")
            self.experiment_id = None
            self.experiment_name = None
    
    def create_run(self,
        run_name: str,
        tags: Optional[Dict] = None
    ):
        
        assert self.experiment_id is not None, f"experiment_id or experiment_name is None. Finish 'set_experiment()' first"
        
        # Design Choice - throws an error if run name exists
        run_id = self._get_run_id_within_experiment(
            client=self.client, experiment_name=self.experiment_name, run_name=run_name)
        if run_id is not None:
            raise ValueError(f"A run with the name '{run_name}' in '{self.experiment_name}' already exists. Please choose a different run name or use set_run()")
            
        # Add Owner and Run Name to Tags
        tags = tags or {}
        if "Run Name" not in tags.keys():
            tags['Run Name'] = run_name
            tags['mlflow.runName'] = run_name
        if "Owner" not in tags.keys():
            tags['Owner'] = os.uname()[1].split("-")[1]
            
        # Create Run Object
        self.run_object = self.client.create_run(
            experiment_id = self.experiment_id,
            tags = tags
        )
    
    def set_run(self,
        run_name: str
    ):
        assert self.experiment_id is not None, f"experiment_id or experiment_name is None. Finish 'set_experiment()' first"
        
        run_id = self._get_run_id_within_experiment(
            client=self.client, experiment_name=self.experiment_name, run_name=run_name)
        
        # Design Choice - throws an error if run name DOES NOT exist
        if run_id is None:
            self.run_id = None
            self.run_name = None
            raise ValueError(f"A run with the name '{run_name}' in '{self.experiment_name}' DOES NOT exist!")
            
        self.run_id = run_id
        self.run_name = run_name
        self.run_object = self.get_run(run_name=run_name)
        
        print(f"Set the run to '{run_name}' in experiment '{self.experiment_name}'")
        
    def get_run(self,
        run_name: str
    ) -> Run:
        
        assert self.experiment_id is not None, f"experiment_id or experiment_name is None. Finish 'set_experiment()' first"
        
        run_id = self._get_run_id_within_experiment(
            client=self.client, experiment_name=self.experiment_name, run_name=run_name)
        
        # Design Choice - throws an error if run name DOES NOT exist
        if run_id is None:
            raise ValueError(f"A run with the name '{run_name}' in '{self.experiment_name}' DOES NOT exist!")
            
        run_object = self.client.get_run(run_id)
        
        return run_object
    
    def log_param(self,
        key: str,
        value: Any
    ):
        
        self.client.log_param(
            self.run_id, key=key, value=value)
    
    def log_params(self,
        params: Dict[str, Any]
    ):
        
        params_arr = [Param(key, str(value)) for key, value in params.items()]
        self.client.log_batch(
            run_id = self.run_id,
            metrics = [],
            params = params_arr,
            tags = []
        )

    def get_params(self
        
    ) -> Union[Dict, None]:
        
        params = self.run_object.data.params
        
        return params
        
    def log_metric(self,
        key: str,
        value: float
    ):
        
        self.client.log_metric(
            self.run_id, key=key, value=value)

    def log_metrics(self,
        metrics: Dict[str, float]
    ):
        
        timestamp = get_current_time_millis()
        metrics_arr = [Metric(key, value, timestamp, 0) for key, value in metrics.items()]
        self.client.log_batch(
            run_id = self.run_id,
            metrics = metrics_arr,
            params = [],
            tags = []
        )
    
    def get_metrics(self
        
    ) -> Union[Dict, None]:
        
        metrics = self.run_object.data.metrics
        
        return metrics
    
    def log_tag(self,
        key: str,
        value: Any
    ):
        
        self.client.set_tag(
            self.run_id, key=key, value=value)

    def log_tags(self,
        tags: Dict[str, Any]
    ):
        tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
        self.client.log_batch(
            run_id = self.run_id,
            metrics = [],
            params = [],
            tags = tags_arr
        )
    
    def get_tags(self

    ) -> Union[Dict, None]:
        
        tags = self.run_object.data.tags
        
        return tags
    
    def log_model_sklearn(self,
        sk_model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature: Optional[str] = None
    ):
        
        # Check if artifact path exists
        model_versions = self.client.search_model_versions(f"run_id='{self.run_id}'")
        artifact_paths = ["/".join(model_version.source.split('/')[8:]) for model_version in model_versions]
        if artifact_path in artifact_paths:
            raise ValueError(f"Artifact path '{artifact_path}' already exists. Please use a different artifact path.")
        
        return self._log_model_mlflow(
            sk_model = sk_model,                                # User-defined
            artifact_path = artifact_path,                      # User-defined
            registered_model_name = registered_model_name,      # User-defined
            run_id = self.run_id,                               # User-defined surgery
            mlflow_client = self.client,                        # User-defined surgery
            flavor = mlflow.sklearn,                            # Model-defined
            serialization_format = 'cloudpickle',               # the rest default values
            signature = signature,
            input_example = None,
            await_registration_for = 300,
            pip_requirements = None,
            extra_pip_requirements = None
        )
        
    def get_model_sklearn(self,
        registered_model_name: str,
        version: int
    ) -> Any:
        
        # Get model versions 
        model_versions = self.client.search_model_versions(f"run_id='{self.run_id}'")
        model_name_version_tuples = [(model_version.name, model_version.version) for model_version in model_versions]
        
        if (registered_model_name, str(version)) not in model_name_version_tuples:
            raise ValueError(f"Model name '{registered_model_name}' version '{version}' in run '{self.run_name}' does not exist. Only these (name, version) tuples exist: {model_name_version_tuples}")

        # Construct the model URI and load
        model_uri = f"models:/{registered_model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model
    
    def log_model_pytorch(self,
        pytorch_model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature: Optional[str] = None
    ) -> Any:
        
        # Check if artifact path exists
        model_versions = self.client.search_model_versions(f"run_id='{self.run_id}'")
        artifact_paths = ["/".join(model_version.source.split('/')[8:]) for model_version in model_versions]
        if artifact_path in artifact_paths:
            raise ValueError(f"Artifact path '{artifact_path}' already exists. Please use a different artifact path.")
            
        return self._log_model_mlflow(
            pytorch_model = pytorch_model,                      # User-defined
            artifact_path = artifact_path,                      # User-defined
            registered_model_name = registered_model_name,      # User-defined
            run_id = self.run_id,                               # User-defined surgery
            mlflow_client = self.client,                        # User-defined surgery
            flavor = mlflow.pytorch,                            # Model-defined
            conda_env=None,                                     # the rest default values
            code_paths=None,
            pickle_module=None,
            signature=signature,
            input_example=None,
            await_registration_for=300,
            requirements_file=None,
            extra_files=None,
            pip_requirements=None,
            extra_pip_requirements=None
        )
    
    def get_model_pytorch(self,
        registered_model_name: str,
        version: int
    ) -> Any:

        # Get model versions 
        model_versions = self.client.search_model_versions(f"run_id='{self.run_id}'")
        model_name_version_tuples = [(model_version.name, model_version.version) for model_version in model_versions]
        
        if (registered_model_name, str(version)) not in model_name_version_tuples:
            raise ValueError(f"Model name '{registered_model_name}' version '{version}' in run '{self.run_name}' does not exist. Only these (name, version) tuples exist: {model_name_version_tuples}")

        # Construct the model URI and load
        model_uri = f"models:/{registered_model_name}/{version}"
        model = mlflow.pytorch.load_model(model_uri)
        
        return model
        
    # We want to move away from this by using the latest version instead of 1.24
    @staticmethod
    def _log_model_mlflow(
        artifact_path,
        flavor,
        run_id,
        mlflow_client,
        registered_model_name=None,
        await_registration_for=300,
        **kwargs,
    ):
        with TempDir() as tmp:
            local_path = tmp.path("model")
            
            # Instantiate mlflow Model Class
            mlflow_model = mlflow.models.Model(artifact_path=artifact_path, run_id=run_id)
            
            # Save model according to flavor to a local path
            flavor.save_model(path=local_path, mlflow_model=mlflow_model, **kwargs)
            
            # Transfer the artifacts from local path to artifact path
            mlflow_client.log_artifacts(run_id, local_path, artifact_path)
            
            # Attach mlflow model to run 
            mlflow_client._record_logged_model(run_id, mlflow_model)
            
            # Register model to model site and for versioning
            if registered_model_name is not None:
                mlflow.register_model(
                    "runs:/%s/%s" % (run_id, artifact_path),
                    registered_model_name,
                    await_registration_for=await_registration_for,
                )

        return mlflow_model.get_model_info()
    
    @staticmethod
    def _list_experiments(
        client: MlflowClient,
    ) -> List[str]:

        experiments = client.list_experiments()
        experiment_names = [e.name for e in experiments]
        
        return experiment_names
        
    @staticmethod
    def _get_run_id_within_experiment(
        client: MlflowClient,
        experiment_name: str,
        run_name: str
    ) -> Union[str, None]:
        
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        df = mlflow.search_runs(
            experiment_ids=experiment_id, filter_string=f'tags."mlflow.runName"="{run_name}"')

        if len(df) > 0:
            run_id = df.loc[0, "run_id"]
        else:
            run_id = None
            
        return run_id
        
    @staticmethod
    def _set_run_name(
        client: MlflowClient, 
        run_id: str, 
        run_name: str
    ):
        
        run = mlflow.get_run(run_id)
        client.set_tag(run_id, "mlflow.runName", run_name)
    
    def __str__(self):
        string = f'experiment_name={self.experiment_name}, experiment_id={self.experiment_id},\n \
        run_object={self.run_object}, run_name={self.run_name}, run_id={self.run_id}'
        return string