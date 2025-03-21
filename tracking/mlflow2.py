from typing import Dict, Optional, Any, Union

import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from mlflow.models import ModelSignature

from .utils import prefix_print

class MLFlow2ClientWrapper:
    def __init__(self):
        load_dotenv()
        os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow-internal.mynt.xyz"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "firstname.lastname"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "check email for password"
        uri = "https://mlflow-internal.mynt.xyz"

        self.client = MlflowClient(uri)

        # class attributes tracked
        self.experiment_name = None
        self.experiment_id = None
        self.run_name = None
        self.run_id = None

        # Do not edit
        self.flavor_lib_dict = {
            "lightgbm": mlflow.lightgbm,
            "xgboost": mlflow.xgboost,
            "catboost": mlflow.catboost,
            "sklearn": mlflow.sklearn
        }
    
    #############################################################################################################
    #
    #                                             Experiment Methods
    #
    #############################################################################################################

    @prefix_print("[EXPERIMENT]")
    def create_experiment(self,
        experiment_name: str
    ):
        experiments = mlflow.search_experiments()
        experiment_names = [experiment.name for experiment in experiments]

        if experiment_name in experiment_names:
            raise ValueError(f"Experiment '{experiment_name}' already exists. Please create a new experiment using a different name or use set_experiment().")
        else:
            print(f"Creating new experiment with '{experiment_name}'")
            experiment_id = self.client.create_experiment(experiment_name)

        self.experiment_id = experiment_id

    @prefix_print("[EXPERIMENT]")
    def set_experiment(self,
        experiment_name: str
    ):
        experiments = mlflow.search_experiments()
        experiment_names = [experiment.name for experiment in experiments]

        if experiment_name in experiment_names:
            print(f"Found experiment '{experiment_name}'. Setting the experiment to '{experiment_name}'")
            self.experiment_id = self.client.get_experiment_by_name(name=experiment_name).experiment_id
            self.experiment_name = experiment_name
        else:
            self.experiment_id = None
            self.experiment_name = None
            raise ValueError(f"Experiment '{experiment_name}' does not exist. Please create it with create_experiment()")

    #############################################################################################################
    #
    #                                               Run Methods
    #
    #############################################################################################################

    @prefix_print("[RUN]")
    def create_run(self,
        run_name: str,
        tags: Optional[Dict] = None
    ):
        assert self.experiment_id is not None, f"'experiment_id' or 'experiment_name' is not None. Please set the experiment first by running the 'set_experiment()' method"

        # Get the run names
        run_names = [run.info.run_name for run in self.client.search_runs(self.experiment_id)]
        if run_name in run_names:
            raise ValueError(f"A run with the name '{run_name}' in '{self.experiment_name}' already exists. Please choose a different run name or use set_run()")
        
        # Add Owner and Run Names to Tags
        tags = tags or {}
        if "Run Name" not in tags.keys():
            tags["Run Name"] = run_name
        try:
            owner_name = tags["Owner"] = os.uname()[1].split("-")[1]
        except IndexError:
            owner_name = tags["Owner"] = os.uname()[1]
        if "Owner" not in tags.keys():
            tags["Owner"] = owner_name
        tags["mlflow.runName"] = run_name

        # Crate Run Object
        print(f"Creating new run '{run_name}' in experiment '{self.experiment_name}'")
        self.run_object = self.client.create_run(
            experiment_id = self.experiment_id,
            tags = tags
        )

    @prefix_print("[RUN]")
    def set_run(self,
        run_name: str
    ):
        assert self.experiment_id is not None, f"'experiment_id' or 'experiment_name' is not None. Please set the experiment first by running the 'set_experiment()' method"
        
        # Get the run names and run_id and store them in a dictionary
        runs_dict = {run.info.run_name: run.info.run_id for run in self.client.search_runs(self.experiment_id)}

        if run_name not in runs_dict:
            self.run_id = None
            self.run_name = None
            raise ValueError(f"A run with the name '{run_name}' in '{self.experiment_name}' DOES NOT exist!")
        
        self.run_id = runs_dict[run_name]
        self.run_name = run_name
        self.run_object = self.get_run(run_name=run_name)
        
        print(f"Set the run to '{run_name}' in experiment '{self.experiment_name}'")

    @prefix_print("[RUN]")
    def get_run(self,
        run_name: str
    ):
        assert self.experiment_id is not None, f"'experiment_id' or 'experiment_name' is not None. Please set the experiment first by running the 'set_experiment()' method"

        # Get the run names and run_id and store them in a dictionary
        runs_dict = {run.info.run_name: run.info.run_id for run in self.client.search_runs(self.experiment_id)}

        # Thou shall not pass if DOES NOT exist
        if run_name not in runs_dict:
            raise ValueError(f"A run with the name '{run_name}' in '{self.experiment_name}' DOES NOT exist!")
        
        run_object = self.client.get_run(runs_dict[run_name])
        
        return run_object

    #############################################################################################################
    #
    #                                               Param Methods
    #
    #############################################################################################################

    def log_param(self,
        key: str,
        value: Any
    ):
        self.client.log_param(self.run_id, key=key, value=value)

    def log_params(self,
        params: Dict[str, Any]
    ):
        for key, value in params.items():
            self.log_param(key, value)

    def get_params(self
        
    ) -> Union[Dict[str, Any], None]:
        
        self.run_object = self.get_run(self.run_name)
        params = self.run_object.data.params
        
        return params

    #############################################################################################################
    #
    #                                               Metrics Methods
    #
    #############################################################################################################

    def log_metric(self,
        key: str,
        value: float
    ):
        self.client.log_metric(self.run_id, key=key, value=value)

    def log_metrics(self,
        metrics: Dict[str, float]
    ):
        for key, value in metrics.items():
            self.log_metric(key=key, value=value)

    def get_metrics(self
                    
    ) -> Union[Dict[str,float], None]:
        
        self.run_object = self.get_run(self.run_name)
        metrics = self.run_object.data.metrics

        return metrics

    #############################################################################################################
    #
    #                                               Tags Methods
    #
    #############################################################################################################

    def set_tag (self,
        key: str,
        value: float
    ):
        self.client.set_tag(self.run_id, key=key, value=value)

    def set_tags(self,
        tags: Dict[str, str]
    ):
        for key, value in tags.items():
            self.set_tag(key=key, value=value)

    def get_tags(self
                    
    ) -> Union[Dict[str,str], None]:
        
        self.run_object = self.get_run(self.run_name)
        tags = self.run_object.data.tags

        return tags

    #############################################################################################################
    #
    #                                            Log Model Methods
    #
    #############################################################################################################

    @prefix_print("[LOG MODEL]")
    def log_model(self,
        model: Any,
        artifact_path: str,
        flavor: str,
        signature: Optional[ModelSignature] = None,
        registered_model_name: Optional[str] = None
    ):
        assert self.run_id is not None, "No active run MLFlow run found. Set the run via set_run() before logging a model"

        # Check if artifact path already exists  - Design choice to not overwrite existing paths
        artifact_paths = [file_info.path for file_info in self.client.list_artifacts(self.run_id)]
        if artifact_path in artifact_paths:
            raise ValueError(f"The artifact_path '{artifact_path}' already exists in '{self.run_name}'. Please provide a new artifact_path.")

        # Check flavor
        if flavor not in self.flavor_lib_dict:
            raise ValueError(f"The flavor '{flavor}' is not yet supported. Please ask the MLE to support your carry.")

        with mlflow.start_run(run_id=self.run_id):
            flavor_lib = self.flavor_lib_dict[flavor]
            
            # Register the model or not
            if registered_model_name is not None:
                flavor_lib.log_model(model, artifact_path, signature=signature, registered_model_name=registered_model_name)
                print(f"Logged a '{flavor}' model in '{artifact_path}' for run '{self.run_name}'")
            else:
                flavor_lib.log_model(model, artifact_path, signature=signature)
                print(f"Logged a '{flavor}' model in '{artifact_path}' for run '{self.run_name}'. Registed as '{registered_model_name}'")


    #############################################################################################################
    #
    #                                        Model Registration Methods
    #
    #############################################################################################################

    @prefix_print("[REG MODEL]")
    def register_model(self,
        artifact_path: str,
        registered_model_name: str
    ):
        assert self.run_id is not None, "No active run MLFlow run found. Set the run via set_run() before logging a model"

        # Check if artifact path already exists 
        artifact_paths = [file_info.path for file_info in self.client.list_artifacts(self.run_id)]
        if artifact_path not in artifact_paths:
            raise ValueError(f"The artifact_path '{artifact_path}' does not exist '{self.run_name}'. Please provide an existing artifact_path.")
        
        model_uri = f"runs:/{self.run_id}/{artifact_path}"
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)

        print(f"Registered '{registered_model_name}' from '{artifact_path}' of run '{self.run_name}'")

    #############################################################################################################
    #
    #                                          Model Loading Methods
    #
    #############################################################################################################    

    @prefix_print("[LOAD MODEL]")
    def load_model_from_registry(self,
        registered_model_name: str,
        model_version: str,
        flavor: Optional[str] = None
    ):
        model_uri = f"models:/{registered_model_name}/{model_version}"

        # Check if the registered model name exists raise error if not
        registered_models = [model_object.name for model_object in self.client.search_registered_models()]
        if registered_model_name not in registered_models:
            raise ValueError(f"Registered model '{registered_model_name}' does not exist in MLflow Model Registry.")

        # Check if the model version of the registered model name exists raise error if not
        try:
            self.client.get_model_version(registered_model_name, model_version)
        except Exception as e:
            raise ValueError(f"Model version '{model_version}' for '{registered_model_name}' does not exist.") from e

        # Default to pyfunc
        if flavor is not None:
            if flavor not in self.flavor_lib_dict:
                raise ValueError(f"The flavor '{flavor}' is not yet supported. Please ask the MLE to support your carry. Or consider passing None to use 'pyfunc' flavor")

            flavor_lib = self.flavor_lib_dict[flavor]
        else:
            flavor_lib = mlflow.pyfunc

        model = flavor_lib.load_model(model_uri=model_uri)
        
        print(f"Loaded model '{registered_model_name}' version '{model_version}'")

        return model
    
    @prefix_print("[LOAD MODEL]")
    def load_model_from_run_artifacts(self,
        artifact_path: str
    ):
        assert self.run_id is not None, "No active run MLFlow run found. Set the run via set_run() before logging a model"

        # Check if artifact path already exists - Design choice to not overwrite existing paths
        artifact_paths = [file_info.path for file_info in self.client.list_artifacts(self.run_id)]
        if artifact_path not in artifact_paths:
            raise ValueError(f"The artifact_path '{artifact_path}' does not exist '{self.run_name}'. Please provide an existing artifact_path.")

        model_uri = f"runs:/{self.run_id}/{artifact_path}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        print(f"Loaded model from '{artifact_path}' of run '{self.run_name}'")

        return model
    
    #############################################################################################################
    #
    #                                                  misc
    #
    #############################################################################################################

    def __str__(self):
        string = f"experiment_name={self.experiment_name}, experiment_id={self.experiment_id},\n \
        run_object={self.run_object}, run_name={self.run_name}, run_id={self.run_id}"
        return string