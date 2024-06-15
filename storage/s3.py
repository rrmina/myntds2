import os
import boto3
import pickle
from typing import Any

from io import BytesIO
from dotenv import load_dotenv
import pandas as pd

class SimpleS3Client:
    def __init__(self,
        bucket: str = 'myntanalytics'
    ):
        
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(bucket)
    
    def write_pickle(self,
        key: str,
        data: Any
    ):
        with BytesIO() as f:
            pickle.dump(data, f)
            payload = f.getvalue()
            
        bucket_obj = self.bucket.Object(key=key)
        bucket_obj.put(Body=payload)
        
    def read_pickle(self,
        key: str
    ) -> Any:
        
        bucket_obj = self.bucket.Object(key=key)
        get_object_result = bucket_obj.get()['Body']
        result = get_object_result.read()
        bytes_object = BytesIO(result)
        data = pickle.load(bytes_object)
        
        return data
    
    def write_feather(self,
        key: str,
        df: pd.DataFrame
    ):
        
        with BytesIO() as f:
            df.to_feather(f)
            payload = f.getvalue()
        
        bucket_obj = self.bucket.Object(key=key)
        bucket_obj.put(Body=payload)
    
    def read_feather(self,
        key: str
    ) -> pd.DataFrame:
        
        bucket_obj = self.bucket.Object(key=key)
        get_object_result = bucket_obj.get()['Body']
        result = get_object_result.read()
        bytes_object = BytesIO(result)
        df = pd.read_feather(bytes_object)
        
        return df
        
    def write_parquet(self,
        key: str,
        df: pd.DataFrame
    ):
        
        with BytesIO() as f:
            df.to_parquet(f, index=False)
            payload = f.getvalue()
            
        bucket_obj = self.bucket.Object(key=key)
        bucket_obj.put(Body=payload)
        
    def read_parquet(self,
        key: str
    ) -> pd.DataFrame:
        
        bucket_obj = self.bucket.Object(key=key)
        get_object_result = bucket_obj.get()['Body']
        result = get_object_result.read()
        bytes_object = BytesIO(result)
        df = pd.read_parquet(bytes_object)
        
        return df