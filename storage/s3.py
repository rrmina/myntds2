import os
import boto3
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

class SimpleS3:
    def __init__(self,
        bucket: str = 'myntanalytics'
    ):
        
        s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(bucket)
    
    def write_feather(self,
        key: str,
        df: pd.DataFrame
    ):
        
        with BytesIO() as f:
            df.to_feather(f)
            data = f.getvalue()
        
        bucket_obj = self.bucket.Object(key=key)
        bucket_obj.put(Body=data)
    
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
            data = f.getvalue()
            
        bucket_obj = self.bucket.Object(key=key)
        bucket_obj.put(Body=data)
        
    def read_parquet(self,
        key: str
    ) -> pd.DataFrame:
        
        bucket_obj = self.bucket.Object(key=key)
        get_object_result = bucket_obj.get()['Body']
        result = get_object_result.read()
        bytes_object = BytesIO(result)
        df = pd.read_parquet(bytes_object)
        
        return df