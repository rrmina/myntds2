import os
import oss2
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
    
class SimpleOSS:
    def __init__(self,
        access_id_key: str = 'ODPS_ID',
        secret_access_key_key: str = 'ODPS_SECRET',
        endpoint: str = 'https://oss-ap-southeast-1.aliyuncs.com',
        bucket_name: str = 'mynt-aa' 
    ):
        
        # Load Environment Variables
        load_dotenv()
        access_id = os.getenv(access_id_key)
        secret_access_key = os.getenv(secret_access_key_key)
        
        auth = oss2.Auth(access_key_id=access_id, access_key_secret=secret_access_key)
        bucket = oss2.Bucket(auth=auth, endpoint=endpoint, bucket_name=bucket_name)
        
        self.bucket = bucket
    
    def write_feather(self,
        key: str,
        df: pd.DataFrame
    ):
        
        with BytesIO() as f:
            df.to_feather(f)
            data = f.getvalue()
            
        self.bucket.put_object(key=key, data=data)
    
    def read_feather(self,
        key: str
    ) -> pd.DataFrame:
        
        get_object_result = self.bucket.get_object(key)
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
            
        self.bucket.put_object(key=key, data=data)
        
    def read_parquet(self,
        key: str
    ) -> pd.DataFrame:
        
        get_object_result = self.bucket.get_object(key)
        result = get_object_result.read()
        bytes_object = BytesIO(result)            
        df = pd.read_parquet(bytes_object)
        
        return df
    
