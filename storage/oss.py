import os
import oss2
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from ..sql.odps import SimpleODPSClient
from typing import List
class SimpleOSSClient:
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
        o = SimpleODPSClient()

        self.bucket = bucket
        self.o = o.o

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
    
    def list_parquet_paths_from_odps(self,
        project: str = 'mynt_ds_dev',
        table_name: str = None,
        partition: str = None
    ) -> List[str]:
        
        t = self.o.get_table(name=table_name, project=project)
        parts = [key for key in t.location.split("/")[4:] if len(key) > 0]
        parts.append(partition)
        prefix = "/".join(parts)

        paths = [
            obj.key for obj in oss2.ObjectIteratorV2(self.bucket, prefix=prefix)
            if not obj.key.endswith(".meta")
        ]

        return paths

    def read_parquet_from_odps(self,
        project: str = 'mynt_ds_dev',
        table_name: str = None,
        partition: str = None
    ) -> pd.DataFrame:

        parquet_list = self.list_parquet_paths_from_odps(
            table_name = table_name,
            partition = partition,
            project = project,
        )
        
        if len(parquet_list) == 0:
            print("Parquet Empty. Exiting now.")
            return 0
        
        for i in range(len(parquet_list)):
            parquet_path = parquet_list[i]
            temp = self.read_parquet(parquet_path)
            if i == 0:
                df = temp
            else:
                df = pd.concat([df, temp])

        return df