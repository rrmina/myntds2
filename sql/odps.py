import os
import odps
from dotenv import load_dotenv

class SimpleODPS:
    def __init__(self, 
        access_id_key: str = 'ODPS_ID',
        secret_access_key_key: str = 'ODPS_SECRET',
        project_key: str = 'ODPS_PROJECT',
        endpoint: str = 'https://service.ap-southeast-1.maxcompute.aliyun.com/api',
    ):
        
        # Load Environment Variables
        load_dotenv()
        access_id = os.getenv(access_id_key)
        secret_access_key = os.getenv(secret_access_key_key)
        project = os.getenv(project_key, default='mynt_ds_dev')
        
        # Initialize ODPS
        self.o = odps.ODPS(access_id=access_id, secret_access_key=secret_access_key, project=project, endpoint=endpoint)

    def execute_sql(self,
        query: str = None
    ) -> odps.models.Instance:
        
        return self.o.execute_sql(query)
    
    def execute_sql_to_df(self,
        query: str = None
    ) -> odps.models.Instance:
        
        sql_instance = self.o.execute_sql(query)
    
        with sql_instance.open_reader(tunnel=True, limit=False) as reader:
            df = reader.to_pandas()
            
        return df
    
    def save_df(self,
        
    ):
        return 0