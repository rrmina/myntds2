import os
import odps
from odps.df import DataFrame as ODPSDataFrame
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple

class SimpleODPSClient:
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
        self.project = os.getenv(project_key, default='mynt_ds_dev')
        
        # Initialize ODPS
        self.o = odps.ODPS(access_id=access_id, secret_access_key=secret_access_key, project=self.project, endpoint=endpoint)

    #############################################################################################################
    #
    #                                              Execution Methods
    #
    #############################################################################################################

    def execute_sql(self,
        query: str = None
    ) -> odps.models.Instance:
        
        return self.o.execute_sql(query, hints={"odps.sql.submit.mode" : "script"})
    
    def execute_sql_to_df(self,
        query: str = None
    ) -> pd.DataFrame:
        
        sql_instance = self.execute_sql(query)
    
        values = []
        with sql_instance.open_reader() as reader:
            for record in reader:
                values.append(record.values)

        # To Fix formatting of SHOW PARTITIONS

        df = pd.DataFrame(values, columns=list(record._name_indexes.keys()))
                          
        return df
    
    #############################################################################################################
    #
    #                                               DDL Methods
    #
    #############################################################################################################

    def create_ddl_string(self,
        column_list: List[Tuple[str]],
        table_name: str,
        partition_names: List[str] = None,
        force_drop = False,
        external=False
    ) -> str:
        
        # Header
        create_table_string = f"CREATE {'EXTERNAL ' if external else ''}TABLE IF NOT EXISTS {self.project}.{table_name} ("
        if force_drop:
            create_table_string = f"DROP TABLE IF EXISTS {self.project}.{table_name};\n" + create_table_string

        # Column and Types
        max_width = max(len(table_column[0]) for table_column in column_list)
        columns_strings = []
        for i, table_column in enumerate(column_list):
            name, col_type = table_column
            if i == 0:
                columns_strings.append(f'\t{name.ljust(max_width+3)} {col_type}')
            else:
                columns_strings.append(f'\n\t, {name.ljust(max_width+1)} {col_type}')

        columns_string = "".join(columns_strings)

        # Partition Footer
        partition_string = ''
        if partition_names:
            sub_partition_string = ', '.join([f'{x} STRING' for x in partition_names])      # Partition are default to STRING
            partition_string = f')\nPARTITIONED BY ({sub_partition_string})'

        # External Storage Footer
        if external:
            external_string = f'''STORED AS PARQUET\nLOCATION \'oss://oss-ap-southeast-1-internal.aliyuncs.com/mynt-aa/myntds/{table_name}/\' '''
        else:
            external_string = ''

        # Create the final DDL string
        ddl_string = f'''{create_table_string}\n{columns_string}\n{partition_string}\n{external_string};'''

        return ddl_string
    
    def create_ddl_string_from_df(self, 
        df: pd.DataFrame, 
        table_name: str, 
        partition_names: List[str] = None,
        force_drop = False,
        external = False
    ) -> str:

        # Pandas to MaxCompute Aliyun SQL Types - Can be extended
        sql_type_dict = {
            np.dtype('O'): 'STRING',
            np.dtype('int64'): 'BIGINT',
            np.dtype('float64'): 'DOUBLE'
        }

        # Get the Column list and type
        column_list = []
        for i, (col_name, col_type) in enumerate(df.dtypes.items()):
            sql_type = sql_type_dict.get(col_type, 'STRING')  # Default to STRING if type is unknown
            column_list.append((col_name, sql_type))

        ddl_string = self.create_ddl_string(column_list, table_name, partition_names, force_drop, external)

        return ddl_string

    def create_ddl_from_df(self,
        df: pd.DataFrame, 
        table_name: str, 
        partition_names: List[str] = None,
        force_drop = False,
        external = False
    ):
        
        ddl_string = self.create_ddl_string_from_df(df, table_name, partition_names, force_drop, external)
        result = self.execute_sql(ddl_string)

        return result

    def create_ddl_string_from_odps(self,
        reference_odps_table_name: str,
        table_name: str,
        partition_names: List[str] = None,
        force_drop = False,
        external = False
    ) -> str:
        
        table = self.o.get_table(reference_odps_table_name)
        schema = table.table_schema
        table_columns = schema.simple_columns

        column_list = []
        for column in table_columns:
            name = column.name
            sql_type = column.type.name
            if name in partition_names:
                continue

            column_list.append((name, sql_type.upper()))

        ddl_string = self.create_ddl_string(column_list, table_name, partition_names, force_drop, external)

        return ddl_string

    def create_ddl_from_odps(self,
        reference_odps_table_name: str,
        table_name: str,
        partition_names: List[str] = None,
        force_drop = False,
        external = False
    ):
        ddl_string = self.create_ddl_string_from_odps(reference_odps_table_name, table_name, partition_names, force_drop, external)
        result = self.execute_sql(ddl_string)

        return result

    #############################################################################################################
    #
    #                                             Persisting Methods
    #
    #############################################################################################################

    # TODO : Support multiple partitions
    def save_df(self,
        df: pd.DataFrame,
        table_name: str,
        partitions: str = None
    ):
        # create kwargs if with partition
        create_partition = True if partitions else False
        kwargs = {}
        if create_partition:
            kwargs['partition'] = partitions
        
        result = odps.DataFrame(df).persist(
            table_name,
            overwrite=True,
            drop_partition=False,
            create_partition=create_partition,
            odps=self.o,
            **kwargs
        )

        return result
