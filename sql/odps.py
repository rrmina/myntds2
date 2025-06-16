import os
import odps
from odps.df import DataFrame as ODPSDataFrame
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

from .utils import random_alphanumeric_string

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
        self.project = os.getenv(project_key, default='gcash_ml_sandbox')
        
        # Initialize ODPS
        self.o = odps.ODPS(access_id=access_id, secret_access_key=secret_access_key, project=self.project, endpoint=endpoint)

    #############################################################################################################
    #
    #                                        Run and Run-Tracking Methods
    #
    #############################################################################################################

    def run_sql(self,
        query: str = None
    ) -> odps.models.Instance:
        
        return self.o.run_sql(query, hints={"odps.sql.submit.mode" : "script"})

    def run_sql_template(self,
        query_template: str = None,
        args_dict: Dict = None
    ) -> odps.models.Instance:
        
        for key in args_dict.keys():
            query_template = query_template.replace('${'+key+'}', args_dict[key])

        return self.run_sql(query_template)

    def parallel_run_sql_template(self,
        query_template: str = None,
        partition_values_dict: Dict[str, List[str]] = None,
        max_workers: int = None
    ) -> List[odps.models.Instance]:
        
        return self._parallel_executor_template(
            fn_executor=self.run_sql_template,
            query_template=query_template,
            partition_values_dict=partition_values_dict,
            max_workers=max_workers
        )

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

        df = pd.DataFrame(values, columns=list(record._name_indexes.keys()))
                          
        return df
    
    def execute_sql_template(self,
        query_template: str = None,
        args_dict: Dict = None
    ) -> odps.models.Instance:
        
        for key in args_dict.keys():
            query_template = query_template.replace('${'+key+'}', args_dict[key])

        return self.execute_sql(query_template)
    
    def mapreduce_execute_sql_to_df(self,
        query: str = None,              # must be a select statement ONLY
        num_partitions: int = 32,
        temp_table_name: str = None
    ) -> pd.DataFrame:
        
        if temp_table_name is None:
            temp_table_name = random_alphanumeric_string(length=10)

        partition_query = \
        f'''
            DROP TABLE IF EXISTS {self.project}.{temp_table_name};
            CREATE TABLE IF NOT EXISTS {self.project}.{temp_table_name} AS (
                SELECT
                    *
                    , NTILE({num_partitions}) OVER () AS partition_num
                FROM (
                    {query}
                )
            )
        '''

        # a partition blocking function
        print(f'Dividing query result into {num_partitions} partitions stored in {self.project}.{temp_table_name}')
        _ = self.execute_sql(partition_query)   

        # map definition
        def fetch_partition(i):
            df = self.execute_sql_to_df(
                f'''
                SELECT t.`(partition_num)?+.+`
                FROM {self.project}.{temp_table_name} t
                WHERE partition_num = '{i+1}'
                '''
            )

            return df
        
        # map proper
        result_list = [None] * num_partitions  # preallocate to maintain order
        with ThreadPoolExecutor(max_workers=num_partitions) as executor:

            # map
            future_to_index = {executor.submit(fetch_partition, i): i for i in range(num_partitions)}

            # track status
            for future in tqdm(as_completed(future_to_index), total=num_partitions, desc=f'Fetching partitions from {self.project}.{temp_table_name}: '):
                i = future_to_index[future]
                try:
                    result_list[i] = future.result()
                except Exception as e:
                    print(f"Partition {i + 1} failed: {e}")

        # reduce proper
        concatenated_df = pd.concat(result_list, ignore_index=True)

        # delete temp partition table
        print(f'Dropping temp partition table: {self.project}.{temp_table_name}')
        self.execute_sql(
            f'''
            DROP TABLE IF EXISTS {self.project}.{temp_table_name}; 
            '''
        )

        return concatenated_df

    def execute_sql_template_to_df(self,
        query_template: str = None,
        args_dict: Dict = None
    ) -> pd.DataFrame:
        
        sql_instance = self.execute_sql_template(query_template, args_dict)

        with sql_instance.open_reader(tunnel=True, limit=False) as reader:
            df = reader.to_pandas()
            
        return df

    def parallel_execute_sql_template(self,
        query_template: str = None,
        partition_values_dict: Dict[str, List[str]] = None,
        max_workers: int = None
    ) -> List[odps.models.Instance]:
        
        return self._parallel_executor_template(
            fn_executor=self.execute_sql_template,
            query_template=query_template,
            partition_values_dict=partition_values_dict,
            max_workers=max_workers
        )

    #############################################################################################################
    #
    #                                             Parallel Methods
    #
    #############################################################################################################

    def _parallel_executor_template(self,
        fn_executor: Callable,
        query_template: str = None,
        partition_values_dict: Dict[str, List[str]] = None,
        max_workers: int = None
    ) -> List[odps.models.Instance]:
        
        assert partition_values_dict, "Must provide at least one partition key with values"
        
        partition_lengths = [len(v) for v in partition_values_dict.values()]
        assert partition_lengths, "Partition lists cannot be empty"
        assert all(x == partition_lengths[0] for x in partition_lengths), "All partition lists must have the same length"

        num_partitions = partition_lengths[0]
        arg_dicts = [{k: v[i] for k,v in partition_values_dict.items()} for i in range(num_partitions)]

        def execute_query(
            query_template: str,
            args_dict: Dict[str, str]
        ) -> odps.models.Instance:
            
            return fn_executor(
                query_template=query_template,
                args_dict=args_dict
            )
        
        partial_func = partial(execute_query, query_template)

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(partial_func, curr_dict) for curr_dict in arg_dicts]

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f'Error: {e}')

        return futures

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

        if isinstance(partition_names, str):
            partition_names = [partition_names] 

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

        if isinstance(partition_names, str):
            partition_names = [partition_names] 

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
