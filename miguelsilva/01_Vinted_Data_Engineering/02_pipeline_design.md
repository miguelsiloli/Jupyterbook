
# Pipeline Design

### Best practices

#### Structural Level

- **Data Sources and Destinations**: Consider the format, volume, and frequency of data from each source and how it will be integrated into the pipeline.

- **Slowly Changing Dimensions (SCD)**: This is a key concept in data warehousing and data engineering. In particular, when I dealt with sales data at Bosch, SCD provided not only a way to get data lineage, but it also prevented operating issues. For instance, when dealing with SAP orderbooks, order quantities, purchasing orders, purchasing requisitions, etc, buyers should be able to track price changing and to know which contracts are currently active.

What happens in ERP softwares is that contracts are usually MYC (multi year contracts) and so prices and delivery conditions are negotiated prior to their execution. Sometimes, the contracts are subject to change according to the landscape and strategies and SCD keeps this track record (this case was vality dates, Type 2 SCD).

- **Data Quality and Validation**: This may involve validating data against predefined rules, detecting and handling missing or erroneous values, and logging any data anomalies for further investigation.

- **Fault Tolerance**: Implement mechanisms such as retry logic, error logging, and checkpointing to recover from errors and resume processing without losing data or compromising data integrity.

- Other out of scope such as Security, Infrastructure Reliability, Documentation, Monitoring and Logging, Data Compliance, Change Data Capture (CDC)

#### Atomic Level

- **Atomicity**: a function should only do one task
- **Idempotency**: 
    - running the same code multiple times with the same input should return the same output
    - the output should not be duplicated
- **Data encapsulation**: function encapsulation should be limited to the scope it refers to, no additional external data.
- **Functional composition**: a higher order mechanism which allows to increase the level of abstraction and atomicity. Can be implemented in prefect via subflow calls.
    - factoring/atomicity
    - code reuse
    - + abstraction layers (f(g(x)))


## Implementation

### Some background on prefect

Prefect allows you to define complex workflows as code, making it easy to orchestrate the execution of tasks, handle dependencies, and manage error handling and retries. Let's break down its core concepts:

**Tasks (UoW)**
- Tasks are the building blocks of Prefect workflows. Each task represents a **single unit of work**, such as querying a database, processing data, or sending an email.
- Tasks are defined as Python functions or classes annotated with @task decorators. These functions encapsulate the logic required to perform the task.
- Tasks can have inputs and outputs, allowing them to communicate with other tasks in the workflow.

**Flows (DAGS)**
- Flows are directed acyclic graphs (DAGs) that represent the workflow logic in Prefect.
- Flows consist of interconnected tasks, where the output of one task serves as the input to another.
- Flows define the execution order of tasks and handle the orchestration of task execution.

**Subflows**
- Subflows are reusable components that encapsulate a subset of tasks within a larger workflow.
- Subflows allow you to modularize your workflow logic by grouping related tasks together and abstracting away implementation details.
- Subflows can be nested within other subflows or included as part of larger flows, providing flexibility in workflow design.

#### Cool out-of-the-box prefect utils

- caching keys
  > An optional callable that, given the task run context and call parameters, generates a string key. If the key matches a previous completed state, that state result will be restored instead of running the task again.
- exponential backoff
- flow signals: allow tasks to communicate with each other and trigger downstream tasks based on conditions or events

### Catalogs flow

![Local Image](../assets/data_engineering/catalog_flow.png)
Flow sequential dependency graph.

We can see here how a single flow is decomposed into lowers levels of abstraction until we reach a single unit of work (a task).

#### Single unit of work: Load block

``` python
@task(name="Load from api", 
      log_prints= True,
      retries=3, 
      retry_delay_seconds=exponential_backoff(backoff_factor=6),
      retry_jitter_factor=2)
def load_data_from_api(vinted, nbrRows: int, batch_size: int, item: str) -> pd.DataFrame:
    """
    Loads data from the Vinted API based on specified parameters.

    Args:
        nbrRows (int): Number of rows to fetch from the API.
        batch_size (int): Batch size for API requests.
        item (str): Item include in the API request.

    Returns:
        pd.DataFrame: DataFrame containing data fetched from the Vinted API.
    """
    df = vinted.items.search_catalog(url = f"https://www.vinted.pt/catalog/items?catalog_ids[]={item}&order=newest_first")
    # cant process latin characters
    df["catalog_id"] = item
    
    return (df)
```

- **Atomicity**: Loads data from API and assigns the catalog_id to dataframe (optimally should only load data from API, data transformations should be done in different blocks). However, the reason I did this is to have a cleaner code and prevent overloading other tasks with unnecessary variables.

- **Encapsulation**: the load function has no side effects. It starts by accepting an existing connection via dependency injection (vinted obj).

**Takeaways**: cursors should be passed outside of loading blocks; input variables should be limited to their use scope; transformation blocks should be apart from load/upload blocks.

##### Subflow block

![Local Image](../assets/data_engineering/catalog_subflow.png)
Tasks sequential dependency graph within a subflow.

``` python
@flow(name="Subflow for catalog.", 
      flow_run_name= "Subflow for catalog {item}",
      log_prints= True)
def catalog_subflow(item, nbrRows, batch_size, vinted, engine, sample_frac):
    df = load_data_from_api(vinted = vinted,
                            nbrRows = nbrRows,
                            batch_size = batch_size,
                            item = item)
    df = transform(data = df)
    df = parse_size_title(data = df)
    export_data_to_postgres(data = df, 
                            engine = engine)     # upload first to products due to FK referencing
    export_sample_to_postgres(df, 
                              sample_frac= sample_frac,
                              engine = engine)
    return

```

- **Functional decomposition**: increasing the levels of abstraction by decomposing a single flow into sub flows. 
    - Before: main_flow -> task1 -> task2 -> etc
    - After: main_flow -> subflow1(task1, task2, etc) -> subflow2(task1, task2, etc)

- **Atomicity**: Transformer, loading, export functions are in different blocks.

- **Encapsulation**: external connections are passed as argument to code blocks.

#### Flow block

``` python
@flow(name= "Fetch from vinted", 
      log_prints= True,
      description= """
      Main flow: 
      start node: fetch vinted/items endpoint 
      -> simple preprocessing 
      -> dumps into postgres staging table""")
def fetch_data_from_vinted(sample_frac = 0.01, 
                           item_ids = [], 
                           batch_size = 500, 
                           nbrRows = 1000):
    """
    Fetches data from the vinted/items endpoint, preprocesses it, and exports it to a PostgreSQL staging table.

    Parameters:
    - sample_frac (float): Fraction of data to sample.
    - item_ids (list): List of item IDs to fetch data for.
    - batch_size (int): Size of each batch to fetch.
    - nbrRows (int): Number of rows to fetch.

    Returns:
    None
    """
    vinted = Vinted()
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    for __item in item_ids:
        catalog_subflow(item = __item, 
                        nbrRows= nbrRows,
                        batch_size= batch_size,
                        vinted = vinted, 
                        engine= engine, 
                        sample_frac= sample_frac)
        time.sleep(60)

    return
```

## Conclusions

- Catalog flow is well structured according to the rules in place. It's robust, tasks are decoupled and independent and it's easier to troubleshoot once an issue arises. The code is clean, having proper descriptions and task names is very helpful.
- Subflows should be decoupled. They are decoupled from a functional perspective, since the successful execution of one flow is independent from another's, however if one subflow fails it triggers the fail of the whole flow and stops the pipeline.