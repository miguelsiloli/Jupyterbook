
# 03: Pipeline Design

## Introduction

Our data pipeline wants to capture the stream of new items being added to Vinted's marketplace in real time. This was the initial idea but it's not possible to do it due to Vinted API request limits - having a streaming connection would quickly result in me getting timed out or banned.

This is what we are trying to do:

```{mermaid}
graph TD
    actor[User]

    subgraph System
        A[Check Prices]
        B[Retrieve Data]
        C[Select Subset of Articles]
        D[View Price Information]
        E[View Data Information]
        F[Group by catalog_id]

        actor --> A
        actor --> B
        
        A --> C
        A --> D
        
        B --> C
        B --> E
        
        C --> F
    end
```

Instead, our solution is to create a windowed pipeline which slices these streams into chunks and process these chunks into fixed time windows [(Google Research on Watermarking)](https://static.googleusercontent.com/media/research.google.com/pt-PT//pubs/archive/43864.pdf). We then use a timestamp watermark to keep track of when the articles were first fetched.

## Table of Contents

- [03: Pipeline Design](#03-pipeline-design)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Pipeline design fundamentals](#pipeline-design-fundamentals)
    - [Extraction pattern](#extraction-pattern)
      - [Time Range](#time-range)
    - [Source \& Sink](#source--sink)
      - [Source Replayability](#source-replayability)
      - [Sink Overwritability \& Idempotency](#sink-overwritability--idempotency)
    - [Windowing](#windowing)
  - [Designing Pipelines with Prefect](#designing-pipelines-with-prefect)
    - [Some background on prefect](#some-background-on-prefect)
    - [Prefect Pipeline](#prefect-pipeline)
  - [Example](#example)
    - [Catalog flow](#catalog-flow)
    - [Analysis of Catalog Flow](#analysis-of-catalog-flow)
      - [Single unit of work](#single-unit-of-work)
        - [Subflow block](#subflow-block)
      - [Wrapping up with a main flow](#wrapping-up-with-a-main-flow)
    - [DAGs](#dags)
      - [Catalog flow](#catalog-flow-1)
      - [Tracking flow](#tracking-flow)
      - [Images flow](#images-flow)
      - [Dependencies](#dependencies)
  - [Conclusions](#conclusions)


## Pipeline design fundamentals

<table border="0">
 <tr>
    <td style="vertical-align: top;"><b style="font-size:30px">Structural Level</b></td>
    <td style="vertical-align: top;"><b style="font-size:30px">Atomic Level</b></td>
 </tr>
 <tr>
    <td style="vertical-align: top;">
        <b>Data Sources and Destinations</b>: <br>
        Consider the format, volume, read/write frequency and frequency of data from each source and how it will be integrated into the pipeline.
        </ul>
    </td>
    <td style="vertical-align: top;">
        <b>Atomicity</b>:<br> 
        A function should only do one task.
    </td>
 </tr>
 <tr>
    <td style="vertical-align: top;">
        <b>Data Quality and Validation</b>:<br> 
         This may involve validating data against predefined rules, detecting and handling missing or erroneous values, and logging any data anomalies for further investigation.
    </td>
    <td style="vertical-align: top;">
        <b>Idempotency</b>: 
        <ul>
            <li>Running the same code multiple times with the same input should return the same output.</li>
            <li>The output should not be duplicated.</li>
        </ul>
    </td>
 </tr>
 <tr>
    <td style="vertical-align: top;">
        <b>Fault Tolerance</b>:<br> 
        Implement mechanisms such as retry logic, error logging, and checkpointing to recover from errors and resume processing without losing data or compromising data integrity.
    </td>
    <td style="vertical-align: top;">
        <b>Data Encapsulation</b>:<br> 
        Function encapsulation should be limited to the scope it refers to, no additional external data.
    </td>
 </tr>
 <tr>
    <td style="vertical-align: top;">
        <b>Other concerns</b>:<br> 
        Security, Infrastructure, Documentation, Monitoring and Logging, Data Compliance, Change Data Capture (CDC)
    </td>
    <td style="vertical-align: top;">
        <b>Functional Composition</b>: A higher order mechanism which allows to increase the level of abstraction and atomicity. Can be implemented in prefect via subflow calls.
        <ul>
            <li>Factoring/atomicity</li>
            <li>Code reuse</li>
            <li>Abstraction layers (f(g(x)))</li>
        </ul>
    </td>
 </tr>
</table>

### Extraction pattern

#### Time Range

    With a time ranged pull, the data pipeline only pulls the data corresponding to a specific time frame.

A run of the pipeline will pull data from an interval of time. In this case, If I run the pipeline every hour I expect the data to be fetched is 1h old max.

Example: Run catalogs flow hourly at 13h, I expect the data fetched is between 12-13h.

### Source & Sink

- **Source**: Systems which provide input(s) to your data pipeline.
- **Sink**: Systems where your data pipeline stores the processed data.

#### Source Replayability

    *What did the data look like n periods(min/hour/day/months/years) ago?(n can be any reasonable arbitrary number)*, 
    then it is replayable, else it is not. Replayability is critical for backfills [(Data Pipeline Patterns)](https://www.startdataengineering.com/post/design-patterns/#3-data-pipeline-patterns).

This data is a replayable source, because if we can easily do a backfill by increasing the parameters on the request. However, this may not be replayable if we want to keep a stateful track of the system state.


In the tracking pipeline, we are taking a snapshot at a specific time. If we miss one day, it's gone forever.

```{hint}
One can create a replayable source from a non-replayable source by dumping the incoming data into a raw/loading/base area. The raw/loading/base area now acts as a replayable source.
```

**Replayability**
[x] Catalog Pipeline
[x] Images Pipeline
[] Tracking Pipeline

#### Sink Overwritability & Idempotency

    Overwritability is crucial to prevent duplicates or partial records when a pipeline fails.


This is easily achieved by loading the data into an RDS using upsert.

### Windowing

|![Pipeline Schema](../assets/data_engineering/windowing.png)|
|:--:| 
| Windowing|

Windowing slices up a dataset into finite chunks for processing as a group. When dealing with unbounded data, windowing is required for some operations (to delineate finite boundaries in most forms of grouping: aggregation, outer joins, time-bounded operations, etc.), and unnecessary for others (filtering, mapping, inner joins, etc.).

## Designing Pipelines with Prefect

### Some background on prefect

::::{grid}
:gutter: 2

:::{grid-item-card} Tasks
- Tasks are the building blocks of Prefect workflows. Each task represents a **single unit of work**, such as querying a database, processing data, or sending an email.
- Tasks are defined as Python functions or classes annotated with @task decorators. These functions encapsulate the logic required to perform the task.
- Tasks can have inputs and outputs, allowing them to communicate with other tasks in the workflow.
:::

:::{grid-item-card} Flows
- Flows are directed acyclic graphs (DAGs) that represent the workflow logic in Prefect.
- Flows consist of interconnected tasks, where the output of one task serves as the input to another.
- Flows define the execution order of tasks and handle the orchestration of task execution.
:::
::::

### Prefect Pipeline

```{hint}
   **Subflows**
   
   - Subflows are reusable components that encapsulate a subset of tasks within a larger workflow.
   - Subflows allow you to modularize your workflow logic by grouping related tasks together and abstracting away implementation details.
   - Subflows can be nested within other subflows or included as part of larger flows, providing flexibility in workflow design.
```

Prefect allows you to define complex workflows as code, making it easy to orchestrate the execution of tasks, handle dependencies, and manage error handling and retries. Let's break down its core concepts:

```{dropdown} Cool out-of-the-box prefect utils
- **caching keys**
  > An optional callable that, given the task run context and call parameters, generates a string key. If the key matches a previous completed state, that state result will be restored instead of running the task again.
- **exponential backoff**
- **flow signals**: allow tasks to communicate with each other and trigger downstream tasks based on conditions or events
```



## Example

### Catalog flow

Suppose we want to check the prices and data for a specific subset of articles. These articles are grouped into taxonomy groups according to a hierarchical structure of *catalog_id's*.

```{mermaid}
graph TD
    A[fetch_data_from_vinted]
    B[item_ids, frequency, db_connection]
    C[vinted API]
    D[catalog_subflow]
    F[export_data_quality_artifacts]
    H[export_data_to_postgres]
    I[export_sample_to_postgres]

    B -->|input| A
    A -->|connects to| C
    A -->|calls| D
    D -->|calls| F
    D -->|calls| H
    D -->|calls| I
```

### Analysis of Catalog Flow

|![Local Image](../assets/data_engineering/catalog_flow.png)|
|:--:| 
| Flow sequential dependency graph. |

We can see here how a single flow is decomposed into lowers levels of abstraction until we reach a single unit of work (a task).

#### Single unit of work
**Load Block**

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

  - [x] **Atomicity**: 
  Loads data from API and assigns the catalog_id to dataframe (optimally should only load data from API, data transformations should be done in different blocks). However, the reason I did this is to have a cleaner code and prevent overloading other tasks with unnecessary variables.

  - [x] **Encapsulation**: 
  The load function has no side effects. It starts by accepting an existing connection via dependency injection (vinted obj).


  ```{important}
  Cursors should be passed outside of loading blocks; input variables should be limited to their use scope; transformation blocks should be apart from load/upload blocks.
  ```

##### Subflow block

|![Local Image](../assets/data_engineering/catalog_subflow.png)|
|:--:| 
|Tasks sequential dependency graph within a subflow.|

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

- [x] **Functional decomposition**: increasing the levels of abstraction by decomposing a single flow into sub flows. 

- [x] **Atomicity**: 
Transformer, loading, export functions are in different blocks.

- [x] **Encapsulation**: 
External connections are passed as argument to code blocks. Vinted and engine are kept as external objectes passed onto the functions.


#### Wrapping up with a main flow

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

### DAGs

| **Title**           | **Explanation**                                                                                                                                                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Catalog Flow**    | The catalog flow involves retrieving and processing product listings from the Vinted API and HTML sources. It ensures data is fetched in batch, integrated into the pipeline, and validated for quality and completeness.                 |
| **Tracking Flow**   | The tracking flow monitors the status and updates of catalog items. It includes handling updates to existing data, ensuring idempotency, and maintaining accurate and consistent tracking information without duplicating records.        |
| **Images Flow**     | The images flow manages the retrieval and processing of images associated with catalog items. This flow handles the download, storage, and linking of images to corresponding products, ensuring efficient and reliable image management. |


#### Catalog flow

```{mermaid}
flowchart LR
    subgraph catalog_subflow
        direction TB
        top1[load_data_from_api] --> bottom1[transform]
        bottom1[transform] --> bottom2[export_data_to_postgres]
    end

    B[Initialize Vinted and Engine] --> C[Loop Through Item Catalogs]
    C --> D{Item Catalogs Available?}
    D -- Yes --> E[Call catalog_subflow]
    D -- No --> G[End]
    E[Call catalog_subflow] --> C[Loop Through Item Catalogs]
```

#### Tracking flow

```{mermaid}
flowchart LR
    B[Initialize Vinted and Engine] --> C[load_data_from_postgres]
    C[load_data_from_postgres] --> D[load_balancer]
    D[load_balancer] --> E[End]

    subgraph load_balancer
        top1[loop through chunks] --> bottom1[tracking_subflow]
        bottom1[tracking_subflow] --> A1[fetch_sample_data]
        A1[fetch_sample_data] --> A2[transform_images]
        A1[fetch_sample_data] --> A3[transform_users]
        A1[fetch_sample_data] --> A6[transform_items]
        A3[transform_users] --> A4[export_users_to_postgres]
        A2[transform_images] --> A5[export_images_to_postgres]
        A6[transform_items] --> A7[export_images_to_postgres]
    end
```

#### Images flow

```{mermaid}
graph TD
    B[Initialize S3 Client] --> H[Upload Image to S3]
    D[Initialize Vinted and Engine] --> E[Loop image urls]
    E --> F[get_image_bytes]
    F --> H[Upload Image to S3]
    H[Upload Image to S3] --> E
```

#### Dependencies

```{mermaid}
graph TD
    A[Catalog Flow] --> H[(Postgres)]
    H[(Postgres)] --> B[Tracking Flow]
    B[Tracking Flow] --> H[(Postgres)]
    H[(Postgres)] --> D[Images Flow]
    D[Images Flow] --> H[(Postgres)]
```



## Conclusions

- Catalog flow is well structured according to the rules in place. It's robust, tasks are decoupled and independent and it's easier to troubleshoot once an issue arises. The code is clean, having proper descriptions and task names is very helpful.
- Subflows should be decoupled. They are decoupled from a functional perspective, since the successful execution of one flow is independent from another's, however if one subflow fails it triggers the fail of the whole flow and stops the pipeline.

- [x] **Atomicity**: every task has one and one purpose
- [x] **Encapsulation**: each task has restricted access to the data and system objects. Only gets what it needs to deliver the tasks. (*no side effects*)
- [x] **Functional decomposition**: divided pipelines into tasks, subflows and flows
- [x] **Fault tolerance**: exponential timeouts, retry logic and error logging. Employing environment variables for custom timeouts, chunking requests, set flexible throughtput
- [x] **Data Quality**: dumping data report artifacts during runs
- [x] **Time Range Pattern** with flexible range, using a environment variable for run periodicity
- [ ] **Documentation**:
- [ ] **Monitoring**: Monitoring doesnt exist
- [ ] **Data Validation**:


**Prefect limitations**

A lot of features are locked under prefect cloud, which sucks if you want to build an on premise project. This is important since a lot of out of the box integrations are only supported in the cloud version and its impossible (iirc) to create triggers, dynamic flags, conditional workflows with the on premise version.