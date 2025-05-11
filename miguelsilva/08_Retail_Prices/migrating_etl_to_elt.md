# Migrating ETL to ELT pipeline


-----------------------------------------------------------
Problem: Migrate a resource constrainted pipeline to Google Cloud. The current pipeline is hosted in BlackBlaze for blob storage and Supabase for SQL database (postgres). The bottleneck is currently the the SQL database and the BlackBlaze 1GB day networking.

Current design:
Scraping (GitHub Actions): Generates raw data as numerous small CSVs, logically grouped by category.

ETL Bottleneck (GitHub Actions): A Python script transforms these CSVs into structured dataframes and attempts to load them into Supabase Postgres. This critical step is chained to the scraping job.
Serving Layer (REST API): Connects directly to the constrained Supabase DB.

Issues:
The Egress Chokehold: BlackBlaze's 1GB daily limit isn't just a constraint; it's a hard stop, often hit mid-process during essential backfills or large updates.

Database Dead End: The 500MB Supabase limit means we're constantly managing data size instead of focusing on value. Scaling isn't an option here.

I/O & Network Thrashing: Processing hundreds or thousands of tiny CSVs hammers both storage I/O and network bandwidth unnecessarily.

ETL Fragility & The Atomicity Trap: The "all-or-nothing" nature of the current ETL job means a single failure in processing one category's CSVs often halts the entire batch, leading to data gaps, manual reruns, and operational headaches. This tight coupling and pre-load transformation is brittle.

Pipeline Observability: it's awful not to use a traditional data engineering tool such as dagster, airflow, prefect, etc.

Fix:
- Solve the Egress/Networking limits: Post process raw .csv into a compiled parquet file before uploading to file storage (ensures backward compatibility, reduces IO network operations)
- Postgres + Dbt: Move ETL job to SQL with ELT approach
- Get rid of tight coupling jobs: Decouple jobs by implementing partial (differential lods) by ingestion date.


## Pipeline ERD

```mermaid
graph TD
    %% Define subgraphs for each schema
    subgraph SOURCE["SOURCE SCHEMA (CSV FILES)"]
        SOURCE_GENERIC["SOURCE_GENERIC<br/>product_id<br/>product_name<br/>product_price<br/>product_category<br/>product_category2<br/>product_category3<br/>product_image<br/>product_urls<br/>product_ratings<br/>source<br/>timestamp"]
        SOURCE_AUCHAN["SOURCE_AUCHAN<br/>Product_Name<br/>Product_ID<br/>Price<br/>Price_per_unit<br/>Brand<br/>Category<br/>Image_URL<br/>Product_Link<br/>source<br/>tracking_date"]
        SOURCE_CONTINENTE["SOURCE_CONTINENTE<br/>product_id<br/>product_name<br/>product_price<br/>product_image<br/>product_url<br/>product_rating<br/>source<br/>timestamp"]
        SOURCE_PINGO_DOCE["SOURCE_PINGO_DOCE<br/>product_id<br/>product_name<br/>product_price<br/>product_image<br/>product_url<br/>product_rating<br/>source<br/>timestamp"]
    end

    subgraph STAGING["STAGING SCHEMA"]
        OUTER_JOIN_STAGING["OUTER_JOIN_STAGING<br/>product_id<br/>product_name<br/>product_price<br/>category_level1<br/>category_level2<br/>category_level3<br/>image_url<br/>product_url<br/>rating<br/>brand<br/>source<br/>timestamp"]
    end

    subgraph STRUCTURED["STRUCTURED SCHEMA (NORMALIZED TABLES)"]
        PRODUCT["PRODUCT<br/>product_id_pk (PK)<br/>product_id<br/>product_name<br/>source<br/>brand<br/>image_url<br/>product_url"]
        CATEGORY_HIERARCHY["CATEGORY_HIERARCHY<br/>category_id (PK)<br/>category_level1<br/>category_level2<br/>category_level3"]
        PRODUCT_CATEGORY["PRODUCT_CATEGORY<br/>product_id_pk (PK, FK)<br/>category_id (FK)"]
        PRODUCT_PRICING["PRODUCT_PRICING<br/>product_id_pk (PK, FK)<br/>price_integer<br/>price_decimal<br/>price_currency<br/>price_per_unit<br/>timestamp (PK)"]
        PRODUCT_RATING["PRODUCT_RATING<br/>product_id_pk (PK, FK)<br/>rating_value<br/>timestamp (PK)"]
        PRODUCT_ATTRIBUTES["PRODUCT_ATTRIBUTES<br/>product_id_pk (PK, FK)<br/>product_type<br/>quantity_weight<br/>quantity_units<br/>units<br/>timestamp"]
    end

    %% Define relationships between schemas (cross-schema)
    SOURCE_GENERIC -- "transformed to" --> OUTER_JOIN_STAGING
    SOURCE_AUCHAN -- "transformed to" --> OUTER_JOIN_STAGING
    SOURCE_CONTINENTE -- "transformed to" --> OUTER_JOIN_STAGING
    SOURCE_PINGO_DOCE -- "transformed to" --> OUTER_JOIN_STAGING

    OUTER_JOIN_STAGING -- "loads" --> PRODUCT
    OUTER_JOIN_STAGING -- "extracts categories" --> CATEGORY_HIERARCHY
    OUTER_JOIN_STAGING -- "extracts pricing" --> PRODUCT_PRICING
    OUTER_JOIN_STAGING -- "extracts ratings" --> PRODUCT_RATING
    OUTER_JOIN_STAGING -- "extracts attributes" --> PRODUCT_ATTRIBUTES

    %% Define relationships within structured schema
    PRODUCT -- "has" --> PRODUCT_CATEGORY
    CATEGORY_HIERARCHY -- "belongs to" --> PRODUCT_CATEGORY
    PRODUCT -- "has" --> PRODUCT_PRICING
    PRODUCT -- "has" --> PRODUCT_RATING
    PRODUCT -- "has" --> PRODUCT_ATTRIBUTES

    %% Styling
    classDef sourceClass fill:#e6f7ff,stroke:#1890ff
    classDef stagingClass fill:#fff7e6,stroke:#fa8c16
    classDef structuredClass fill:#f6ffed,stroke:#52c41a

    class SOURCE_GENERIC,SOURCE_AUCHAN,SOURCE_CONTINENTE,SOURCE_PINGO_DOCE sourceClass
    class OUTER_JOIN_STAGING stagingClass
    class PRODUCT,CATEGORY_HIERARCHY,PRODUCT_CATEGORY,PRODUCT_PRICING,PRODUCT_RATING,PRODUCT_ATTRIBUTES structuredClass
```

## Pipeline Class Diagram

```mermaid
classDiagram
    %% Source Dataclasses (Input)
    class SourceGenericData {
        +str product_id
        +str product_name
        +str product_price
        +str product_category
        +str product_category2
        +str product_category3
        +str product_image
        +str product_urls
        +str product_ratings
        +str product_labels
        +str product_promotions
        +str source
        +str timestamp
    }

    class SourceAuchanData {
        +str Product_Name
        +str Product_ID
        +float Price
        +str Price_per_unit
        +str Brand
        +str Category
        +str Image_URL
        +int Minimum_Quantity
        +str Product_Link
        +str cgid
        +str tracking_date
        +str source
    }

    class SourceContinenteData {
        +str product_id
        +str product_name
        +float product_price
        +str product_image
        +str product_url
        +str product_rating
        +str source
        +str timestamp
    }

    class SourcePingDoceData {
        +str product_id
        +str product_name
        +float product_price
        +str product_image
        +str product_url
        +str product_rating
        +str source
        +str timestamp
    }

    %% Staging Dataclass (Output)
    class StagingData {
        +str product_id
        +str product_name
        +float product_price
        +str category_level1
        +str category_level2 
        +str category_level3
        +str image_url
        +str product_url
        +str rating
        +str brand
        +str source
        +datetime timestamp
    }

    %% Main Standardizer Class
    class ProductDataStandardizer {
        +dict FIELD_MAPPINGS
        +static extract_price(str price_str) float
        +static standardize_timestamp(str timestamp_str) datetime
        +static split_categories(DataFrame df) DataFrame
        +text_to_integer_encoding(str text) int
        +standardize_data(DataFrame df, str source) DataFrame
        +standardize_generic(SourceGenericData data) StagingData
        +standardize_auchan(SourceAuchanData data) StagingData
        +standardize_continente(SourceContinenteData data) StagingData
        +standardize_pingo_doce(SourcePingDoceData data) StagingData
    }

    %% Relationships
    SourceGenericData ..> ProductDataStandardizer : input to
    SourceAuchanData ..> ProductDataStandardizer : input to
    SourceContinenteData ..> ProductDataStandardizer : input to
    SourcePingDoceData ..> ProductDataStandardizer : input to
    ProductDataStandardizer ..> StagingData : produces
```