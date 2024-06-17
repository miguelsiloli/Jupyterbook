# Data Modelling

## Introduction

Our pipeline design predicts a staging logical schema, normalized schema and serving schema. 

```{mermaid}
flowchart TD
    A[Staging Database] -- (-) Minimum ETL Jobs<br> (+) Data Source Preservation --> B
    B[Normalized Database] -- Conceptual Division<br> (+) Query Complexity<br> (+) Index Support --> C[Serving Database]
```

:::{grid-item-card} Staging Schema

The staging schema is a minimum ETL approach which makes no transformations to the data aside from flattening the JSON into a dataframe. It acts as an intermediate storage to help with backfills.
:::
:::{grid-item-card} Normalized Schema

The normalized schema operates in a star schema. It consists of fact tables and dimension tables, making I/O operations much faster due to effective indexing for each table. This schema supports the division of unnormalized content into conceptual fields such as users, products, and catalogs tables, is user-friendly, and supports BI operations.
:::
:::{grid-item-card} Serving Schema

The serving layer refers to specific use cases, serving normalized data within a restricted scope with more granularity and transformations.
:::
::::


## Schemas

### Staging Schema

```{mermaid}
flowchart TD
    subgraph source ["Source"]
        Catalog_Pipeline["Catalog Pipeline"]
        Tracking_Pipeline["Tracking Pipeline"]
        Images_Pipeline["Images Pipeline"]
        Brands_Pipeline["Brands Pipeline"]
    end

    subgraph sink ["Sink"]
        Catalog_Staging["Catalog Staging"]
        Tracking_Staging["Tracking Staging"]
        User_Staging["User Staging"]
        S3_Image_Storage["S3 Image Storage"]
        Brands_Staging["Brands Staging"]
    end

    Catalog_Pipeline --> |feeds| Catalog_Staging
    Tracking_Pipeline --> |feeds| Tracking_Staging
    Tracking_Pipeline --> |feeds| User_Staging
    Images_Pipeline --> |feeds| S3_Image_Storage
    Brands_Pipeline --> |feeds| Brands_Staging
```

|![Local Image](../assets/data_engineering/staging_schema.png)|
|:--:| 
|Staging Schema|

### Normalized Schema

```{mermaid}
erDiagram
    users_fact {
        INTEGER user_id PK "Unique identifier for the user"
        VARCHAR username "Username of the user"
        VARCHAR email "Email address of the user"
        BOOLEAN is_active "Status indicating if the user is active"
        TIMESTAMP created_at "Timestamp when the user record was created"
        TIMESTAMP updated_at "Timestamp when the user record was last updated"
        INTEGER country_id FK "Foreign key to the country_dim table"
        VARCHAR country_title "Title of the user's country"
        VARCHAR profile_url "URL to the user's profile"
        DATE date "Date of user-related event"
        INTEGER feedback_count "Count of feedback received by the user"
    }

    tracking_fact {
        INTEGER tracking_id PK "Unique identifier for the tracking record"
        INTEGER user_id FK "Foreign key to the users_fact table"
        INTEGER product_id FK "Foreign key to the catalog_fact table"
        TIMESTAMP tracking_time "Timestamp when the tracking occurred"
        BOOLEAN is_active "Status indicating if the tracking is active"
        TIMESTAMP created_at "Timestamp when the tracking record was created"
        TIMESTAMP updated_at "Timestamp when the tracking record was last updated"
    }

    catalog_fact {
        INTEGER product_id PK "Unique identifier for the product"
        VARCHAR product_name "Name of the product"
        INTEGER brand_id FK "Foreign key to the brand"
        FLOAT price "Price of the product"
        VARCHAR description "Description of the product"
        BOOLEAN is_available "Availability status of the product"
        TIMESTAMP created_at "Timestamp when the product record was created"
        TIMESTAMP updated_at "Timestamp when the product record was last updated"
        INTEGER category_id FK "Foreign key to the category"
        VARCHAR category_title "Title of the product category"
        BOOLEAN is_deleted "Status indicating if the product is deleted"
    }

    city_dim {
        INTEGER city_id PK "Unique identifier for the city"
        VARCHAR city "Name of the city"
        INTEGER country_id FK "Foreign key to the country_dim table"
    }

    color_dim {
        INTEGER color_id PK "Unique identifier for the color"
        VARCHAR color_title "Title of the color"
    }

    country_dim {
        INTEGER country_id PK "Unique identifier for the country"
        VARCHAR country "Name of the country"
    }

    date_dim {
        INTEGER date_id PK "Unique identifier for the date"
        DATE full_date "Full date value"
        INTEGER day_of_week "Day of the week"
        INTEGER day_of_month "Day of the month"
        INTEGER month "Month of the year"
        INTEGER year "Year"
    }

    product_dim {
        INTEGER product_id PK "Unique identifier for the product"
        VARCHAR product_name "Name of the product"
        VARCHAR category "Category of the product"
        INTEGER color_id FK "Foreign key to the color_dim table"
    }

    customer_dim {
        INTEGER customer_id PK "Unique identifier for the customer"
        VARCHAR first_name "First name of the customer"
        VARCHAR last_name "Last name of the customer"
        VARCHAR email "Email address of the customer"
        INTEGER city_id FK "Foreign key to the city_dim table"
    }

    address_dim {
        INTEGER address_id PK "Unique identifier for the address"
        VARCHAR address "Address details"
        VARCHAR postal_code "Postal code"
        VARCHAR phone "Phone number"
        INTEGER city_id FK "Foreign key to the city_dim table"
    }

    users_fact ||--o{ country_dim : "belongs_to"
    users_fact ||--o{ city_dim : "located_in"
    tracking_fact ||--o{ users_fact : "tracked_by"
    tracking_fact ||--o{ catalog_fact : "tracks"
    catalog_fact ||--o{ product_dim : "describes"
    catalog_fact ||--o{ color_dim : "has_color"
    catalog_fact ||--o{ address_dim : "available_at"
    customer_dim ||--o{ city_dim : "lives_in"
    address_dim ||--o{ city_dim : "located_in"
```

#### Users Fact

| Column                  | Type    | Collation | Nullable | Default | Primary Key | Foreign Key                                            |
|-------------------------|---------|-----------|----------|---------|--------------|--------------------------------------------------------|
| user_id                 | bigint  |           | not null |         | Yes          | References users_dim(user_id)                          |
| item_count              | integer |           |          |         | No           |                                                        |
| given_item_count        | integer |           |          |         | No           |                                                        |
| taken_item_count        | integer |           |          |         | No           |                                                        |
| followers_count         | integer |           |          |         | No           |                                                        |
| following_count         | integer |           |          |         | No           |                                                        |
| positive_feedback_count | integer |           |          |         | No           |                                                        |
| negative_feedback_count | integer |           |          |         | No           |                                                        |
| feedback_reputation     | integer |           |          |         | No           |                                                        |
| feedback_count          | integer |           |          |         | No           |                                                        |
| date                    | date    |           | not null |         | Yes          | References date_dimension(date_value)                  |
| city_id                 | integer |           |          |         | No           | References city_dim(city_id)                           |
| country_id              | integer |           |          |         | No           | References country_dim(country_id)                     |

#### Tracking Fact

| Column                 | Type                   | Collation | Nullable | Default | Primary Key | Foreign Key                                    |
|------------------------|------------------------|-----------|----------|---------|--------------|------------------------------------------------|
| product_id             | bigint                 |           | not null |         | Yes          |                                                |
| catalog_id             | integer                |           |          |         | No           |                                                |
| brand_title            | character varying(155) |           |          |         | No           |                                                |
| date                   | date                   |           | not null |         | Yes          | References date_dimension(date_value)          |
| size_title             | character varying(255) |           |          |         | No           |                                                |
| color_id               | integer                |           |          |         | No           |                                                |
| favourite_count        | integer                |           |          |         | No           |                                                |
| view_count             | integer                |           |          |         | No           |                                                |
| created_at             | date                   |           |          |         | No           |                                                |
| original_price_numeric | double precision       |           |          |         | No           |                                                |
| price_numeric          | double precision       |           |          |         | No           |                                                |
| package_size_id        | smallint               |           |          |         | No           |                                                |
| service_fee            | double precision       |           |          |         | No           |                                                |
| user_id                | bigint                 |           |          |         | No           | References users_dim(user_id)                  |
| status                 | character varying(155) |           |          |         | No           |                                                |
| description            | text                   |           |          |         | No           |                                                |


### Indexes

"tracking_fact_pkey" PRIMARY KEY, btree (product_id, date)

"catalog_fact_pkey" PRIMARY KEY, btree (catalog_id, date)

"users_fact_pkey" PRIMARY KEY, btree (user_id, date)


## DBT

DBT (Data Build Tool) is an ETL (Extract, Transform, Load) tool that enables data analysts and engineers to transform data in the warehouse more effectively. It leverages SQL scripts to build tables and views in the database.


Before using DBT, I actually built a solution in pure python and SQL. I had to build from scratch each function and each step. It was very time consuming, prone to errors and cucumbersome to document.

**Example workloads**


```{mermaid}
graph TD
    title[Tracking staging]
    B[create_engine]
    B --> C[load_from_tracking_staging]
    C --> D[color_dim_transform]
    C --> E[tracking_fact_transform]
    D --> F[export_color_dim]
    E --> G[export_tracking_fact]
```

```{mermaid}
graph TD
    title[Users staging]
    B[create_engine]
    B --> C[load_from_users_staging]
    C --> D[city_dim_transform]
    C --> E[country_dim_transform]
    C --> F[users_dim_transform]
    C --> G[users_fact_transform]
    D --> H[export_city_dim]
    E --> I[export_country_dim]
    F --> J[export_users_dim]
    G --> K[export_users_fact]
```

```{python}
@flow(name = "Normalize tracking_staging")
def normalize_tracking_staging():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_from_tracking_staging(engine = engine)
    color_dim = color_dim_transform(data)
    fact_data = tracking_fact_transform(data)
    export_color_dim(color_dim, engine = engine)
    export_tracking_fact(fact_data, engine= engine)
```

```{python}
@flow(name = "Normalize users_staging")
def normalize_users_staging():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_from_users_staging(engine = engine)
    city_dim = city_dim_transform(data)
    country_dim = country_dim_transform(data)
    users_dim = users_dim_transform(data)
    users_fact = users_fact_transform(data)
    export_country_dim(country_dim, 
                       engine = engine)    
    export_city_dim(city_dim, 
                    engine = engine)
    export_users_dim(users_dim, 
                     engine = engine)
    export_users_fact(users_fact, 
                     engine = engine)
```

.. tabs::

   .. tab:: Apples

      Apples are green, or sometimes red.

   .. tab:: Pears

      Pears are green.

   .. tab:: Oranges

      Oranges are orange.

### Configurations and connections

**profiles.yml**
```{yaml}
star_schema_profile:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      user: your_user
      password: your_password
      port: 5432
      dbname: your_db
      schema: public
```

**dbt_project.yml**

```{yaml}
name: star_schema_project
version: 1.0.0
profile: star_schema_profile

config-version: 2

model-paths: ["models"]
seed-paths: ["seeds"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_modules"

models:
  star_schema_project:
    marts:
      core:
        materialized: table
```