## Ingestion Layer

```mermaid
erDiagram
    SOURCE_GENERIC {
        string product_id
        string product_name
        float product_price
        string product_category
        string product_category2
        string product_category3
        string product_image
        string product_urls
        string product_ratings
        string product_labels
        string product_promotions
        string source
        timestamp timestamp
    }

    SOURCE_AUCHAN {
        string Product_Name
        string Product_ID
        float Price
        string Price_per_unit
        string Brand
        string Category
        string Image_URL
        int Minimum_Quantity
        string Product_Link
        string cgid
        timestamp tracking_date
        string source
    }

    SOURCE_CONTINENTE {
        string product_id
        string product_name
        float product_price
        string product_image
        string product_url
        string product_rating
        string source
        timestamp timestamp
    }

    SOURCE_PINGO_DOCE {
        string product_id
        string product_name
        float product_price
        string product_image
        string product_url
        string product_rating
        string source
        timestamp timestamp
    }

    OUTER_JOIN_STAGING {
        string product_id
        string product_name
        float product_price
        string category_level1
        string category_level2 
        string category_level3
        string image_url
        string product_url
        string rating
        string brand
        string source
        timestamp timestamp
    }

    SOURCE_GENERIC ||--o{OUTER_JOIN_STAGING : "transformed to"
    SOURCE_AUCHAN ||--o{OUTER_JOIN_STAGING : "transformed to"
    SOURCE_CONTINENTE ||--o{OUTER_JOIN_STAGING : "transformed to"
    SOURCE_PINGO_DOCE ||--o{OUTER_JOIN_STAGING : "transformed to"
```

- Staging Layer belongs to first layer of Postgres database;
- Staging Layer has a relaxed format resulting of the outer join of source schemas
- Staging Layer has a small retention period (less than 1 month)

## Structured layer

```mermaid
erDiagram
    PRODUCT {
        int product_id_pk PK
        string product_id
        string product_name
        string source
        string brand
        string image_url
        string product_url
    }
    
    CATEGORY_HIERARCHY {
        int category_id PK
        string category_level1
        string category_level2
        string category_level3
    }
    
    PRODUCT_CATEGORY {
        int product_id_pk PK, FK
        int category_id FK
    }
    
    PRODUCT_PRICING {
        int product_id_pk PK, FK
        int price_integer
        int price_decimal
        string price_currency
        string price_per_unit
        int timestamp PK
    }
    
    PRODUCT_RATING {
        int product_id_pk PK, FK
        string rating_value
        int timestamp PK
    }
    
    PRODUCT ||--o{ PRODUCT_CATEGORY : has
    CATEGORY_HIERARCHY ||--o{ PRODUCT_CATEGORY : belongs_to
    PRODUCT ||--o{ PRODUCT_PRICING : has
    PRODUCT ||--o{ PRODUCT_RATING : has
```

- Product pricing was quantized due to supabase 500MB limits
- Data types optimized

## Augmented layer

```mermaid
erDiagram
    PRODUCT_ATTRIBUTES {
        int product_id_pk PK, FK
        string product_type
        float quantity_weight
        int quantity_units
        string units
        int timestamp
    }
```