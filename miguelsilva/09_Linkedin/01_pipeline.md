# Part I: Meet the market
> Using Foundational Models to Extract Career Guiding Insights from LinkedIn Listings

---

## Introduction & Challenge

> 💡 **As a junior developer, how can I optimize my specialization path based on the assumption that not all skills are equally valuable?**

This project addresses the challenge of navigating today's complex job market by leveraging large language models to extract actionable insights from LinkedIn job listings.

*   🔍 **Market Navigation:** Understanding job requirements
*   🏆 **Skill Prioritization:** Identifying valuable skills
*   📈 **Career Planning:** Optimizing learning path
*   💡 **Opportunity Spotting:** Finding real demands

This is going to be split into two parts: traditional **pipeline setup** and **gemini augmentation architecture**.

## Architecture Overview

[View on Eraser![](https://app.eraser.io/workspace/L6NaMhN79etO75ci6TWx/preview?elements=v2IerBw6zG3liAsf_yPNyA&type=embed)](https://app.eraser.io/workspace/L6NaMhN79etO75ci6TWx?elements=v2IerBw6zG3liAsf_yPNyA)

### Components

**Tech Stack:** `GitHub Actions`, `Python`, `Selenium`, `GCP`, `BigQuery`, `Dataform`, `Parquet`

| Component              | Technology         | Description                     |
| :--------------------- | :----------------- | :------------------------------ |
| **Runner Environment** | GitHub Cloud       | Executing pipeline processes    |
| **Orchestrator**       | GitHub Actions     | Scheduling pipeline executions  |
| **Web Scraping**       | Python + Selenium  | Extract data from LinkedIn      |
| **Object Storage**     | Google Cloud Storage | Staging raw scraped data        |
| **Data Warehousing**   | Google BigQuery    | Structured job data storage     |
| **Training Hardware**  | NVIDIA P100        | GPU-accelerated ML training   |

---

### Pipelines

Each workflow is a github actions yml.

Scrape and Ingest GCS >> GCS to BQ sink >> Gemini Augmentation 


```{mermaid}
graph TD
    %% Workflow 3
    subgraph W3[Workflow 3: Gemini Augmentation]
        A3[Staging Updated Event]
        A3 --> W3_Entry[Start Augment]
        W3_Entry --> W3_S1[Select job_ids]
        W3_S1 --> W3_S2[Call Gemini API]
        W3_S2 --> W3_S3[Store Results]
        W3_S3 --> W3_Output[Augmented Ready]
    end
    %% Workflow 2
    subgraph W2[Workflow 2: GCS to BQ sink]
        A2[GCS Event: New Parquet]
        A2 --> W2_Entry[Start BQ Load]
        W2_Entry --> W2_S1[Fetch Parquet]
        W2_S1 --> W2_S2[Load to Temp Table]
        W2_S2 --> W2_S3[Merge to Staging]
        W2_S3 --> W2_Output[Staging Updated]
    end
    %% Workflow 1
    subgraph W1[Workflow 1: Scrape and Ingest GCS]
        A1[External Job Listings]
        A1 --> W1_Entry[Start Scrape]
        W1_Entry --> W1_S1[Scrape Listings]
        W1_S1 --> W1_S2[Store HTML]
        W1_S2 --> W1_S3[HTML to Parquet]
        W1_S3 --> W1_S4[Upload to GCS]
        W1_S4 --> W1_Output[GCS Ready]
    end
    
    %% Styling
    style W1_Entry fill:#4285F4,stroke:#333,stroke-width:2px,color:#fff
    style W1_Output fill:#2E7D32,stroke:#333,stroke-width:2px,color:#fff
    style W2_Entry fill:#EA4335,stroke:#333,stroke-width:2px,color:#fff
    style W2_Output fill:#2E7D32,stroke:#333,stroke-width:2px,color:#fff
    style W3_Entry fill:#34A853,stroke:#333,stroke-width:2px,color:#fff
    style W3_Output fill:#2E7D32,stroke:#333,stroke-width:2px,color:#fff
    
    %% Node styles
    classDef w1_nodes_style fill:#81D4FA,stroke:#01579B,color:#000
    classDef w2_nodes_style fill:#FFCDD2,stroke:#B71C1C,color:#000
    classDef w3_nodes_style fill:#C8E6C9,stroke:#1B5E20,color:#000
    class W1_S1,W1_S2,W1_S3,W1_S4 w1_nodes_style
    class W2_S1,W2_S2,W2_S3 w2_nodes_style
    class W3_S1,W3_S2,W3_S3 w3_nodes_style
    class A1,A2,A3 fill:#FBBC05,stroke:#333,stroke-width:2px,color:#000
    
    %% Subgraph styles
    classDef workflow1 fill:#E3F2FD,stroke:#1976D2,color:#000
    classDef workflow2 fill:#FFEBEE,stroke:#C62828,color:#000
    classDef workflow3 fill:#E8F5E9,stroke:#2E7D32,color:#000
    class W1 workflow1
    class W2 workflow2
    class W3 workflow3
```

### Data Model

#### Staging

```{mermaid}
classDiagram
    class SchemaLinkedInSink {
        <<Schema>>
    }
    
    class SchemaLinkedInDataform {
        <<Schema>>
    }
    
    class linkedin_jobs_staging {
        +string job_id PK
        +string job_title
        +string company_name
        +string location
        +string employment_type
        +string experience_level
        +string workplace_type
        +string job_description
        +date ingestionDate PK
    }
    
    class linkedin_augmented_staging {
        +string job_id PK
        +struct job_summary
        +struct company_information
        +struct location_and_work_model
        +struct required_qualifications
        +struct preferred_qualifications
        +struct role_context
        +struct benefits
    }
    
    class stg_linkedin_tech_cloud_platforms {
        +string job_id FK
        +string technology_category
        +string technology_name
    }
    
    class stg_linkedin_tech_cloud_services_tools {
        +string job_id FK
        +string technology_category
        +string technology_name
    }
    
    class stg_linkedin_tech_data_architecture {
        +string job_id FK
        +string technology_category
        +string technology_name
    }
    
    class stg_linkedin_tech_programming_languages {
        +string job_id FK
        +string technology_category
        +string technology_name
    }
    
    class int_linkedin_tech_all_categories {
        +string job_id FK
        +string technology_category
        +string technology_name
    }
    
    %% Schema relationships
    SchemaLinkedInSink -- linkedin_jobs_staging : contains
    SchemaLinkedInSink -- linkedin_augmented_staging : contains
    
    SchemaLinkedInDataform -- stg_linkedin_tech_cloud_platforms : contains
    SchemaLinkedInDataform -- stg_linkedin_tech_cloud_services_tools : contains
    SchemaLinkedInDataform -- stg_linkedin_tech_data_architecture : contains
    SchemaLinkedInDataform -- stg_linkedin_tech_programming_languages : contains
    SchemaLinkedInDataform -- int_linkedin_tech_all_categories : contains
    
    %% Table relationships
    linkedin_jobs_staging -- linkedin_augmented_staging : job_id
    
    linkedin_augmented_staging --> stg_linkedin_tech_cloud_platforms : extract
    linkedin_augmented_staging --> stg_linkedin_tech_cloud_services_tools : extract
    linkedin_augmented_staging --> stg_linkedin_tech_data_architecture : extract
    linkedin_augmented_staging --> stg_linkedin_tech_programming_languages : extract
    
    stg_linkedin_tech_cloud_platforms --> int_linkedin_tech_all_categories : "UNION ALL"
    stg_linkedin_tech_cloud_services_tools --> int_linkedin_tech_all_categories : "UNION ALL"
    stg_linkedin_tech_data_architecture --> int_linkedin_tech_all_categories : "UNION ALL"
    stg_linkedin_tech_programming_languages --> int_linkedin_tech_all_categories : "UNION ALL"
```

#### Augmented Normalized

```{mermaid}
classDiagram
    class JobDescriptionSchema {
        +job_summary
        +company_information
        +location_and_work_model
        +required_qualifications
        +preferred_qualifications
        +role_context
        +benefits
    }
    
    class JobSummary {
        +String role_title
        +String role_objective
        +String role_seniority
    }
    
    class CompanyInformation {
        +String company_type
        +String[] company_values_keywords
    }
    
    class LocationAndWorkModel {
        +String specification_level
        +String remote_status
        +String[] flexibility
        +String[] locations
    }
    
    class RequiredQualifications {
        +Number experience_years_min
        +Number experience_years_max
        +String experience_description
        +String education_requirements
        +TechnicalSkills technical_skills
        +String[] methodologies_practices
        +String[] soft_skills_keywords
    }
    
    class TechnicalSkills {
        +ProgrammingLanguages programming_languages
        +String[] cloud_platforms
        +String[] cloud_services_tools
        +String[] databases
        +DataArchitectureConcepts data_architecture_concepts
        +String[] etl_integration_tools
        +String[] data_visualization_bi_tools
        +String[] devops_mlops_ci_cd_tools
        +String[] orchestration_workflow_tools
        +String[] other_tools
    }
    
    class ProgrammingLanguages {
        +String[] general_purpose
        +String[] scripting_frontend
        +String[] query
        +String[] data_ml_libs
        +String[] platform_runtime
        +String[] configuration
        +String[] other_specialized
    }
    
    class DataArchitectureConcepts {
        +String[] data_modeling
        +String[] data_storage_paradigms
        +String[] etl_elt_pipelines
        +String[] data_governance_quality
        +String[] architecture_patterns
        +String[] big_data_concepts
        +String[] cloud_data_architecture
        +String[] ml_ai_data_concepts
        +String[] core_principles_optimization
    }
    
    class PreferredQualifications {
        +String[] skills_keywords
        +String other_notes
    }
    
    class RoleContext {
        +String[] collaboration_with
        +String team_structure
        +String project_scope
        +String[] key_responsibilities
    }
    
    class Benefits {
        +String training_development
        +String[] other_benefits
    }
    
    JobDescriptionSchema -- JobSummary
    JobDescriptionSchema -- CompanyInformation
    JobDescriptionSchema -- LocationAndWorkModel
    JobDescriptionSchema -- RequiredQualifications
    JobDescriptionSchema -- PreferredQualifications
    JobDescriptionSchema -- RoleContext
    JobDescriptionSchema -- Benefits
    RequiredQualifications -- TechnicalSkills
    TechnicalSkills -- ProgrammingLanguages
    TechnicalSkills -- DataArchitectureConcepts
```

#### Phase 1: Scraping and Initial Storage
The first major step is getting the raw job data from LinkedIn. We use a GitHub Actions workflow, aptly named "Scrape and Parse to GCS", to manage this.

1.  **Environment & Dependencies**
    The workflow runs on an `ubuntu-latest` GitHub runner with Python 3.10.6 and Google Chrome for Selenium.
2.  **LinkedIn Scraping & On-the-Fly Parsing**
    Uses Selenium to control Chrome, extracting and structuring data from LinkedIn job listings.
3.  **Loading to Data Warehouse**
    Efficiently batch-loads the parsed data from GCS into Google BigQuery for analysis.

#### Example data in linkedin_staging

| Row | job_id     | job_title             | company_name | location                | employment_type | experience_level | workplace_type | applicant_count        | reposted_info | skills_summary                                | application_type | job_description                                                                                                                                                              | job_link            | company_logo_url | source_file                                                                      | ingestionDate |
|-----|------------|-----------------------|--------------|-------------------------|-----------------|------------------|----------------|------------------------|---------------|-----------------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|------------------|----------------------------------------------------------------------------------|---------------|
| 1   | None       | None                  | None         | None                    | None            | None             | None           | None                   | None          | None                                          | None             | None                                                                                                                                                                         | None                | None             | linkedin_Data Science OR Data Engineer _Portugal_page_34_job_4209744798.html | 2025-05-02    |
| 2   | 4214667742 | DevOps Engineer Mid   | None         | Porto, Porto, Portugal  | None            | Mid-Senior level | Full-time      | 6 people clicked apply | 1 week ago    | Skills: DevOps, Infrastructure as code (IaC), +8 more | Apply            | About Vaibe<br>Vaibe is a leading B2B white-label software gamification venture of Körber Digital, envisions a future where every software provider seamlessly integrates gamification software features tailored to their unique needs and brand identity. Through strategic partnerships and collaboration, w... | DevOps Engineer Mid | None             | linkedin_Data Science OR Data Engineer _Portugal_page_16_job_4214667742.html | 2025-05-02    |
| 3   | 4214667742 | DevOps Engineer Mid   | None         | Porto, Porto, Portugal  | None            | Mid-Senior level | Full-time      | 5 people clicked apply | 5 days ago    | None                                          | Apply            | About Vaibe<br>Vaibe is a leading B2B white-label software gamification venture of Körber Digital, envisions a future where every software provider seamlessly integrates gamification software features tailored to their unique needs and brand identity. Through strategic partnerships and collaboration, w... | DevOps Engineer Mid | None             | linkedin_Data Science OR Data Engineer _Portugal_page_06_job_4214667742.html | 2025-05-02    |

---

## Data Transformation & Example

### Phase 2: Data Cleansing and Normalization
Raw data, even after initial parsing, is rarely perfect. To prepare it for meaningful analysis and LLM fine-tuning, we employ Dataform for in-warehouse transformations within BigQuery.

> **Key Transformations:**
> *   **Normalizing Technology Names:** Standardizing variations like "python", "Python3", "python 3.x" to "python"
> *   **Downcasing Text:** Converting all relevant text fields to lowercase
> *   **Removing Extra Spaces and Special Characters:** Cleaning up text fields

```{mermaid}
erDiagram
    linkedin_jobs_staging {
        STRING job_id PK "Unique identifier"
        DATE ingestionDate PK "Partition key"
        STRING job_title
        STRING company_name
        STRING location
        STRING job_description
    }

    linkedin_augmented_staging {
        STRING job_id PK "Links to staging"
        STRUCT job_summary "Title, objective, seniority"
        STRUCT required_qualifications "Experience, skills"
        STRUCT preferred_qualifications "Skills, notes"
    }

    linkedin_jobs_staging ||--o{ linkedin_augmented_staging : "1-to-1"
```

### Example Scraped Description
```{plaintext}
Job Description:
About Vaibe
Vaibe is a leading B2B white-label software gamification venture of Körber Digital, envisions a future where every software provider seamlessly integrates gamification software features tailored to their unique needs and brand identity.

About The Role
We are looking for a Mid-Level DevOps Engineer to join our team and help scale, automate, and optimize our cloud infrastructure. You will be responsible for designing, implementing, and maintaining CI/CD pipelines, managing Azure cloud infrastructure, and ensuring smooth deployments using Terraform and GitHub Actions.

Key Responsibilities
Infrastructure as Code (IaC): Design, implement, and maintain Azure infrastructure using Terraform.
CI/CD Automation: Build and manage GitHub Actions workflows to streamline deployments and testing.
Cloud Operations: Ensure high availability, security, and performance of our Azure-based environments.
Monitoring & Observability: Set up logging, monitoring, and alerting solutions to proactively identify issues.

Must-Have Skills
Terraform – Experience writing and managing infrastructure as code (IaC).
Azure Cloud – Hands-on experience with Azure services like AKS, Azure DevOps, App Services, Networking, and Storage.
GitHub Actions – Experience setting up and optimizing CI/CD pipelines.
Linux & Scripting – Familiarity with shell scripting, PowerShell, or Python for automation.
```

---

## Part II: Gemini Augmentation

> Distilled from Gemini-2.0-flash for structured information extraction

#### Prompt

```python
JOB_DESCRIPTION_PROMPT = """You are an expert AI assistant tasked with parsing job descriptions and extracting key information into a structured JSON format based on the predefined schema below. Analyze the input job description text carefully, standardize relevant terms (like tool names, locations, role titles), apply controlled vocabularies where specified, and generate the corresponding JSON output. Ensure lists contain standardized values and are empty `[]` if no relevant information is found.

**JSON Schema Definition:**

{
  "_comment": "Schema for storing structured job description data.",
  "job_summary": {
    "role_title": {
      "type": ["string", "null"],
      "description": "Standardized primary role title inferred from the description (e.g., 'Data Engineer', 'Data Scientist', 'Cloud Engineer', 'BI Developer'). Use the most specific fitting category."
    },
    "role_objective": {
      "type": ["string", "null"],
      "description": "A concise summary or direct quote of the primary goal or objective of the role as stated in the description."
    },
    "role_seniority": {
      "type": ["string", "null"],
      "description": "Inferred or stated seniority level. Use one of: ['Internship', 'Junior', 'Mid-Level', 'Senior', 'Lead', 'Staff', 'Principal', 'Manager', 'Director', 'Executive', 'Not Specified']."
    },
    "visa_sponsorship": {
        "type": ["boolean", "null"],
        "description": "Set to true if the company explicitly states they Concepts", "NoSQL Concepts", "Other"]
      },
      "data_architecture_concepts": {
          "type": "object",
           "properties": {
                "data_modeling": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Data modeling techniques."},
                "data_storage_paradigms": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Data storage concepts/systems."},
                "etl_elt_pipelines": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Data movement/transformation concepts."},
                "data_governance_quality": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Data governance/quality concepts."},
                "architecture_patterns": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Data architecture patterns."},
                "big_data_concepts": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Big Data specific concepts."},
                "cloud_data_architecture": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Cloud-specific data architecture concepts."},
                "ml_ai_data_concepts": { "type": ["array", "null"], "items": {"type": "string"}, "description": "ML/AI infrastructure/data concepts."},
                "core_principles_optimization": { "type": ["array", "null"], "items": {"type": "string"}, "description": "Core design/optimization principles."}
           },
          "description": "Categorized required knowledge of data architecture concepts."
          // Possible Values within sub-arrays: Standardized concepts like "Dimensional Modeling", "Data Lake Architecture", "ETL Design & Development", "Data Quality Management", "Medallion Architecture", etc.
      },
      "etl_integration_tools": {
          "type": ["array", "null"],
          "items": { "type": "string" },
          "description": "List specific ETL, ELT, or Data Integration tools required."
          // Possible Values: Standardized tool names like "Azure Data Factory", "AWS Glue", "dbt (Data Build Tool)", "Informatica PowerCenter / IDMC", "Talend", "Microsoft SSIS", "Airbyte", "Fivetran", "Matillion", etc.
      },
      "data_visualization_bi_tools": {
          "type": ["array", "null"],
          "items": { "type": "string" },
          "description": "List specific Business Intelligence or Data Visualization tools required."
          // Possible Values: Standardized tool names like "Tableau", "Microsoft Power BI", "Looker / Looker Studio", "QlikView / Qlik Sense", "MicroStrategy", "Apache Superset", "Metabase", "Grafana", "Kibana", "Power Query (Excel/Power BI)", "DAX", "LookML", etc.
      },
      "devops_mlops_ci_cd_tools": {
          "type": ["array", "null"],
          "items": { "type": "string" },
          "description": "List specific DevOps, MLOps, CI/CD, IaC, or Monitoring tools required."
           // Possible Values: Standardized tool names like "Git", "Jenkins", "Terraform", "Kubernetes", "Docker", "Azure DevOps", "GitHub Actions", "MLflow", "Kubeflow", "Datadog", "Prometheus", "Boto3 (AWS SDK for Python)", etc.
      },
      "orchestration_workflow_tools": {
          "type": ["array", "null"],
          "items": { "type": "string" },
          "description": "List specific workflow orchestration tools required."
          // Possible Values: Standardized tool names like "Apache Airflow", "Prefect", "Dagster", "Luigi", "AWS Step Functions", "Azure Logic Apps", etc.
      },
      "other_tools": {
          "type": ["array", "null"],
          "items": { "type": "string" },
          "description": "List other relevant tools not offer visa sponsorship for this role, false if they state they do not. Null if not mentioned."
    }
  },
  "company_information": {
    "company_type": {
      "type": ["string", "null"],
      "description": "Categorize the company based on its primary business model or industry. Use one of: ['Software Product / SaaS', 'E-commerce / Marketplace Platform', 'Fintech', 'Gaming Company / GameTech', 'IT Consulting / System Integration', 'IT Outsourcing / Nearshore / Dev Shop', 'Managed Service Provider (MSP)', 'AI / Data Science Focused', 'Open Source Software Company', 'Low-Code / No-Code Platform', 'Cloud / IT Infrastructure Services', 'Digital Services / Agency', 'Tech Hub / Academy / Recruitment', 'Testing / Inspection / Certification', 'Banking / Financial Institution', 'Healthcare / Pharma / Biotech', 'Automotive / Mobility Provider', 'Manufacturing / Industrial', 'Logistics / Transportation', 'Energy', 'Telecommunications', 'Engineering Services (Non-IT specific)', 'Internal IT / Shared Services', 'Unspecified / Generic Tech', 'Not Specified / Other']."
    },
    "company_values_keywords": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List keywords or short phrases representing explicitly stated company values or culture aspects (e.g., 'Inovação', 'Collaboration', 'Transparency', 'Work-life balance')."
    }
  },
  "location_and_work_model": {
    "specification_level": {
        "type": "string",
        "enum": ["Specific Location / Remote Status Identified", "Not Specified"],
        "description": "Indicates if specific location, remote status, or 'Global' was identified."
    },
    "remote_status": {
      "type": ["string", "null"],
      "description": "Identify the primary work model. Use one of: ['Fully Remote', 'Remote (Region Specific)', 'Hybrid', 'Office-based', 'Not Specified']."
    },
    "flexibility": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List specific flexibility options mentioned, e.g., ['Flexible Schedule']."
    },
    "locations": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List standardized, Title Cased locations (Cities, Countries, Regions, 'Global') mentioned. Sort alphabetically."
    }
  },
  "required_qualifications": {
    "experience_years_min": {
      "type": ["integer", "null"],
      "description": "Minimum years of experience required (e.g., from '1-6 years' extract 1, from '3+ years' extract 3)."
    },
    "experience_years_max": {
      "type": ["integer", "null"],
      "description": "Maximum years of experience specified (e.g., from '1-6 years' extract 6). Null if only minimum or range isn't specified."
    },
    "experience_description": {
      "type": ["string", "null"],
      "description": "The raw text describing the experience requirement (e.g., '1 e 6 anos em projetos de Data', '3+ years of hands-on experience')."
    },
    "education_requirements": {
      "type": ["string", "null"],
      "description": "Required level of education or field of study (e.g., 'BSc Computer Science', 'Licenciatura/ Mestrado nas áreas de Engenharia Informática...')."
    },
    "technical_skills": {
      "programming_languages": {
        "type": "object",
        "properties": {
            "general_purpose": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List standardized general-purpose languages (e.g., Python, Java, Scala, Go, C#, R)."},
            "scripting_frontend": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List standardized scripting or frontend languages/frameworks (e.g., Bash / Shell Scripting, JavaScript, TypeScript, Angular)."},
            "query": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List standardized query languages (e.g., SQL, T-SQL, PL/SQL, Spark SQL, DAX, MDX, Power Query (M))."},
            "data_ml_libs": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List standardized data/ML specific libraries/frameworks (e.g., Pandas, PySpark, Scikit-learn, PyTorch, TensorFlow, R Shiny). Note: Base frameworks like Spark/Flink go here too."},
            "platform_runtime": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List specific platforms/runtimes like '.NET Platform'."},
            "configuration": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List configuration languages like 'YAML'."},
            "other_specialized": {"type": ["array", "null"], "items": {"type": "string"}, "description": "List other specialized languages like ' fitting neatly into the above categories (e.g., IDEs, Data Catalogs, Vector DBs)."
          // Possible Values: Standardized tool names like "Jupyter Notebooks/Lab", "Alation (Data Catalog)", "Dataiku", "VS Code", "Weaviate (Vector DB)", "Pinecone (Vector DB)", "Minio", "ActiveMQ", "RabbitMQ", etc.
      }
    },
    "methodologies_practices": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List required development methodologies or practices."
      // Possible Values: ["Agile Principles", "Scrum", "Kanban", "Extreme Programming (XP)", "Lean Principles", "SAFe", "LeSS", "Waterfall", "DevOps Culture/Practices", "Test-Driven Development (TDD)", "Behavior-Driven Development (BDD)", "CI/CD Practices", "A/B Testing"]
    },
    "soft_skills_keywords": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List required soft skills or general keywords."
    }
  },

  "preferred_qualifications": {
    "_comment": "Nice-to-have skills and qualifications.",
    "skills_keywords": {
        "type": ["array", "null"],
        "items": { "type": "string" },
        "description": "List of preferred skills, tools, languages, or concepts."
    },
    "other_notes": {
        "type": ["string", "null"],
        "description": "Any other text describing preferred qualifications."
    }
  },

  "role_context": {
     "_comment": "Information about the role's interactions and scope.",
     "collaboration_with": {
        "type": ["array", "null"],
        "items": { "type": "string" },
        "description": "List of teams or roles this position collaborates with."
     },
     "team_structure": {
        "type": ["string", "null"],
        "description": "Description of the team structure or context."
     },
     "project_scope": {
        "type": ["string", "null"],
        "description": "Description of the type or scope of projects involved."
     },
     "key_responsibilities": {
        "type": ["array", "null"],
        "items": { "type": "string" },
        "description": "List of key tasks and responsibilities mentioned."
     }
  },

  "benefits": {
    "_comment": "Perks and benefits offered.",
    "training_development": {
        "type": ["string", "null"],
        "description": "Description of training and development opportunities."
    },
    "learning_platforms": {
        "type": ["array", "null"],
        "items": { "type": "string" },
        "description": "List specific learning platforms mentioned by name (e.g., 'Udemy', 'LinkedIn Learning')."
    },
    "paid_time_off_days": {
        "type": ["integer", "null"],
        "description": "Specific number of paid time off days mentioned."
    },
    "other_benefits_keywords": {
        "type": ["array", "null"],
        "items": { "type": "string" },
        "description": "List keywords for other benefits (e.g., 'Health Insurance', 'Meal Allowance', 'Well-being Program')."
    }
  }
}
"""
```


#### OpenAI schema template

```python
JOB_RESPONSE_SCHEMA = genai.types.Schema(
    type=genai.types.Type.OBJECT,
    properties={
        # === Job Summary ===
        'job_summary': genai.types.Schema(
            type=genai.types.Type.OBJECT,
            description="High-level information about the role.",
            properties={
                'role_title': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Standardized primary role title inferred from the description (e.g., 'Data Engineer', 'Data Scientist')."
                ),
                'role_objective': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="A concise summary or direct quote of the primary goal or objective of the role."
                ),
                'role_seniority': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Inferred or stated seniority level (e.g., 'Internship', 'Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Not Specified')."
                    # Enum constraint typically handled by validation after extraction or in the prompt
                ),
                 'visa_sponsorship': genai.types.Schema(
                    type=genai.types.Type.BOOLEAN,
                    description="Set to true if the company explicitly states they offer visa sponsorship for this role, false if they state they do not. Null if not mentioned."
                 )
            }
            # Optionality implied by not being in a top-level 'required' list if one were defined
        ),

        # === Company Information ===
        'company_information': genai.types.Schema(
            type=genai.types.Type.OBJECT,
            description="Details about the hiring company.",
            properties={
                'company_type': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Categorization of the company's primary business model or industry."
                    # Enum constraint handled by validation/prompt.
                ),
                'company_values_keywords': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List of explicitly stated company values or cultural keywords.",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                )
            }
        ),

        # === Location and Work Model ===
        'location_and_work_model': genai.types.Schema(
            type=genai.types.Type.OBJECT,
            description="Where the role is based and the work model.",
            properties={
                'specification_level': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Indicates if specific location or remote status was found ('Specific Location / Remote Status Identified' or 'Not Specified')."
                    # Enum constraint handled by validation/prompt.
                ),
                'remote_status': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Primary work model regarding location ('Fully Remote', 'Remote (Region Specific)', 'Hybrid', 'Office-based', 'Not Specified')."
                    # Enum constraint handled by validation/prompt.
                ),
                'flexibility': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List flags indicating schedule flexibility (e.g., ['Flexible Schedule']).",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                ),
                'locations': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="Standardized list of mentioned Cities, Countries, or Regions (e.g., ['Lisbon', 'Portugal', 'EMEA', 'Global']). Sorted alphabetically.",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                )
            }
        ),

        # === Required Qualifications ===
        'required_qualifications': genai.types.Schema(
            type=genai.types.Type.OBJECT,
            description="Mandatory requirements for the role.",
            properties={
                'experience_years_min': genai.types.Schema(
                    type=genai.types.Type.INTEGER,
                    description="Minimum years of experience required."
                ),
                'experience_years_max': genai.types.Schema(
                    type=genai.types.Type.INTEGER,
                    description="Maximum years of experience specified."
                ),
                'experience_description': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Raw text describing the experience requirement."
                ),
                'education_requirements': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Required level of education or field of study text."
                ),
                'technical_skills': genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    description="Specific technical tools, platforms, languages, and concepts.",
                    properties={
                        'programming_languages': genai.types.Schema(
                            type=genai.types.Type.OBJECT,
                            description="Categorized required programming languages and related libraries/frameworks.",
                            properties={
                                'general_purpose': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="List of general-purpose backend/versatile languages."),
                                'scripting_frontend': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="List of scripting/frontend languages or frameworks."),
                                'query': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="List of query languages."),
                                'data_ml_libs': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="List of data/ML libraries or related frameworks."),
                                'platform_runtime': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="List of specific platforms/runtimes like .NET."),
                                'configuration': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="List of configuration/markup languages."),
                                'other_specialized': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="List of other specialized languages.")
                            }
                        ),
                        'cloud_platforms': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List of major cloud providers or core data platforms mentioned (e.g., AWS, Azure, GCP, Snowflake, Databricks).",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        ),
                        'cloud_services_tools': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List of specific cloud services or tools identified (e.g., S3, ADF, Glue, Lambda, GCS, BigQuery).",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        ),
                        'database_technologies': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List of specific database technologies or general concepts required.",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        ),
                        'data_architecture_concepts': genai.types.Schema(
                            type=genai.types.Type.OBJECT,
                            description="Categorized required knowledge of data architecture concepts.",
                            properties={
                                'data_modeling': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Data modeling techniques."),
                                'data_storage_paradigms': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Data storage concepts/systems."),
                                'etl_elt_pipelines': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Data movement/transformation concepts."),
                                'data_governance_quality': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Data governance/quality concepts."),
                                'architecture_patterns': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Data architecture patterns."),
                                'big_data_concepts': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Big Data specific concepts."),
                                'cloud_data_architecture': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Cloud-specific data architecture concepts."),
                                'ml_ai_data_concepts': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="ML/AI infrastructure/data concepts."),
                                'core_principles_optimization': genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING), description="Core design/optimization principles.")
                            }
                        ),
                        'etl_integration_tools': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List specific ETL, ELT, or Data Integration tools required.",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        ),
                        'data_visualization_bi_tools': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List specific Business Intelligence or Data Visualization tools required.",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        ),
                        'devops_mlops_ci_cd_tools': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List specific DevOps, MLOps, CI/CD, IaC, or Monitoring tools required.",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        ),
                        'orchestration_workflow_tools': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List specific workflow orchestration tools required.",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        ),
                        'other_tools': genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="List other relevant tools not fitting neatly into the above categories (e.g., IDEs, Data Catalogs, Vector DBs).",
                            items=genai.types.Schema(type=genai.types.Type.STRING)
                        )
                    }
                ),
                'methodologies_practices': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List required development methodologies or practices (e.g., 'Agile Principles', 'Scrum', 'TDD').",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                ),
                'soft_skills_keywords': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List required soft skills or general keywords (e.g., 'Communication', 'Teamwork', 'English').",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                )
            }
        ),

        # === Preferred Qualifications ===
        'preferred_qualifications': genai.types.Schema(
            type=genai.types.Type.OBJECT,
            description="Nice-to-have skills and qualifications.",
            properties={
                'skills_keywords': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List of preferred skills, tools, languages, or concepts (e.g., 'French', 'Certifications').",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                ),
                'other_notes': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Any other text describing preferred qualifications."
                )
            }
        ),

        # === Role Context ===
        'role_context': genai.types.Schema(
            type=genai.types.Type.OBJECT,
            description="Information about the role's interactions and scope.",
            properties={
                'collaboration_with': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List of teams or roles this position collaborates with (e.g., 'Stakeholders', 'Data Scientists').",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                ),
                'team_structure': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Description of the team structure or context."
                ),
                'project_scope': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Description of the type or scope of projects involved."
                ),
                'key_responsibilities': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List of key tasks and responsibilities mentioned.",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                )
            }
        ),

        # === Benefits ===
        'benefits': genai.types.Schema(
            type=genai.types.Type.OBJECT,
            description="Perks and benefits offered.",
            properties={
                'training_development': genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Description of training and development opportunities."
                ),
                'learning_platforms': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List specific learning platforms mentioned by name (e.g., 'Udemy', 'LinkedIn Learning').",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                ),
                'paid_time_off_days': genai.types.Schema(
                    type=genai.types.Type.INTEGER,
                    description="Specific number of paid time off days mentioned."
                ),
                'other_benefits_keywords': genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    description="List keywords for other benefits (e.g., 'Health Insurance', 'Meal Allowance', 'Well-being Program').",
                    items=genai.types.Schema(type=genai.types.Type.STRING)
                )
            }
        )
    }
)
```

#### Example output

```json
{
  "job_summary": {
    "role_title": "DevOps Engineer",
    "role_objective": "To help scale, automate, and optimize our cloud infrastructure. Responsible for designing, implementing, and maintaining CI/CD pipelines, managing Azure cloud infrastructure, and ensuring smooth deployments using Terraform and GitHub Actions.",
    "role_seniority": "Mid-Level",
    "visa_sponsorship": null
  },
  "company_information": {
    "company_type": "Software Product / SaaS",
    "company_values_keywords": [
      "Agile",
      "Collaboration",
      "Fun Environment",
      "Future-Oriented",
      "Growth",
      "Innovation"
    ]
  },
  "location_and_work_model": {
    "specification_level": "Specific Location / Remote Status Identified",
    "remote_status": "Hybrid",
    "flexibility": [
      "Flexible Schedule"
    ],
    "locations": []
  },
  "required_qualifications": {
    "experience_years_min": null,
    "experience_years_max": null,
    "experience_description": null,
    "education_requirements": null,
    "technical_skills": {
      "programming_languages": {
        "general_purpose": [
          "Python"
        ],
        "scripting_frontend": [
          "Bash / Shell Scripting",
          "PowerShell"
        ],
        "query": [],
        "data_ml_libs": [],
        "platform_runtime": [],
        "configuration": [],
        "other_specialized": []
      },
      "cloud_platforms": {
        "iaas_paas_providers": [
          "Microsoft Azure"
        ],
        "specific_cloud_services": [
          "Azure App Services",
          "Azure DevOps",
          "Azure Kubernetes Service (AKS)",
          "Azure Networking",
          "Azure Storage"
        ],
        "containerization_orchestration": [
          "Docker",
          "Kubernetes"
        ],
        "serverless_faas": [],
        "database_storage_services": [],
        "networking_security_services": [
          "Azure Firewall",
          "Azure RBAC",
          "Azure VPN Gateway",
          "Azure Virtual Network (vNET)"
        ],
        "monitoring_logging_services": [
          "Azure Monitor"
        ],
        "data_analytics_ai_ml_services": [],
        "iot_other_specialized_services": []
      },
      "data_storage_databases": {
        "relational_sql_databases": [],
        "nosql_databases": [],
        "data_warehouses_lakes": [],
        "search_engines": [],
        "message_queues_streaming": [],
        "file_object_storage": [
          "Azure Blob Storage / Azure Files"
        ],
        "graph_databases": [],
        "time_series_databases": [],
        "vector_databases": [],
        "other_db_storage_systems": []
      },
      "data_processing_frameworks": [],
      "operating_systems": [
        "Linux"
      ],
      "foundational_concepts": [
        "Networking Concepts",
        "Security Concepts"
      ],
      "data_architecture_concepts": {
        "data_modeling": [],
        "data_storage_paradigms": [],
        "etl_elt_pipelines": [],
        "data_governance_quality": [],
        "architecture_patterns": [],
        "big_data_concepts": [],
        "cloud_data_architecture": [],
        "ml_ai_data_concepts": [],
        "core_principles_optimization": []
      },
      "etl_integration_tools": [],
      "data_visualization_bi_tools": [],
      "devops_mlops_ci_cd_tools": [
        "Azure DevOps",
        "Docker",
        "GitHub Actions",
        "Grafana",
        "Kubernetes",
        "Prometheus",
        "Terraform"
      ],
      "orchestration_workflow_tools": [],
      "other_tools": []
    },
    "methodologies_practices": [
      "DevOps Culture/Practices"
    ],
    "soft_skills_keywords": [
      "Collaboration"
    ]
  },
  "preferred_qualifications": {
    "skills_keywords": [
      "SOC 2 Compliance",
      "Security Best Practices"
    ],
    "other_notes": "Experience with SOC 2 compliance and security best practices"
  },
  "role_context": {
    "collaboration_with": [
      "Software Engineers"
    ],
    "team_structure": null,
    "project_scope": "Scaling, automating, and optimizing cloud infrastructure; designing and maintaining CI/CD pipelines and Azure infrastructure.",
    "key_responsibilities": [
      "Infrastructure as Code (IaC): Design, implement, and maintain Azure infrastructure using Terraform.",
      "CI/CD Automation: Build and manage GitHub Actions workflows to streamline deployments and testing.",
      "Cloud Operations: Ensure high availability, security, and performance of our Azure-based environments.",
      "Monitoring & Observability: Set up logging, monitoring, and alerting solutions to proactively identify issues.",
      "Security & Compliance: Implement best practices for identity management, role-based access control (RBAC), and security policies in Azure.",
      "Collaboration: Work closely with software engineers to improve deployment strategies and DevOps best practices.",
      "Troubleshooting & Optimization: Identify performance bottlenecks and optimize cloud resources."
    ]
  },
  "benefits": {
    "training_development": "Professional development opportunities for training to advance your career.",
    "learning_platforms": [],
    "paid_time_off_days": null,
    "other_benefits_keywords": [
      "Caring Benefits",
      "Competitive Salary",
      "Dental Insurance",
      "Extra Days Off",
      "Flexible Working Hours",
      "Health Insurance",
      "Partner Discounts",
      "Performance Bonus",
      "Perks Package",
      "Remote Work Options",
      "Team Building Activities"
    ]
  }
}
```

