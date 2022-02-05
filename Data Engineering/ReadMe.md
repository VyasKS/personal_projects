# Data engineering project
## _ETL pipelines using Apache Spark, Databricks_

This personal project contains basics of building an ETL pipeline (extract, transform & load) that is common in an enterprise data engineering. It contains the phases of data collection from raw sources(_a.k.a. data lake_) that contains csv and json formats of world development indicators dataset provided by the world bank and CO2 emissions dataset provided by the European Environment agency.

Key descriptions:
1. DATA LAKES: Storage repositories for large volumes of data. Certainly, one of the greatest features of this solution is the fact that you can store all your data in native format within it. It contains 3 layers.
2. LAYERING: Data Lakes are not single repositories and have the flexibility to divide them into separate layers. We can distinguish 3-5 layers that can be applied to most cases.
3. RAW LAYER: a.k.a. Ingestion Layer/Landing Area, because it is literally the sink of our Data Lake. The main objective is to ingest data into Raw as quickly and as efficiently as possible. To do so, data should remain in its native format. We don’t allow any transformations at this stage. With Raw, we can get back to a point in time, since the archive is maintained. No overriding is allowed, which means handling duplicates and different versions of the same data. Despite allowing the above, Raw still needs to be organized into folders. Best practice is to start with generic division: subject area/data source/object/year/month/day of ingestion/raw data. It is important to mention that end users shouldn’t be granted access to this layer. The data here is not ready to be used, it requires a lot of knowledge in terms of appropriate and relevant consumption. Raw layer is quite similar to the well-known DWH staging. Can have a standardized layer (optional) that hormonizes (makes data uniform) if required. Main objective is to improve performance.
4. CURATED LAYER: a.k.a. Cleansed layer/Conformed layer. Data transformed into consumable data sets, and it may be stored in files or tables. The purpose of the data and its structure at this stage is already known. You should expect cleansing and transformations before this layer. Also, denormalization and consolidation of different objects is common. Due to all of the above, this is the most complex part of the whole Data Lake solution. In regards to organizing your data, the structure is quite simple and straightforward. For example: Purpose/Type/Files. Usually, end users are granted access only to this layer. Can have application layer (to store surrogate keys shared among applications, row level security etc. specific requirements for access and also for downstream usage of machine learning tasks, this layer is required)
5. SERVING DATA LAYER: Used to process tables of data into KPI dashboards. End consumption is the main goal here (this is in progress & will update the notebook with visualized dashboards using [![Preset]()](https://preset.io/))

Workflow followed as follows:

- Created compute cluster on databricks platform with Spark v3
- Instantiated a notebook environment with all necessary dependencies for pyspark
- Pulled the data sets using a shell script from the urls provided above
- Created 3 separate directories for 3 layers of data lake 
    - Raw layer, Curated layer, Serving layer
- Created dataframes, read descriptive statistics
- Performed data cleaning in curated layer
- Created tables of cleaned data ready for serving


## Instantiation

Run the IPython notebook in a databricks cluster node with Spark version >3.0.
Alternatively you can use databricks offering as a PaaS. For example, Azure Databricks, a Platform as a Service (PaaS) that provides a unified data analysis system to organizations. Cloud-based big data solution used for processing and transforming massive quantities of data. Other similar options exist at AWS & GCP as well. For this notebook, a community edition of databricks' own server is enough.