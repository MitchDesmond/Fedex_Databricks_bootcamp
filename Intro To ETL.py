# Databricks notebook source
# MAGIC %md # Intro to Databricks Export Transflorm Load (ETL)
# MAGIC
# MAGIC This tutorial covers the following steps:
# MAGIC - Import data from your local machine into the Databricks File System (DBFS)
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Update the dataframe and look for data quality issues
# MAGIC - Save the data to a databricks Delta file
# MAGIC
# MAGIC In this example, we will explore one of the Databricks sample datasets - Portugese "Vinho Verde" wine and the wine's physicochemical properties. 
# MAGIC
# MAGIC The example uses a dataset from the UCI Machine Learning Repository, presented in [*
# MAGIC Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC
# MAGIC The original version of this notebook can be found [here](https://www.databricks.com/notebooks/gallery/MLEndToEndExampleAWS.html)
# MAGIC
# MAGIC ### Setup
# MAGIC - This notebook requires Databricks Runtime 7.6+, which includes the latest MLFlow, along with other machine learning frameworks like sklearn, PyTorch, TensorFlow, XGBoost, etc. No need for you to install them.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create notebook paramaters with widgets
# MAGIC
# MAGIC Input widgets allow you to add parameters to your notebooks and dashboards. The widget API consists of calls to create various types of input widgets, remove them, and get bound values.
# MAGIC
# MAGIC If you are running Databricks Runtime 11.0 or above, you can also use ipywidgets in Databricks notebooks.
# MAGIC
# MAGIC Databricks widgets are best for:
# MAGIC
# MAGIC Building a notebook or dashboard that is re-executed with different parameters
# MAGIC
# MAGIC Quickly exploring results of a single query with different parameters
# MAGIC
# MAGIC To view the documentation for the widget API in Scala, Python, or R, use the following command: dbutils.widgets.help()

# COMMAND ----------

#optional: you can remove all widgests and start over
#Note: any paramater passed into a notebook from a job will overwrite existing values or defaults
#dbutils.widgets.removeAll()

# COMMAND ----------

#two types of widgets - dropdown or text
dbutils.widgets.dropdown("save_data_flag", "False", ["True", "False"], "Save Data Inventory?")
dbutils.widgets.text("save_data_location", "field_demos.mitch_desmond", "Save Data Location")

# COMMAND ----------

#save the input param to a varaiable
save_data_flag = dbutils.widgets.get("save_data_flag")
save_data_location = dbutils.widgets.get("save_data_location")

# COMMAND ----------

#note: you can also have dynamic values in a dropdown

df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/databricks-datasets/flights/departuredelays.csv")

origin_list = df.select("origin").distinct().rdd.flatMap(lambda x: x).collect()
dbutils.widgets.dropdown("origin", "ABE", [str(x) for x in origin_list])

# COMMAND ----------

# MAGIC %md ## Importing Data
# MAGIC   
# MAGIC In this section, you will load the dataset from the Databricks example datasets
# MAGIC
# MAGIC The below %fs command is used for file system actions. The below two commands have the same output but the second one is in python.

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/

# COMMAND ----------

# this command does the same action as the %fs above, but in python
display(dbutils.fs.ls('dbfs:/databricks-datasets'))

# COMMAND ----------

# display the files in the wine quality data set
display(dbutils.fs.ls('dbfs:/databricks-datasets/wine-quality/'))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC   **Note**: if you don't have to /dbfs/databricks-datasets/, you can load the dataset from the UI. Uncomment and run the cell below after uploading the file.
# MAGIC
# MAGIC download a dataset from the web and upload it to Databricks File System (DBFS).
# MAGIC
# MAGIC 1. Navigate to https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ and download both `winequality-red.csv` and `winequality-white.csv` to your local machine.
# MAGIC
# MAGIC 1. From this Databricks notebook, select *File* > *Upload Data*, and drag these files to the drag-and-drop target to upload them to the Databricks File System (DBFS). 
# MAGIC
# MAGIC 1. Click *Next*. Some auto-generated code to load the data appears. Select *pandas*, and copy the example code. 
# MAGIC
# MAGIC 1. Create a new cell, then paste in the sample code. It will look similar to the code shown in the following cell. Make these changes:
# MAGIC   - Pass `sep=';'` to `pd.read_csv`
# MAGIC   - Change the variable names from `df1` and `df2` to `white_wine` and `red_wine`, as shown in the following cell.

# COMMAND ----------

# # If you do not have access to the sample datasets, follow the instructions in the previous cell to upload the data from your local machine.
# # The generated code, including the required edits described in the previous cell, is shown here for reference.

# import pandas as pd

# # In the following lines, replace <username@...> with your username.
# white_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username@...>/winequality_white.csv", sep=';')
# red_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username@....>/winequality_red.csv", sep=';')

# COMMAND ----------

import pandas as pd

# In the following lines, replace <username@...> with your username.
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=';')
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=';')

# COMMAND ----------

#display the data in the white wine dataset
white_wine

# COMMAND ----------

red_wine

# COMMAND ----------

#other examples of reading sample data
#display(spark.read.json("dbfs:/databricks-datasets/nyctaxi/sample/json/"))
#spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/databricks-datasets/flights/departuredelays.csv").show()

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

df = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ##Data Visualization
# MAGIC
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.distplot(df.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

high_quality = (df.quality >= 7).astype(int)
df.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful in noticing correlations between features and a binary label.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in df.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=df[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocessing Data
# MAGIC Prior to saving data, check for missing values and data quality issues

# COMMAND ----------

df.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

# MAGIC %md ## Save data as databricks delta file
# MAGIC

# COMMAND ----------

spark_df = spark.createDataFrame(df)

# COMMAND ----------

print("Delta table where data will be saved " + save_metadata_location)

# COMMAND ----------

df.write.mode("overwrite").saveAsTable(save_metadata_location + ".output_data")
