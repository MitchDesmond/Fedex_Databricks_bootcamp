# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://images.app.goo.gl/YVCzoe229tSUYAxU7" width="900"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Keyboard Shortcut
# MAGIC **For full list of keyboard shortcuts click help > keyboard shortcuts**
# MAGIC
# MAGIC - Ctrl+Option+F	:	Find and Replace
# MAGIC - ⌘+Shift+F	:	Format code
# MAGIC - **Shift+Enter	:	Run command and move to next cell**
# MAGIC - Option+Enter	:	Run command and insert new cell below
# MAGIC - Ctrl+Enter	:	Run command (do not go to next cell)
# MAGIC - Shift+Option+Enter	:	Run all commands
# MAGIC - ⌘+F8	:	Run all above commands (exclusive)
# MAGIC - ⌘+F10	:	Run all below commands (inclusive)
# MAGIC - Shift+Ctrl+Enter	:	Run selected text
# MAGIC - Option+Up/Down	:	Move to previous/next cell
# MAGIC - Ctrl+Option+P	:	Insert a cell above
# MAGIC - Ctrl+Option+N	:	Insert a cell below
# MAGIC - Ctrl+Option+-	:	Split a cell at cursor
# MAGIC - Ctrl+Option+Up	:	Move a cell up
# MAGIC - Ctrl+Option+Down	:	Move a cell down
# MAGIC - Ctrl+Option+M	:	Toggle comments panel
# MAGIC - Ctrl+Option+D	:	Delete current cell
# MAGIC - Up	:	Move up or to previous cell
# MAGIC - Down	:	Move down or to next cell
# MAGIC - Ctrl+Space	:	Autocomplete
# MAGIC - ⌘+Z	:	Undo typing
# MAGIC - ⌘+Shift+Z	:	Redo typing
# MAGIC - ⌘+/	:	Toggle line comment
# MAGIC - ⌘+K,⌘+C	:	Add line comment
# MAGIC - ⌘+K,⌘+U	:	Remove line comment
# MAGIC - ⌘+Click	:	Select multiple cells

# COMMAND ----------

df_airlines = spark.read.option("header",True).csv("/databricks-datasets/asa/airlines/2008.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Magic Commands
# MAGIC Databricks cotains several different magic commands.
# MAGIC
# MAGIC #### Mix Languages
# MAGIC You can override the default language by specifying the language magic command `%<language>` at the beginning of a cell. The supported magic commands are: 
# MAGIC * `%python`
# MAGIC * `%r`
# MAGIC * `%scala`
# MAGIC * `%sql`
# MAGIC
# MAGIC #### Auxiliary Magic Commands
# MAGIC * `%sh`: Allows you to run shell code in your notebook. To fail the cell if the shell command has a non-zero exit status, add the -e option. This command runs only on the Apache Spark driver, and not the workers. 
# MAGIC * `%fs`: Allows you to use dbutils filesystem commands.
# MAGIC * `%md`: Allows you to include various types of documentation, including text, images, and mathematical formulas and equations.
# MAGIC
# MAGIC #### %run Magic Command
# MAGIC The `%run` command allows you to include another notebook within a notebook. You can use `%run` to modularize your code, for example by putting supporting functions in a separate notebook. You can also use it to concatenate notebooks that implement the steps in an analysis. When you use %run, the called notebook is immediately executed and the functions and variables defined in it become available in the calling notebook.
# MAGIC
# MAGIC ⚠️`%run` must be in a cell by itself, because it runs the entire notebook inline.
# MAGIC
# MAGIC More information on Notebook workflows can be found [here](https://docs.databricks.com/notebooks/notebook-workflows.html).
# MAGIC
# MAGIC #### Other Magic Commands
# MAGIC * `%tensorboard`: starts a TensorBoard server and embeds the TensorBoard user interface inside the Databricks notebook for data scientists and machine learning engineers to visualize and debug their machine learning projects. More info [here](https://databricks.com/blog/2020/08/25/tensorboard-a-new-way-to-use-tensorboard-on-databricks.html)
# MAGIC * `%pip`: Allows you to easily customize and manage your Python packages on your cluster

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM csv.`/databricks-datasets/asa/airlines/2008.csv`

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Data
# MAGIC   
# MAGIC In this section, you download a dataset from the web and upload it to Databricks File System (DBFS).
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

# If you have the File > Upload Data menu option, follow the instructions in the previous cell to upload the data from your local machine.
# The generated code, including the required edits described in the previous cell, is shown here for reference.

import pandas as pd

# In the following lines, replace <username@...> with your username.
white_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username@...>/winequality_white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username@....>/winequality_red.csv", sep=';')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Temporary Views
# MAGIC Temporary views are session-scoped and are dropped when session ends because it skips persisting the definition in the underlying metastore.  These are a great way to simplify SQL queries, swtich easily between languages to perform quick analysis, develop a visualization, etc.  Note: These do not help performance as they are lazily executed
# MAGIC
# MAGIC Creating a temporary view:
# MAGIC * python: `df.createOrReplaceTempView("<NAME>")`
# MAGIC * R (SparkR): `createOrReplaceTempView(df, "<NAME>")`
# MAGIC * SQL: `CREATE [ OR REPLACE ] [ [ GLOBAL ] TEMPORARY ] VIEW [ IF NOT EXISTS ] view_identifier
# MAGIC     create_view_clauses AS query`
# MAGIC
# MAGIC ⚠️ **Note**: These do not help performance as they are lazily executed

# COMMAND ----------

df_airlines.createOrReplaceTempView("airlines")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   UniqueCarrier as airline, ROUND(AVG(AirTime),2) as avg_airTime, ROUND(AVG(ArrDelay),2) as avg_arrTime
# MAGIC FROM airlines
# MAGIC GROUP BY UniqueCarrier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Function
# MAGIC Databricks supports various types of visualizations out of the box using the `display` function.
# MAGIC
# MAGIC #### DataFrames
# MAGIC The easiest way to create a Spark DataFrame visualization in Databricks is to call `display(<dataframe-name>)`.  `Display` also supports Pandas DataFrames.
# MAGIC
# MAGIC 💡If you see `OK` with no rendering after calling the `display` function, mostly likely the DataFrame or collection you passed in is empty.
# MAGIC
# MAGIC #### Images
# MAGIC display renders columns containing image data types as rich HTML. display attempts to render image thumbnails for DataFrame columns matching the Spark ImageSchema. Thumbnail rendering works for any images successfully read in through the spark.read.format('image') function. More info [here](https://docs.databricks.com/notebooks/visualizations/index.html#images).
# MAGIC
# MAGIC #### Visualizations
# MAGIC The display function supports a rich set of plot types that can be configured by clicking the bar chart icon ![bar](https://docs.databricks.com/_images/chart-button.png):
# MAGIC
# MAGIC ![charts](https://docs.databricks.com/_images/display-charts.png)

# COMMAND ----------

display(df_airlines)

# COMMAND ----------

display(df_airlines.groupBy('UniqueCarrier').count())

# COMMAND ----------

# MAGIC %md
# MAGIC 💡Multiple `display` functions can be called in the same cell and the results will be rendered

# COMMAND ----------

print("American Airlines Flights:")
display(df_airlines.filter("UniqueCarrier == 'AA'"))
print("Delta Airlines Flights:")
display(df_airlines.filter("UniqueCarrier == 'DL'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Cursor
# MAGIC Multi-Cursor can be activated with Command + mouse left click (ctrl + alt + left click on window) on all the lines that you want to select to edit multiple lines at once

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ActualElapsedTime,
# MAGIC   AirTime,
# MAGIC   ArrDelay,
# MAGIC   DepDelay
# MAGIC FROM airlines

# COMMAND ----------

# MAGIC %md
# MAGIC ## Format SQL Code
# MAGIC Databricks provides tools that allow you to format SQL code in notebook cells quickly and easily. These tools reduce the effort to keep your code formatted and help to enforce the same coding standards across your notebooks.
# MAGIC
# MAGIC You can trigger the formatter in the following ways:
# MAGIC * Keyboard shortcut: Press **Cmd+Shift+F**.
# MAGIC * Command context menu: Select **Format SQL** in the command context drop-down menu of a SQL cell. This item is visible only in SQL notebook cells and those with a `%sql` language magic.
# MAGIC ![Format SQL](https://docs.databricks.com/_images/notebook-formatsql-cmd-context.png)

# COMMAND ----------

# MAGIC %md
# MAGIC SELECT UniqueCarrier as airline, ROUND(AVG(AirTime),2) as avg_airTime, ROUND(AVG(ArrDelay),2) as avg_arrDelay FROM airlines GROUP BY UniqueCarrier

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   UniqueCarrier as airline,
# MAGIC   ROUND(AVG(AirTime), 2) as avg_airTime,
# MAGIC   ROUND(AVG(ArrDelay), 2) as avg_arrDelay
# MAGIC FROM
# MAGIC   airlines
# MAGIC GROUP BY
# MAGIC   UniqueCarrier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Do's and Don'ts
# MAGIC Below are just a couple of tips for how to improve performance
# MAGIC * Use Spark SQL functions
# MAGIC * Don’t use Python UDFs (use Pandas UDFs)
# MAGIC * Avoid .toPandas() on large datasets
# MAGIC * Avoid .collect()
# MAGIC * Avoid for loops or row-by-row operations
# MAGIC * Checkpoint large joins, or cache
# MAGIC * Split some jobs which have very different characteristics
# MAGIC * Size your clusters - check the Spark UI
# MAGIC * Avoid disk spill
# MAGIC * Benchmark your application (CPU vs memory)
# MAGIC * Use latest Databricks Runtime Versions
# MAGIC * Run optimize (or enable auto-optimize) on Delta Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Commands
# MAGIC There are many table commands that can be used to describe databases and tables.
# MAGIC #### Show Tables
# MAGIC Returns all the tables for an optionally specified database. Additionally, the output of this statement may be filtered by an optional matching pattern. If no database is specified then the tables are returned from the current database.
# MAGIC
# MAGIC `SHOW TABLES [ { FROM | IN } database_name ] [ LIKE regex_pattern ]`
# MAGIC #### Describe Table
# MAGIC Returns the basic metadata information of a table. The metadata information includes column name, column type and column comment. Optionally you can specify a partition spec or column name to return the metadata pertaining to a partition or column respectively.
# MAGIC
# MAGIC `{ DESC | DESCRIBE } TABLE [EXTENDED] [ format ] table_identifier [ partition_spec ] [ col_name ]`
# MAGIC
# MAGIC ###### Extended
# MAGIC Display detailed information about the specified columns
# MAGIC ### Delta Specific Commands
# MAGIC Delta Tables support a number of utility commands.  More information on these commands can be found [here](https://docs.databricks.com/delta/delta-utility.html#table-utility-commands).
# MAGIC #### Delta Table History
# MAGIC You can retrieve information on the operations, user, timestamp, and so on for each write to a Delta table by running the `DESCRIBE HISTORY` command. The operations are returned in reverse chronological order. By default table history is retained for 30 days.
# MAGIC #### Restore a Delta table to an earlier state
# MAGIC You can restore a Delta table to its earlier state by using the `RESTORE` command. A Delta table internally maintains historic versions of the table that enable it to be restored to an earlier state. A version corresponding to the earlier state or a timestamp of when the earlier state was created are supported as options by the `RESTORE` command.
# MAGIC #### Retrieve Delta Table Details
# MAGIC You can retrieve detailed information about a Delta table (for example, number of files, data size) using `DESCRIBE DETAIL`.
# MAGIC #### Convert a Parquet table to a Delta table
# MAGIC Convert a Parquet table to a Delta table in-place. This command lists all the files in the directory, creates a Delta Lake transaction log that tracks these files, and automatically infers the data schema by reading the footers of all Parquet files.
# MAGIC
# MAGIC `CONVERT TO DELTA parquet.<path-to-table>`

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN dbacademy

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE dbacademy.sales_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE EXTENDED dbacademy.sales_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL dbacademy.sales_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY dbacademy.sales_gold

# COMMAND ----------

userhome = "/dbfs/dbacademy/will_block"
deltaDataPath = userhome + "/delta/airlines/"

# COMMAND ----------

# write to delta dataset
df_airlines.write.mode("overwrite").format("delta").save(deltaDataPath)

# COMMAND ----------

spark.sql("""
  DROP TABLE IF EXISTS airlines_data_delta
""")
spark.sql("""
  CREATE TABLE airlines_data_delta 
  USING DELTA 
  LOCATION '{}' 
""".format(deltaDataPath))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   UniqueCarrier as airline, ROUND(AVG(AirTime),2) as avg_airTime, ROUND(AVG(ArrDelay),2) as avg_arrTime
# MAGIC FROM airlines_data_delta
# MAGIC GROUP BY UniqueCarrier

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL airlines_data_delta

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY airlines_data_delta

# COMMAND ----------

# MAGIC %fs ls dbfs:/dbfs/dbacademy/will_block/delta/airlines

# COMMAND ----------

# MAGIC %fs ls dbfs:/dbfs/dbacademy/will_block/delta/airlines/_delta_log/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Jobs
# MAGIC **[Job Clusters](https://docs.databricks.com/jobs.html)** are dedicated clusters that are created and started when you run a job and terminated immediately after the job completes. They are ideal for production-level jobs or jobs that are important to complete, because they provide a fully isolated environment. Jobs clusters offers the following benefits:
# MAGIC * Run on ephemeral clusters that are created for the job, and terminate on completion
# MAGIC * Pre-scheduled or submitted via J[obs API](https://docs.databricks.com/dev-tools/api/latest/jobs.html)
# MAGIC * Full history tracking of run details
# MAGIC * Great for isolation and debugging
# MAGIC * Set timeouts and retry
# MAGIC * Lower cost
# MAGIC
# MAGIC 💡 You are able to pass parameters to a job by leveraging [widgets](https://docs.databricks.com/notebooks/widgets.html)
# MAGIC
# MAGIC 💡 For critical workloads it is recommended to enable retries to account for service outages that occur on occasion

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Settings
# MAGIC Notebooks have a number of default settings that can be changed in the **Notebook Settings** menu.  Access the Notebook Settings page by selecting **![User Settings](https://docs.databricks.com/_images/account-icon.png)> User Settings > Notebook Settings**.
# MAGIC The default settings that can be updated in the Notebook Settings menu inlcude:
# MAGIC * When you run a cell, the notebook automatically attaches to a running cluster without prompting.
# MAGIC * When you press **shift+enter**, the notebook auto-scrolls to the next cell if the cell is not visible.
# MAGIC * Databricks Advisor
# MAGIC * Warning when deleting cells
# MAGIC * Dark Mode
# MAGIC
# MAGIC #### Databricks Advisor
# MAGIC Databricks Advisor automatically analyzes commands every time they are run and displays appropriate advice in the notebooks. The advice notices provide information that can assist you in improving the performance of workloads, reducing costs, and avoiding common mistakes.
# MAGIC ##### View Advise
# MAGIC A blue box with a lightbulb icon signals that advice is available for a command. The box displays the number of distinct pieces of advice.
# MAGIC
# MAGIC ![Databricks Advisor](https://docs.databricks.com/_images/advice-collapsed.png)
# MAGIC
# MAGIC Click the lightbulb to expand the box and view the advice. One or more pieces of advice will become visible.
# MAGIC
# MAGIC ![Tip](https://docs.databricks.com/_images/advice-expanded.png)
# MAGIC
# MAGIC Click the **Learn more** link to view documentation providing more information related to the advice.
# MAGIC
# MAGIC Click the **Don’t show me this again** link to hide the piece of advice. The advice of this type will no longer be displayed. This action can be reversed in Notebook Settings.
# MAGIC
# MAGIC Click the lightbulb again to collapse the advice box.
