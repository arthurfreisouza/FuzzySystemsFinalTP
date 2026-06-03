from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Connect to the AzureML workspace
subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace = "<AZUREML_WORKSPACE_NAME>"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# Set the version number of the data asset (for example: '1')
VERSION = "<VERSION>"

# Set the path, supported paths include:
# local: './<path>/<file>' (this will be automatically uploaded to cloud storage)
# blob:  'wasbs://<container_name>@<account_name>.blob.core.windows.net/<path>/<file>'
# ADLS gen2: 'abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>/<file>'
# Datastore: 'azureml://datastores/<data_store_name>/paths/<path>/<file>'
path = "<SUPPORTED PATH>"

# Define the Data asset object
my_data = Data(
    path=path,
    type=AssetTypes.URI_FILE,
    description="<ADD A DESCRIPTION HERE>",
    name="<NAME OF DATA ASSET>",
    version=VERSION,
)

# Create the data asset in the workspace
ml_client.data.create_or_update(my_data)