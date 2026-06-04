import os
import logging
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Configure terminal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def register_data_asset():
    """Authenticates with Azure ML and registers a local CSV as a Data Asset."""
    
    # 1. Load environment variables
    load_dotenv(find_dotenv('.env'))
    
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace = os.getenv("AZURE_WORKSPACE_NAME")

    # Guard clause to catch missing credentials early
    if not all([subscription_id, resource_group, workspace]):
        logging.error("Missing one or more required Azure environment variables in .env.")
        return

    try:
        # 2. Authenticate and connect
        ml_client = MLClient(
            DefaultAzureCredential(), 
            subscription_id, 
            resource_group, 
            workspace
        )

        # 3. Define the Data asset
        my_data = Data(
            path="./data/wdbc.data",
            type=AssetTypes.URI_FILE,
            description="Cancer prediction dataset",
            name="dataset",
            version="1",
        )

        # 4. Create or update the data asset in the workspace
        logging.info(f"Registering data asset: {my_data.name} (v{my_data.version})...")
        created_asset = ml_client.data.create_or_update(my_data)
        logging.info(f"Successfully registered data asset with ID: {created_asset.id}")

    except Exception as e:
        logging.error(f"Failed to create data asset in Azure ML: {e}")

if __name__ == "__main__":
    register_data_asset()