import os
import logging
from dotenv import load_dotenv, find_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

# Configure terminal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_compute_cluster():
    """Authenticates with Azure ML and provisions an AML compute cluster."""
    
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

        compute_name = "cpu-cluster"

        # 3. Define the Compute Cluster
        logging.info(f"Defining compute cluster: {compute_name}...")
        compute_cluster = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="STANDARD_DS3_V2",           # Standard CPU node. Change to a GPU SKU like 'STANDARD_NC6s_v3' if needed.
            min_instances=0,                  # Scales down to 0 when idle to prevent incurring costs.
            max_instances=2,                  # Maximum number of nodes to scale out to.
            idle_time_before_scale_down=120,  # Wait time in seconds before scaling down an idle node.
            tier="Dedicated",                 # Use "LowPriority" for cheaper, interruptible nodes.
        )

        # 4. Create or update the compute resource in the workspace
        logging.info(f"Provisioning compute cluster '{compute_name}'. This may take a few minutes...")
        
        # We use begin_create_or_update for asynchronous operations, and .result() to wait for it to finish.
        created_compute = ml_client.compute.begin_create_or_update(compute_cluster).result()
        
        logging.info(f"Successfully provisioned compute cluster: {created_compute.name}")
        logging.info(f"Provisioning state: {created_compute.provisioning_state}")

    except Exception as e:
        logging.error(f"Failed to create compute cluster in Azure ML: {e}")

if __name__ == "__main__":
    create_compute_cluster()