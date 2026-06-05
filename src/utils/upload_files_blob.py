import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from dotenv import load_dotenv, find_dotenv

def load_environment_variables():
    # It is usually safer to just let find_dotenv() search for '.env' automatically, 
    # but specifying the path is fine if it is strictly enforced.
    env_path = find_dotenv('.env')
    if env_path:
        load_dotenv(env_path)
        print(f"Environment variables loaded from: {env_path}")
    else:
        print("Warning: No .env file found. Falling back to system environment variables.")

def upload_folder_to_storage(local_folder_path, connection_string, container_name):
    print(f"Starting upload from local folder: {local_folder_path}")
    
    # Safety check before proceeding
    if not os.path.exists(local_folder_path):
        print(f"Error: The local folder '{local_folder_path}' does not exist.")
        return

    try:
        # Initialize the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Ensure the container exists (creates it if it doesn't)
        try:
            container_client = blob_service_client.create_container(container_name)
            print(f"Created new container: {container_name}")
        except ResourceExistsError:
            print(f"Container '{container_name}' already exists.")

        # Walk through the local directory recursively
        for root, dirs, files in os.walk(local_folder_path):
            for file_name in files:
                # 1. Get the full absolute local path to the file
                local_file_path = os.path.join(root, file_name)
                
                # 2. Determine the blob name
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                blob_name = relative_path.replace(os.path.sep, '/')
                
                # 3. Get the client for this specific blob
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                
                # 4. Upload the file
                print(f"Uploading: {blob_name} ...")
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                    
        print("\nAll files uploaded successfully!")

    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    load_environment_variables()
    
    # --- CONFIGURATION ---
    # Removed the leading slash so it looks in the current working directory
    LOCAL_FOLDER_PATH = "deploy_model/azure-files" 
    CONTAINER_NAME = "azure-files" 
    
    CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    
    # Safeguard: Check if the connection string exists before slicing or running
    if CONNECTION_STRING:
        print(f"Using connection string: {CONNECTION_STRING[:50]}...")  
        # Uncommented the function call to actually run the upload
        upload_folder_to_storage(LOCAL_FOLDER_PATH, CONNECTION_STRING, CONTAINER_NAME)
    else:
        print("Error: AZURE_STORAGE_CONNECTION_STRING is missing. Cannot proceed with upload.")