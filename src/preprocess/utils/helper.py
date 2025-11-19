import os
import zipfile
import logging
import requests
from tqdm import tqdm
from google.cloud import storage

logger = logging.getLogger("lazyobjectplacement")


def download_file(url, target_path):
    logger.info(f"Starting download from {url} to {target_path}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded file from {url} to {target_path}")
    else:
        logger.error(f"Failed to download file: {response.status_code}")

def unzip_file(zip_path, extract_to=None, remove_zip=False):
    if extract_to is None:
        extract_to = zip_path.replace('.zip', '')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    logger.info(f"Unzipped file {zip_path} to {extract_to}")

    if remove_zip:
        os.remove(zip_path)
        logger.info(f"Removed zip file {zip_path}")

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in tqdm(os.walk(folder_path), desc=f"Zipping {folder_path}"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

    logger.info(f"Zipped folder {folder_path} to {zip_path}")

def merge_folders(base_folder_path):
    folder_path_list = [os.path.join(base_folder_path, d) for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d))]
    for folder_path in folder_path_list:
        for file_name in os.listdir(folder_path):
            src_file_path = os.path.join(folder_path, file_name)
            dst_file_path = os.path.join(base_folder_path, file_name)

            os.rename(src_file_path, dst_file_path)
        logger.info(f"Merged files from {folder_path} to {base_folder_path}")

        os.rmdir(folder_path)
        logger.info(f"Removed empty folder {folder_path}")

def parse_image_id(mask_folder_path, set_name):
    mask_file_list = os.listdir(mask_folder_path)
    image_id_list = [set_name + "/" + mask_file.split("_")[0] for mask_file in mask_file_list]
    unique_image_id_list = list(set(image_id_list))
    return unique_image_id_list

def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    logger.info(f"File {file_path} uploaded to {destination_blob_name} in bucket {bucket_name}.")