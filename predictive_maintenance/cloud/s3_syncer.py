import os

class s3_syncer:
    def sync_folder_to_s3(self, folder: str, aws_bucket_url: str):
        """
        Syncs a local folder to an S3 bucket using AWS CLI.

        Args:
            folder (str): Local folder path to sync.
            aws_bucket_url (str): S3 bucket URL.
        """
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(command)

    def sync_folder_from_s3(self, aws_bucket_url: str, folder: str):
        """
        Syncs an S3 bucket to a local folder using AWS CLI.

        Args:
            aws_bucket_url (str): S3 bucket URL.
            folder (str): Local folder path to sync.
        """
        command = f"aws s3 sync {aws_bucket_url} {folder}"
        os.system(command)
        