"""
Cleanup Script — Run after workshop to avoid ongoing charges
"""

import boto3
import os
import time
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET", "pytorch-workshop-2026")
DOMAIN_ID = os.getenv("DOMAIN_ID")
ENDPOINT_NAME = "pytorch-workshop-endpoint"

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.resource("s3", region_name=REGION)


def delete_endpoints():
    print("Deleting SageMaker endpoints...")
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"  Deleted: {ENDPOINT_NAME}")
    except sm.exceptions.ClientError:
        print("  No active endpoints found.")


def delete_student_profiles():
    print("Deleting student profiles...")
    if not DOMAIN_ID:
        print("  DOMAIN_ID not set in .env — skipping")
        return
    for i in range(1, 21):
        name = f"student-{i:02d}"
        try:
            sm.delete_user_profile(DomainId=DOMAIN_ID, UserProfileName=name)
            print(f"  Deleted: {name}")
            time.sleep(0.5)
        except Exception:
            pass


def delete_sagemaker_domain():
    print("Deleting SageMaker domain...")
    if not DOMAIN_ID:
        print("  DOMAIN_ID not set — skipping")
        return
    try:
        sm.delete_domain(DomainId=DOMAIN_ID, RetentionPolicy={"HomeEfsFileSystem": "Delete"})
        print(f"  Deleted domain: {DOMAIN_ID}")
    except Exception as e:
        print(f"  Error: {e}")


def empty_and_delete_bucket():
    print(f"Emptying and deleting s3://{BUCKET_NAME}...")
    try:
        bucket = s3.Bucket(BUCKET_NAME)
        bucket.object_versions.delete()
        bucket.delete()
        print(f"  Deleted: s3://{BUCKET_NAME}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    print("=" * 50)
    print("  PyTorch AWS Workshop — Cleanup")
    print("=" * 50)
    confirm = input("This will DELETE all workshop resources. Type 'yes' to continue: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    delete_endpoints()
    delete_student_profiles()
    delete_sagemaker_domain()
    empty_and_delete_bucket()

    print("\nCleanup complete. Check AWS console to verify no resources remain.")


if __name__ == "__main__":
    main()
