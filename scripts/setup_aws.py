"""
AWS Infrastructure Setup Script
Provisions: IAM Role, S3 Bucket, SageMaker Domain, ECR Repository
"""

import boto3
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
BUCKET_NAME = os.getenv("S3_BUCKET", "pytorch-workshop-2026")
ROLE_NAME = "SageMakerWorkshopRole"
DOMAIN_NAME = "pytorch-workshop"
ECR_REPO = os.getenv("ECR_REPO", "pytorch-workshop")

iam = boto3.client("iam", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
ecr = boto3.client("ecr", region_name=REGION)


def create_iam_role():
    print("Creating IAM role...")
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker role for PyTorch workshop",
        )
        iam.attach_role_policy(
            RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        iam.attach_role_policy(
            RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
        )
        iam.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        )
        print(f"  Role created: {role['Role']['Arn']}")
        return role["Role"]["Arn"]
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=ROLE_NAME)
        print(f"  Role already exists: {role['Role']['Arn']}")
        return role["Role"]["Arn"]


def create_s3_bucket():
    print("Creating S3 bucket...")
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME, CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        s3.put_bucket_versioning(Bucket=BUCKET_NAME, VersioningConfiguration={"Status": "Enabled"})
        # Block public access
        s3.put_public_access_block(
            Bucket=BUCKET_NAME,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print(f"  Bucket created: s3://{BUCKET_NAME}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  Bucket already exists: s3://{BUCKET_NAME}")


def create_ecr_repository():
    print("Creating ECR repository...")
    try:
        repo = ecr.create_repository(
            repositoryName=ECR_REPO, imageScanningConfiguration={"scanOnPush": True}
        )
        print(f"  ECR repo: {repo['repository']['repositoryUri']}")
        return repo["repository"]["repositoryUri"]
    except ecr.exceptions.RepositoryAlreadyExistsException:
        repo = ecr.describe_repositories(repositoryNames=[ECR_REPO])
        uri = repo["repositories"][0]["repositoryUri"]
        print(f"  ECR repo already exists: {uri}")
        return uri


def get_default_vpc():
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]]
    return vpc_id, subnet_ids


def create_sagemaker_domain(role_arn):
    print("Creating SageMaker domain...")
    vpc_id, subnet_ids = get_default_vpc()
    try:
        domain = sm.create_domain(
            DomainName=DOMAIN_NAME,
            AuthMode="IAM",
            DefaultUserSettings={
                "ExecutionRole": role_arn,
                "JupyterServerAppSettings": {"DefaultResourceSpec": {"InstanceType": "system"}},
                "KernelGatewayAppSettings": {
                    "DefaultResourceSpec": {"InstanceType": "ml.g4dn.xlarge"}
                },
            },
            VpcId=vpc_id,
            SubnetIds=subnet_ids[:2],
        )
        domain_id = domain["DomainArn"].split("/")[-1]
        print(f"  Domain created: {domain_id}")
        return domain_id
    except sm.exceptions.ResourceInUse:
        domains = sm.list_domains()
        domain_id = domains["Domains"][0]["DomainId"]
        print(f"  Domain already exists: {domain_id}")
        return domain_id


def create_student_profiles(domain_id, count=20):
    print(f"Creating {count} student profiles...")
    for i in range(1, count + 1):
        name = f"student-{i:02d}"
        try:
            sm.create_user_profile(DomainId=domain_id, UserProfileName=name)
            print(f"  Created: {name}")
        except sm.exceptions.ResourceInUse:
            print(f"  Already exists: {name}")
        time.sleep(0.3)  # avoid throttling


def upload_notebooks(domain_id):
    print("Uploading notebooks to S3...")
    notebooks_dir = "../notebooks"
    for fname in os.listdir(notebooks_dir):
        if fname.endswith(".ipynb"):
            s3.upload_file(f"{notebooks_dir}/{fname}", BUCKET_NAME, f"notebooks/{fname}")
            print(f"  Uploaded: {fname}")


def set_billing_alarm():
    print("Setting billing alarm at $100...")
    cw = boto3.client("cloudwatch", region_name="us-east-1")
    sns = boto3.client("sns", region_name="us-east-1")

    topic = sns.create_topic(Name="workshop-billing-alert")
    topic_arn = topic["TopicArn"]

    cw.put_metric_alarm(
        AlarmName="pytorch-workshop-budget",
        MetricName="EstimatedCharges",
        Namespace="AWS/Billing",
        Statistic="Maximum",
        Period=86400,
        EvaluationPeriods=1,
        Threshold=100,
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=[topic_arn],
        Dimensions=[{"Name": "Currency", "Value": "USD"}],
    )
    print(f"  Billing alarm set at $100, SNS topic: {topic_arn}")


def main():
    print("=" * 50)
    print("  PyTorch AWS Workshop — Infrastructure Setup")
    print("=" * 50)

    role_arn = create_iam_role()
    create_s3_bucket()
    ecr_uri = create_ecr_repository()
    set_billing_alarm()

    domain_id = "skipped"
    try:
        domain_id = create_sagemaker_domain(role_arn)
        create_student_profiles(domain_id, count=20)
    except Exception as e:
        print(f"  Warning: SageMaker domain skipped ({e})")
        print("  (Domain is only needed for SageMaker Studio — training jobs will still work)")

    print("\n" + "=" * 50)
    print("Setup complete! Save these values in .env:")
    print(f"  SAGEMAKER_ROLE_ARN={role_arn}")
    print(f"  DOMAIN_ID={domain_id}")
    print(f"  ECR_URI={ecr_uri}")
    print("=" * 50)


if __name__ == "__main__":
    main()
