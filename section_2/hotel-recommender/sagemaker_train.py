import boto3
import sagemaker
from sagemaker import get_execution_role


def train_on_sagemaker():
    boto_session = boto3.Session(profile_name="packt-sagemaker")
    session = sagemaker.Session(boto_session=boto_session)

    account = session.boto_session.client('sts').get_caller_identity()['Account']
    region = session.boto_session.region_name
    image = '{}.dkr.ecr.{}.amazonaws.com/hotel-recommender:latest'.format(account, region)

    role = get_execution_role(session)
    estimator = sagemaker.estimator.Estimator(
        image,
        role,
        1,
        'ml.c4.2xlarge',
        output_path="s3://{}/output".format(session.default_bucket()),
        sagemaker_session=session
    )

    local_data_directory = 'data/processed'
    s3_prefix = 'packt-sagemaker-hotel-recommender'

    data_location = session.upload_data(local_data_directory, key_prefix=s3_prefix)

    estimator.fit(data_location, logs=False, wait=True)

    print(estimator.model_data)


if __name__ == '__main__':
    train_on_sagemaker()
