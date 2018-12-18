import boto3
import click
import sagemaker
from sagemaker import get_execution_role


@click.command()
@click.argument('s3_input_data')
def train_on_sagemaker(s3_input_data):
    boto_session = boto3.Session(profile_name="packt-sagemaker")
    session = sagemaker.Session(boto_session=boto_session)

    account = session.boto_session.client('sts').get_caller_identity()['Account']
    region = session.boto_session.region_name
    image = '{}.dkr.ecr.{}.amazonaws.com/ner-travel-queries:latest'.format(account, region)

    role = get_execution_role(session)
    estimator = sagemaker.estimator.Estimator(
        image,
        role,
        1,
        'ml.c4.2xlarge',
        output_path="s3://{}/output".format(session.default_bucket()),
        sagemaker_session=session
    )

    estimator.fit(s3_input_data, logs=False, wait=True)

    print(estimator.model_data)


if __name__ == '__main__':
    train_on_sagemaker()
