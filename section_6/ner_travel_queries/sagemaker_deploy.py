import boto3
import click
import sagemaker
from sagemaker import get_execution_role


def _construct_image_location(boto_session, image_name):
    account = boto_session.client('sts').get_caller_identity()['Account']
    region = boto_session.region_name

    return '{account}.dkr.ecr.{region}.amazonaws.com/{image}'.format(
        account=account,
        region=region,
        image=image_name
    )


@click.command()
@click.argument('s3_model_location')
def deploy_on_sagemaker(s3_model_location):
    boto_session = boto3.Session(profile_name="packt-sagemaker")
    session = sagemaker.Session(boto_session=boto_session)
    image_name = 'ner-travel-queries'

    image = _construct_image_location(boto_session, image_name)

    role = get_execution_role(session)
    model = sagemaker.Model(
        model_data=s3_model_location,
        image=image,
        role=role,
        sagemaker_session=session
    )

    model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium'
    )


if __name__ == '__main__':
    deploy_on_sagemaker()
