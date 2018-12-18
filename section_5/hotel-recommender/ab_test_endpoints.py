import boto3
import click
import sagemaker
from sagemaker import get_execution_role, utils

from sagemaker_deploy import construct_image_location


def _create_model(
        s3_model_location,
        image,
        role,
        sagemaker_session,
        instance_type
):
    model_a = sagemaker.Model(
        model_data=s3_model_location,
        image=image,
        role=role,
        sagemaker_session=sagemaker_session
    )

    container_definition = model_a.prepare_container_def(
        instance_type
    )

    return sagemaker_session.create_model(
        name=utils.name_from_image(container_definition['Image']),
        role=role,
        container_defs=container_definition
    )


@click.command()
@click.argument('s3_model_location_a', type=click.Path())
@click.argument('s3_model_location_b', type=click.Path())
@click.argument('model_a_weight', type=int)
@click.argument('model_b_weight', type=int)
def main(
        s3_model_location_a,
        s3_model_location_b,
        model_a_weight,
        model_b_weight
):
    boto_session = boto3.Session(
        profile_name="packt-sagemaker"
    )
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session
    )

    image_name = 'hotel-recommender'
    image = construct_image_location(boto_session, image_name)
    role = get_execution_role(sagemaker_session)

    instance_type = 'ml.t2.medium'

    model_name_a = _create_model(
        s3_model_location=s3_model_location_a,
        image=image,
        role=role,
        sagemaker_session=sagemaker_session,
        instance_type=instance_type
    )

    model_name_b = _create_model(
        s3_model_location=s3_model_location_b,
        image=image,
        role=role,
        sagemaker_session=sagemaker_session,
        instance_type=instance_type
    )

    client = boto_session.client("sagemaker")
    client.create_endpoint_config(
        EndpointConfigName='hotelRecommenderEndPointConfig',
        ProductionVariants=[
            {
                'VariantName': 'model1',
                'ModelName': model_name_a,
                'InitialInstanceCount': 1,
                'InstanceType': instance_type,
                'InitialVariantWeight': model_a_weight
            },
            {
                'VariantName': 'model2',
                'ModelName': model_name_b,
                'InitialInstanceCount': 1,
                'InstanceType': instance_type,
                'InitialVariantWeight': model_b_weight
            },
        ]
    )

    client.create_endpoint(
        EndpointName='hotelRecommenderEndPoint',
        EndpointConfigName='hotelRecommenderEndPointConfig'
    )


if __name__ == '__main__':
    main()
