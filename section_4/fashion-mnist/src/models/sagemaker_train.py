import os

import boto3
from torchvision import datasets, transforms
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    boto_session = boto3.Session(profile_name="packt-sagemaker")
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-fashion-mnist'

    role = sagemaker.get_execution_role(sagemaker_session)

    datasets.FashionMNIST('data', download=True, transform=transforms.Compose([
        transforms.ToTensor()  # Convert a PIL Image or numpy.ndarray to tensor, default : range [0, 255] -> [0.0,1.0].
    ]))

    inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
    print('input spec (in this case, just an S3 path): {}'.format(inputs))

    entry_point_path = os.path.join(
        DIR_PATH,
        "fashion_mnist.py"
    )
    estimator = PyTorch(entry_point=entry_point_path,
                        role=role,
                        sagemaker_session=sagemaker_session,
                        framework_version='1.0.0.dev',
                        train_instance_count=1,
                        train_instance_type='ml.m4.xlarge',
                        hyperparameters={
                            'epochs': 6,
                            'backend': 'gloo'
                        })

    hyperparameter_ranges = {
        'lr': ContinuousParameter(0.001, 0.1),
        'batch-size': CategoricalParameter([32, 64, 128, 256, 512])
    }

    objective_metric_name = 'average test loss'
    objective_type = 'Minimize'
    metric_definitions = [
        {
            'Name': 'average test loss',
            'Regex': 'Test set: Average loss: ([0-9\\.]+)'
        }
    ]

    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                max_jobs=3,
                                max_parallel_jobs=3,
                                objective_type=objective_type,
                                base_tuning_job_name="fashion-mnist"
                                )

    tuner.fit({'training': inputs})

    tuner.wait()

    print(tuner.best_training_job())
