# This script is used as CLI for run/build/deploy this service

import subprocess

import click



PROJECT_ID = "data-science-258408"
IMAGE_NAME = "linkedinfo-ml-model"
REGION = "us-central1"
IMAGE_BUILD = f"gcr.io/{PROJECT_ID}/{IMAGE_NAME}"
SERVICE_NAME = "linkedinfo-tag-pred"
MEM_SIZE = '1024'
CLOUD_PORT = '80'


@click.group()
def cli():
    pass


@cli.command()
@click.option('--local/--container', default=True)
def run(local: bool):
    if local:
        from webapp import app
        app.debug = True
        # app.run(host='0.0.0.0', threaded=True)
        app.run(host='0.0.0.0')
    else:
        run_container()

#         docker build --tag $ImageBuild . && docker run -it -p 5000:80 $ImageBuild
# gcloud builds submit --tag gcr.io/$ProjectID/$ImageName
@cli.command()
@click.option('--local/--cloud', default=True)
@click.option('--run', is_flag=True, default=False)
@click.option('--deploy', is_flag=True, default=False)
def build(local: bool, run: bool, deploy: bool):
    if local:
        build_local()
    else:
        build_gcloud()

    if run:
        run_container()

    if deploy:
        deploy_gcloud()


#        docker push $ImageBuild
#        gcloud run deploy $ServiceName --platform managed --region $Region --image $ImageBuild
@cli.command()
def deploy():
    deploy_gcloud()


def run_container():
    print('Running image')
    subprocess.run(['docker', 'run', '-it', '-p', '5000:80', IMAGE_BUILD])


def build_local():
    print('Building image locally')
    subprocess.run(['docker', 'build', '--tag', IMAGE_BUILD, '.'])


def build_gcloud():
    print('Building image in Google Cloud Build')
    subprocess.run(['gcloud', 'builds', 'submit',
                    '--tag', IMAGE_BUILD])


def deploy_gcloud():
    print('Deploying image')
    subprocess.run(['docker', 'push', IMAGE_BUILD])
    subprocess.run(['gcloud', 'run', 'deploy',
                    SERVICE_NAME, '--platform', 'managed', '--region', REGION,
                    '--image', IMAGE_BUILD, '--memory', MEM_SIZE, '--port',
                    CLOUD_PORT])


# MODEL_PATH = "models"
# BUCKET = "data-science-258408-skin-lesion-cls-models"
# MODEL_CLOUD = "models/dense161.pth.tar"
#
# # gsutil cp $ModelPath gs://$Bucket/$ModelCloud
# # https://storage.cloud.google.com/data-science-258408-skin-lesion-cls-models/models/dense161.pth.tar
# @cli.command()
# @click.argument('model', type=click.Path(exists=True))
# def upload(model: str):
#     pass


if __name__ == '__main__':
    cli()
