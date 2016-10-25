import os,sys,logging,time,shutil
from fabric.api import env,local,run,sudo,put,cd,lcd,puts,task,get,hide
from fabric.context_managers import cd
try:
    import inception
except ImportError:
    print "could not import main module limited to boostrap actions"
    pass

env.hosts = ['bohr-py1.s-9.us']


@task
def install_requirements():
    with cd(env.PROJECT_PATH):
        run("sudo pip install -r requirements.txt")


@task
def setup(project_path):
    """
    Sets up the environment. Needs to be the first step
    Args:
        project_path: the path to the tensorflow repo
    """
    env.PROJECT_PATH = project_path
    env.DATA_DIR = os.path.join(env.PROJECT_PATH, 'data')
    env.MODEL_DIR = os.path.join(env.PROJECT_PATH, 'models', 'inception')
    env.TRAIN_DIR = os.path.join(env.DATA_DIR, 'train')
    env.VALIDATION_DIR = os.path.join(env.DATA_DIR, 'validate')
    env.OUTPUT_DIRECTORY = os.path.join(env.DATA_DIR, 'output')
    env.LABELS_FILE = os.path.join(env.MODEL_DIR, 'output_labels.txt')
    env.TRAIN_EVAL_DIR = os.path.join(env.DATA_DIR, 'train_eval')
    env.VALIDATION_EVAL_DIR = os.path.join(env.DATA_DIR, 'validate_eval')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename='logs/fab.log',
        filemode='a'
    )


@task
def build_training_data(project_path):
    setup(project_path)
    with cd(os.path.join(env.MODEL_DIR)):
        run("su -c 'bazel build inception/build_image_data' srprd")
        bazel_command = 'bazel-bin/inception/build_image_data --train_directory="{}" --validation_directory="{}" --output_directory="{}" --labels_file="{}" --train_shards=128  --validation_shards=24  --num_threads=8'
        bazel_command = bazel_command.format(
            env.TRAIN_DIR, env.VALIDATION_DIR,
            env.OUTPUT_DIRECTORY, env.LABELS_FILE
        )
        run("su -c '%s' srprd" % bazel_command)


@task
def run_training_on_data(project_path):
    setup(project_path)
    with cd(env.PROJECT_PATH):
        run("su -c 'bazel build tensorflow/examples/image_retraining:retrain' srprd")
        # command = """
        # sudo bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir {} --bottleneck_dir {} --model_dir {} --output_graph {}
        # """.format(env.TRAIN_DIR, env.MODEL_DIR,
        #            env.MODEL_DIR, env.TRAIN_EVAL_DIR)
        # run(command)


@task
def train(project_path):
    setup(project_path)
    build_training_data(project_path)
    run_training_on_data(project_path)
