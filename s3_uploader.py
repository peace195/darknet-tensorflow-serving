import s3fs
import os


# Functions to interact with s3
def upload_data_to_s3(local_path, s3_model_path):
    try:
        print('Uploading train from {} to s3.'.format(local_path))
        fs = s3fs.S3FileSystem()
        checkpoint = open(local_path, "rb")
        with fs.open(s3_model_path, "wb") as fobj:
            while True:
                buffer = checkpoint.read(1024)
                if not buffer:
                    break
                fobj.write(buffer)
        # print('Finish uploading.')
    except Exception as e:
        print('Error while uploading train to s3: ' + str(e))
        pass


def download_data_from_s3(s3_model_path, local_path):
    try:
        print('Downloading file from {} to local.'.format(s3_model_path))
        fs = s3fs.S3FileSystem()
        fs.get(s3_model_path, local_path)
        print('Finish downloading.')
    except Exception as e:
        print('Error while downloading train from s3: ' + str(e))
        pass


def check_if_file_exist_in_s3(file_name):
    fs = s3fs.S3FileSystem()
    return fs.exists(file_name)


pb_dir = [f for f in os.listdir('built_graph') if os.path.isfile(os.path.join('built_graph', f))]

# Upload train from local host to s3
# Change this directory to get the necessary train
local_host_parent_directory = 'built_graph/'
s3_parent_directory = 's3://tiki-bot/dev/built_graph/'
for i, pb_file in enumerate(pb_dir):
    print('Sample {}:'.format(i))
    upload_data_to_s3(local_host_parent_directory + pb_file,
                      s3_parent_directory + pb_file)

    if (check_if_file_exist_in_s3(s3_parent_directory + pb_file)):
        print('Sample {} uploaded successfully.'.format(i))
    else:
        print('Can''t upload sample {}.'.format(i))

# for i, variables_file in enumerate(variables_dir):
#     print('Sample {}:'.format(i))
#     upload_data_to_s3(local_host_parent_directory + '1/variables/' + variables_file,
#                       s3_parent_directory + '1/variables/' + variables_file)
