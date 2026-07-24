import os
import gdown
import requests


def download_from_gdrive(file_id, directory):
    os.makedirs(directory, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=directory, quiet=False)


def download_from_url(url, directory):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, url.split("/")[-1])

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


try:
    download_from_gdrive('170Qldnn28IwnVm9CQEq1AZhVsK7PJ0Xz', 'training_artifacts/ffhq')
    download_from_gdrive('1QESywJW8N-g3n0Csy0clztuJV99g8pRm', 'training_artifacts/ffhq')
    download_from_gdrive('18BzFYKS3icFd1DQKKTeje7CKbEKXPVug', 'training_artifacts/ffhq')
except Exception:
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_submitted.pth', 'training_artifacts/ffhq')
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_194.pth', 'training_artifacts/ffhq')
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth', 'training_artifacts/ffhq')


try:
    download_from_gdrive('1T4gkE7-COHpX38qPwjMYO-xU-SrY_aT4', 'training_artifacts/celeba')
except Exception:
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba/model_final.pth', 'training_artifacts/celeba')


try:
    download_from_gdrive('1gmYbc6Z8qJHJwICYDsB4aBMxXjnKeXA_', 'training_artifacts/bedroom')
except Exception:
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/bedroom/model_final.pth', 'training_artifacts/bedroom')


try:
    download_from_gdrive('1ihJvp8iJWcLxTIjkV5cyA7l9TrxlUPkG', 'training_artifacts/celeba-hq256')
    download_from_gdrive('1gFQsGCNKo-frzKmA3aCvx07ShRymRIKZ', 'training_artifacts/celeba-hq256')
except Exception:
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_262r.pth', 'training_artifacts/celeba-hq256')
    download_from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_580r.pth', 'training_artifacts/celeba-hq256')