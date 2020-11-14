from oauth2client.client import GoogleCredentials
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from google.colab import auth


if __name__ == "__main__":

    auth.authenticate_user()

    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    bio_bert = ("1GJpGjQj6aZPV-EfbiQELpBkvlGtoKiyA", 'biobert_large_v1.1_pubmed.tar.gz')
    bio_nlp_vec = ("0BzMCqpcgEJgiUWs0ZnU0NlFTam8", 'bio_nlp_vec.tar.gz')

    for file_name, file_id in [bio_nlp_vec, bio_bert]:
        print(f'downloading {file_name}...')
        downloaded = drive.CreateFile({'id': file_id})
        downloaded.GetContentFile(file_name)
