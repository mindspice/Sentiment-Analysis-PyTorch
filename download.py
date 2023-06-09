import os
import requests

def download_file(url, save_dir):
    file_name = url.split("/")[-1]
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, file_name), 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        print(f"File '{file_name}' downloaded successfully.")
    else:
        print(f"Failed to download file '{file_name}'.")

model = "https://f004.backblazeb2.com/file/school-bucket/sentiment_model.pt"
vocab = "https://f004.backblazeb2.com/file/school-bucket/vocab.pt"
download_file(model, "models/")
download_file(vocab, "models/")