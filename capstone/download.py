import os
import requests

def download_file(url, save_dir):
    # Get file name from url
    file_name = url.split("/")[-1]

    # Send a GET request to the url
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Open the file in write mode
        with open(os.path.join(save_dir, file_name), 'wb') as file:
            # Write the contents of the response to the file
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        print(f"File '{file_name}' downloaded successfully.")
    else:
        print(f"Failed to download file '{file_name}'.")

# URLs of the files to download
model = "https://f004.backblazeb2.com/file/school-bucket/sentiment_model.pt"
vocab = "https://f004.backblazeb2.com/file/school-bucket/vocab.pt"

# Download files
download_file(model, "models/")
download_file(vocab, "models/")