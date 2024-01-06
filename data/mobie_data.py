import requests
import os
url = "https://www.dropbox.com/s/739uo8ebwbic9kb/mobile_cleaned.csv"
destination_path = '/media/sahitya/BE60ABB360AB70B7/Scratches/data/mobile_data.csv'
response = requests.get(url)

if response.status_code == 200:
    with open(destination_path, "wb") as file:
        file.write(response.content)
    print("Download successful.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
