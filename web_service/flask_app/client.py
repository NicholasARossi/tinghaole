import requests

URL = "http://127.0.0.1/predict"
#URL = "http://0.0.0.0:8000/predict"


#URL = "asdfasd:5-0"

#URL = "http://0.0.0.0:8000/predict"
TEST_AUDIO_FILE_PATH ='../../data/ma_data/ma1_USER2.wav'


if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"audio_data" :(TEST_AUDIO_FILE_PATH,audio_file,"audio/mp3")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted tone is: {data['tone']}")
