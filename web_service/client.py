import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH ='data/ma_data/ma1_FV1_MP3.mp3'


if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file" :(TEST_AUDIO_FILE_PATH,audio_file,"audio/mp3")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted tone is: {data['tone']}")
