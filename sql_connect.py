import mysql.connector
import datetime
import requests
import base64
import json
import timedelta

lastsentdate = None

def upload():
    with open('config.json') as json_file:
        parsed_config = json.load(json_file)

        now = datetime.datetime.now()
        last = parsed_config["last-sent-date"]

        if (now > datetime.timedelta(seconds=10) + datetime.datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')):
            #past the limit to success

            parsed_config["last-sent-date"] = str(datetime.datetime.now())

            print("Sending data to server...")
            mydb = mysql.connector.connect(
                host="65.19.141.67",
                user="xtornado_no",
                passwd="newhacks2020",
                database="xtornado_no"
            )

            mycursor = mydb.cursor()

            date = str(datetime.datetime.now())

            client_id = '73b8a64c8ae54da'

            headers = {"Authorization": "Client-ID 73b8a64c8ae54da"}

            api_key = 'c49490c1a0e6130ab11ee00026ae634a042b551e'

            url = "https://api.imgur.com/3/upload.json"

            j1 = requests.post(
                url, 
                headers = headers,
                data = {
                    'key': api_key, 
                    'image': base64.b64encode(open('images/temp.jpg', 'rb').read()),
                    'type': 'base64',
                    'name': date + '.jpg',
                    'title': 'Picture no. 1'
                }
            )

            loaded_json = json.loads(j1.text)

            link = loaded_json["data"]["link"]


            sql = "INSERT INTO alerts (image, time) VALUES (%s, %s)"
            val = (link, date)

            mycursor.execute(sql, val)

            mydb.commit()

            with open('config.json', 'w') as f:
                json.dump(parsed_config, f, indent=4)

            print(mycursor.rowcount, "record inserted.")
        else:
            print("Please wait before sending another record")