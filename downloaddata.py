import os
import os.path
import sys
import requests
from retry import retry
import pandas as pd
from pandas.io.json import json_normalize
import warnings
from datetime import date


@retry(exceptions=(requests.exceptions.HTTPError,
                  requests.exceptions.ConnectionError,
                  requests.exceptions.Timeout,
                  requests.exceptions.RequestException), 
       delay=1, 
       backoff=2, 
       max_delay=4, 
       tries=3)
def download_current_data(url):
    
    try:
        response = requests.get(url)
        response.raise_for_status()     
        
        csv = response.content
        
        return csv
    
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
        sys.exit(1)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
        sys.exit(1)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
        sys.exit(1)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
        sys.exit(1)


def call_request():

    # Save as cvs
    today = date.today()
    file_name = "Covid-19-Brasil_" + str(today) + ".csv"

    url = 'https://brasil.io/dataset/covid19/caso_full/?is_repeated=False&format=csv'
    csv_response = download_current_data(url)

    with open('data/' + file_name, 'wb') as f:
        read_data = f.write(csv_response)

        if read_data:
            if os.path.islink("data/Covid-19-Brasil_Current.csv"):
                print("Remove old symlink")
                os.unlink("data/Covid-19-Brasil_Current.csv")

            print("Create current symlink")
            os.symlink(file_name, "data/Covid-19-Brasil_Current.csv" )

    print(f"File saved to data/{file_name}")
 

def main():

    file_name = "data/Covid-19-Brasil_" + str(date.today()) + ".csv"
    if not os.path.exists(file_name):
        call_request()
    else:
        print(file_name, "Already downloaded")

if __name__ == '__main__':
    main()