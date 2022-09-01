import dbm
import json
import requests
import time

def cached_request(url):
    with dbm.open('semanticscholarcache', 'c') as db:
        if url in db:
            return json.loads(db[url])
        else:
            wait_time = 5
            while 1:
                q = requests.get(url)
                if q.status_code == 200:
                    break
                    
                # Exponential retry
                time.sleep(wait_time)
                wait_time *= 2
            db[url] = q.text
            return q.json()