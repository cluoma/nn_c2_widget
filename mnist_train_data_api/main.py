# libraries for fastapi
from typing import List, Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# libraries for other stuff
import sqlite3

class Digit(BaseModel):
    claimed_digit: int
    predicted_digit: int
    pixels: List[float] = []

# origins = [
#     "http://0.0.0.0:5000",
#     "http://0.0.0.0:8000",
#     "http://cluoma.com/",
#     "http://cluoma.com/",
#     "https://www.cluoma.com/",
#     "https://www.cluoma.com/",
# ]
origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def insert_into_database(digit: Digit):
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("insert into digits (digit) "
                    "values (:digit)",
                    {"digit": digit.json()})
    cur.fetchall()
    cur.close()
    con.commit()
    con.close()
    return None

@app.post("/mnistwidget/train")
async def receive_data(digit: Digit):
    #print(digit.json())
    insert_into_database(digit)
    return {"status": 200}

if __name__ == '__main__':
    uvicorn.run(app, port=9595, host='127.0.0.1')
