import re
import json
import spacy
import requests
import numpy as np
from joblib import load
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.utils import pad_sequences
from nltk.tokenize import word_tokenize


model = load_model("NLP\ACPM1111.h5")
hash = load("NLP\hash.joblib")


def predict(text):
    text = text.lower()
    text = text.replace("á", "a")
    text = text.replace("é", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o")
    text = text.replace("ú", "u")

    whitelist = set("abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ 0123456789")
    text = "".join(filter(whitelist.__contains__, text))

    max_length = 164
    nombres_clases = [
        "CONTAMINACION",
        "DEFORESTACION",
        "MINERIA",
        "NINGUNA",
        "NINGUNA",
    ]
    encoded_test = [hash(word_tokenize(text)).numpy()]
    padded_test = pad_sequences(encoded_test, maxlen=max_length, padding="post")
    x = nombres_clases[np.argmax(model.predict(padded_test))]
    return x


def finalLabel(text):
    nlp = spacy.load("es_core_news_lg")
    doc = nlp(text)
    dic = {}
    for ent in doc.ents:
        llave = ent.label_  # PER
        valor = ent.text  # Manuel
        if llave not in dic:
            dic[llave] = []
        dic[llave].append(valor)

    x = getDates(text)
    if len(x) > 0:
        dic["DATE"] = x

    return dic


def getDates(text):
    patrones = [
        r"(\d{2}[/]\d{2}[/]\d{4})",
        r"/(\d{2}[-]\d{2}[-]\d{4})",
        r"/(\d{2}[/]\d{2}[/]\d{2})",
        r"/(\d{2}[-]\d{2}[-]\d{2})",
        r"(\d{4}[/]\d{2}[/]\d{2})",
        r"(\d{4}[-]\d{2}[-]\d{2})",
        r"(\d{2} de \w+ de \d{4})",
        r"(\d{2} de \w+)",
        r"(\w+ de \d{4})",
    ]

    rt = []
    for i in patrones:
        rt += re.findall(i, text)

    return rt


def getDict(text):
    diccionario = {
        "text": text,
        "impact": predict(text),
    }
    
    diccionario.update(finalLabel(text))

    return diccionario


def nerFromString(text, outroute, outname="out"):
    diccionario = getDict(text)

    jsonobj = json.dumps(diccionario, indent=4, ensure_ascii=False)

    with open((outroute + "/" + outname + ".json"), "w", encoding="utf8") as outfile:
        outfile.write(jsonobj)


def nerFromCSV(inroute, outroute, outname="out"):
    x = []
    with open(inroute, encoding="utf-8") as csvfile:
        spamreader = csvfile.readlines()
        for row in spamreader:
            x.append(getDict(row))

    jsonobj = json.dumps(x, indent=4, ensure_ascii=False)

    with open((outroute + "/" + outname + ".json"), "w", encoding="utf8") as outfile:
        outfile.write(jsonobj)
        

def nerFromURL(url, outroute, outname="out"):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        text = soup.get_text()
        
        jsonobj = json.dumps(getDict(text), indent=4, ensure_ascii=False)

        with open((outroute + "/" + outname + ".json"), "w", encoding="utf8") as outfile:
            outfile.write(jsonobj)

    else:
        print("Failed to retrieve the URL:", response.status_code)
