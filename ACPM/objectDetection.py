import csv
import os

import cv2
import keras as kr
import numpy as np
import torch
from matplotlib.image import imread
from moviepy.editor import VideoFileClip


def loadModel():

    # Funcion que se encarga de cargar los modelos como variables globales

    global selector
    selector = kr.models.load_model("ACPM/videoModels/IMAGESELECTORV.2")

    global detector
    detector = torch.hub.load(
        'ACPM/videoModels/yoloLib', 'custom', path='ACPM/videoModels/yoloLib/runs/train/exp9/weights/best.pt', source='local')  # local model


def getFrames(videoPath, framesPerSecond):

    # Parámetros
    # videoPath: ruta del video a procesar
    # framesPerSecond: cantidad de imagenes a tomar por segundo
    # Retorno
    # El numero de frames extraidos
    # Guarda las imagenes extraidas en la carpeta frames_extracted

    # Se carga el video haciendo uso de la libreria moviepy

    video = VideoFileClip(videoPath)

    # Se crea la ruta a la carpeta que almacenara las imagenes obtenidas

    outputFileName, _ = os.path.splitext("frames_extracted")

    # Se crea la carpeta en caso de que no exista

    if not os.path.isdir(outputFileName):
        os.mkdir(outputFileName)
    else:
        for file in os.listdir(outputFileName):
            if file[0] != ".":
                os.remove(os.path.join(outputFileName, file))

    # Se inicializa el contador de frames

    counter = 0

    # Se define el paso o avance en la duracion del video

    step = 1/framesPerSecond
    print("Nombre: ", video.filename)
    print("Duracion: ", video.duration)

    iterator = np.arange(0, video.duration, step)
    loadIcon = "▓"
    count = 0
    barCount = 0

    # Se extraen los frames del video

    for current_duration in iterator:

        frame_filename = os.path.join(outputFileName, f"frame{counter}.jpg")
        video.save_frame(frame_filename, current_duration)
        barCount = int((current_duration / video.duration) * 100)
        loadbar = loadIcon * barCount + "_" * (100 - barCount)
        print(f"[{loadbar}] {round(current_duration, 2)}/{video.duration}", end="\r")

        frameFilename = os.path.join(
            outputFileName, f"frame{counter}.jpg")
        video.save_frame(frameFilename, current_duration)
        counter += 1
        
        count += step
    print(f"[{loadIcon * 100}] {video.duration}/{video.duration}")


    # Se cierra el archivo asociado al video

    video.close()

    return counter


def categorizarImagen(path):

    img = imread(path).astype(float)/255
    img = cv2.resize(img, (224, 224))
    prediccion = selector.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1)


def detectObjectsInVideo(videoPath, outputPath, framesPerSecond):

    # Obtenemos los frames del video

    numImages = getFrames(videoPath, framesPerSecond)

    # Creamos el arreglo donde se guardaran las imagenes etiquetadas como "1"

    validImages = []

    for frameNumber in range(0, numImages):

        imgpath = "frames_extracted/frame"+str(frameNumber)+".jpg"

        categoria = categorizarImagen(imgpath)

        # Si la imagen es analizable, la añadimos al arreglo creado anteriormente

        if categoria == 1:

            validImages.append(imgpath)

    results = detector(validImages)
    results.save(save_dir=outputPath)
    create_csv_file(results.names, outputPath + "/results.csv")


def create_csv_file(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
