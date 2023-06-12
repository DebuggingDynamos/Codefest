import ACPM.ner as ner
import ACPM.objectDetection as obj


def init_models():

    #Se inicializa el modelo de análisis de texto

    ner.loadModels()

    #Se inicializan los modelos de análisis de video

    obj.loadModel()

    print("Modelos inicializados correctamente")


def detect_objects_in_video(video_path, output_path, frames_per_second = 1):

    print("Inicio Análisis")

    #Se llama a la funcion que extrae los frames del video

    obj.detectObjectsInVideo(video_path,output_path,frames_per_second)

    print("Análisis Terminado :D")


def ner_from_str(text, output_path, out_name = "out"):

    print("Inicio Análisis")

    #Se llama la funcion correspondeinte del modulo ner

    ner.nerFromString(text, output_path, out_name)

    print("Análisis Terminado :D")


def ner_from_file(text_path, output_path, out_name = "out"):

    print("Inicio Análisis")

    #Se llama la funcion correspondeinte del modulo ner

    ner.nerFromCSV(text_path, output_path, out_name)

    print("Análisis Terminado :D")


def ner_from_url(url, output_path, out_name = "out"):

    print("Inicio Análisis")
    
    #Se llama la funcion correspondeinte del modulo ner
    
    ner.nerFromURL(url, output_path, out_name)

    print("Análisis Terminado :D")

