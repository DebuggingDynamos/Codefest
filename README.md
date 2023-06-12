![Navy Blue Futuristic Virtual Technology Banner-min](https://github.com/DebuggingDynamos/Codefest/assets/98624209/c1d5253c-7dab-46eb-8893-2d69c31406c8)

# CODEFEST AD ASTRA 2023: Software y tecnología aérea, espacial y cibernética para la protección de la Amazonía
Este proyecto es el resultado de nuestra participación en el Codefest Ad Astra 2023 y se enfoca en el desarrollo de modelos de Machine Learning para abordar dos desafíos principales: 
- Indexación de objetos y el análisis de la afectación de la Amazonía en videos captados por sensores aerotransportados
- Análisis de la información disponible en noticias públicas para comprender mejor la afectación de la Amazonía colombiana
## Pre-requisitos
~~~~
pip install beautifulsoup4
pip install joblib
pip install jsonlib
pip install keras
pip install matplotlib
pip install nltk
pip install numpy
pip install opencv-python
pip install regex
pip install requests
pip install spacy
~~~~
## Instalación
## Uso
## Estrategia
### Análisis de video
1. **Extracción de imágenes**: El video es procesado para convertirlo en una secuencia de imágenes. Se extraen los frames cada cierto tiempo. Por defecto, se selecciona una frecuencia de 1 frame por segundo (fps), pero el usuario tiene la opción de cambiar este valor según sus necesidades.

2. **Clasificación de imágenes**: Se emplea una modificación del modelo [`MobileNet_v2`](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4), originalmente desarrollado por Google. La capa de salida se ha modificado para tener dos neuronas con función de activación softmax. El propósito es etiquetar las imágenes como 0 (no de interés) o 1 (de interés), con el fin de eliminar aquellas que no aporten valor al análisis.

3. **Detección de objetos**: Las imágenes clasificadas como "de interés" por el primer modelo son ingresadas a un segundo modelo basado en el algoritmo de detección de objetos [`YOLO`](). Gracias al entrenamiento realizado, este modelo retorna las imágenes con los objetos relevantes señalados por un recuadro.
### Análisis de noticias
1. **Identificación de Entidades**: Para la identificación de las entidades PER, LOC, ORG y MISC, se utiliza el modelo [`es_core_news_lg`](https://spacy.io/models/es#es_core_news_lg) de spaCy, una biblioteca de procesamiento de lenguaje natural. Este modelo preentrenado ofrece un rendimiento adecuado para el idioma español.

2. **Identificación de Fechas**: Para la identificación de entidades de tipo DATE (fechas), se emplean patrones de expresiones regulares que representan diferentes formatos de fecha. Esta decisión se tomó después de intentar entrenar sin éxito un modelo específico para el reconocimiento de este tipo de entidad en spaCy. El modelo entrenado solo reconocía fechas exactas presentes en los datos de entrenamiento, lo que no resultaba eficiente para nuestro propósito.

3. **Clasificación del Impacto Ambiental**
    - **Creación del Conjunto de Datos**: Se creó un conjunto de datos para la clasificación del impacto ambiental, que incluye las noticias proporcionadas en la competencia y otros 400 datos obtenidos de fuentes en línea.
    - **Vectorización de los Datos**: Los datos se vectorizan utilizando una capa de hashing (fuera del modelo), que devuelve un array donde cada elemento representa una palabra. Al combinar los arrays de cada dato, se obtiene una matriz de arrays con diferentes longitudes. Luego, se realiza un padding para obtener una matriz con dimensiones consistentes.
    - **Creación del Modelo**: Se crea un modelo secuencial con las siguientes capas: una capa de embedding, dos capas de Long Short-Term Memory (LSTM) bidireccionales, dos capas densas y capas de dropout entre cada capa. Este modelo se entrena con el conjunto de datos creado anteriormente durante 150 épocas.
    - **Predicciones**: Para realizar predicciones, se debe realizar el mismo preprocesamiento de los datos utilizando la capa de hashing. Es importante tener en cuenta que al ejecutar pruebas desde otro archivo, esta capa de hashing modificará los parámetros de decisión para vectorizar las cadenas de texto. Por esta razón, se exporta la capa utilizada en el preprocesamiento de los datos de entrenamiento para ser utilizada en las predicciones.
## Autores
- Alejandro Pulido - [`alejandroPulido03`](https://github.com/alejandroPulido03)
- Nicolas Rozo Fajardo - [`MrCheesyBurgerU`](https://github.com/MrCheesyBurgerU)
- Manuela Pacheco Malagón - [`itsmemanuu`](https://github.com/itsmemanuu)
- Luis Felipe Torres - [`Luisfetoga2`](https://github.com/Luisfetoga2)
- Juan Montenegro - 
## Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo [`LICENSE`](https://github.com/DebuggingDynamos/Codefest/blob/main/LICENSE).

