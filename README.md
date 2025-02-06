# Proyecto: Utilización de un modelo de lenguaje.

## Diagrama de flujo de los procesos internos del Chatbot.

En el presente proyecto se utiliza un Modelo de Lenguaje para responder en base a un prompt.

![Descripción de la imagen](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/EstructuraChat2.png)

## Descripción de los modelos utilizados:

###

**_Descripción:_** Llama (Large Language Model Meta AI) es una serie de modelos de lenguaje desarrollados por Meta (anteriormente conocida como Facebook). El modelo Llama 3.2-Vision-Instruct-11B es una versión avanzada que combina capacidades de lenguaje con visión artificial. "3.2" hace referencia a la versión del modelo, "Vision" indica que el modelo tiene capacidades de procesamiento visual, y "Instruct" sugiere que está optimizado para realizar tareas bajo instrucciones.

**_Uso:_** Este modelo se utiliza para tareas que combinan procesamiento de lenguaje natural (NLP) y visión por computadora. Esto incluye la capacidad de analizar imágenes y describirlas con texto, interpretar instrucciones relacionadas con imágenes, y generar respuestas a partir de contenidos visuales. Ideal para aplicaciones como la interpretación de imágenes, generación de descripciones automáticas de escenas visuales, y asistentes de inteligencia artificial que necesitan procesar tanto texto como imágenes.

### [Whisper.](https://openai.com/index/whisper/)

**_Descripción:_** Whisper es un modelo de OpenAI diseñado para la transcripción de audio a texto, también conocido como un sistema de reconocimiento de voz. Está entrenado para manejar una variedad de idiomas y es capaz de realizar tareas de transcripción, traducción y detección de idiomas en tiempo real.

**_Uso:_** Se utiliza principalmente en aplicaciones de reconocimiento de voz, como la transcripción automática de reuniones, subtitulado de videos, asistencia por voz, traducción de audio a texto y muchas otras aplicaciones que requieren convertir audio en texto escrito. Además, Whisper es bastante robusto y puede entender diferentes acentos y sonidos, lo que lo hace muy útil en contextos multilingües y diversos.

## Descripción de la transcripción de audio a texto.

Para la transcripción del audio a texto, se escribio un programa en python llamado transcribe.py. Este script toma un archivo de audio como entrada desde la línea de comandos, lo transcribe a texto utilizando el modelo de Whisper, y luego imprime el texto transcrito.

Está diseñado para ser ejecutado desde la terminal o un entorno de desarrollo que pase el archivo de audio como argumento. El texto que se imprime podría ser capturado y procesado por una aplicación en Node.js o cualquier otro sistema que utilice este script, a contnuación se describe las acciones realizadas por el código.

### 1. Importación de librerias

```python
import whisper
import sys
```

#### import whisper

Importa la librería Whisper, que es un modelo de inteligencia artificial desarrollado por OpenAI, capaz de transcribir audio a texto en varios idiomas.

#### import sys

Importa el módulo sys, que proporciona acceso a los argumentos de la línea de comandos y otras funcionalidades del sistema operativo.

### 2. Obtener el nombre del archivo de audio

```python
audio_file = sys.argv[1]
```

#### sys.argv:

Es una lista que contiene los argumentos de la línea de comandos pasados al ejecutar el script.

#### sys.argv[0]:

Es el nombre del script (en este caso, el archivo Python que contiene el código).

#### sys.argv[1]:

Es el primer argumento después del nombre del script. En este caso, se espera que sea el nombre del archivo de audio que se va a transcribir.

El código toma el primer argumento (`sys.argv[1]`) como el archivo de audio que se debe transcribir, y lo asigna a la variable `audio_file`.

### 3. Cargar el modelo de Whisper:

```python
model = whisper.load_model("base")
```

Se carga el modelo de transcripción de Whisper llamado "base". Whisper tiene varios modelos preentrenados con diferentes tamaños y capacidades (por ejemplo, "tiny", "base", "small", "medium", "large"). El modelo "base" es un modelo intermedio en cuanto a tamaño y precisión.

### 4.Transcribir el audio:

```python
result = model.transcribe(audio_file, language="es")
```

Aquí se llama al método transcribe del modelo para transcribir el archivo de audio especificado en audio_file. Se pasa como argumento adicional el parámetro language="es", lo que indica que el audio está en español, y Whisper lo transcribirá en ese idioma.

### 5. Imprimir el texto transcrito:

```python
print(result["text"])
```

Finalmente, el resultado de la transcripción, que es un diccionario, se imprime en la consola. El campo "text" contiene el texto transcrito que Whisper ha generado a partir del audio.

## Fine-Tunning.

## Pasos para la ejecución del entorno.

En la realización del presente proyecto se utilizó el sistema operativo ubuntu 20.04.6 LTS. El servidor con el sistema operativo se encuentra en el laboratorio de postgrado. A continuación se muestran los pasos para la ejecución del entorno.

### 1. Conexión mediante ssh al servidor con sistema operativo ubuntu.

En la figura 1 se puede observar los comandos utilizados para iniciar sesión en el sistema operativo.

![Conexión SSH](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/1.%20conexionSSH.png)

<p align="center">Figura 1. Iniciar sesión mediante ssh.</p>

### 2. Iniciar ejecución del servidor ollama.

En la figura 2 se observa el comando para iniciar el servidor ollama.

![Texto alternativo](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/2.ejecucionOllama.png)

<p align="center">Figura 2. Ejecutar servidor ollama.</p>

### 3. Mostrar la lista de entornos virtuales disponibles.

En la figura 3 se muestra el comando para mostrar los entornos virtuales disponibles.

![Texto alternativo](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/3.%20ListaEntornosEjecutables.png)

<p align="center">Figura 3. Entorno disponibles</p>

### 4. Ejecución del entorno virtual.

En la figura 4, se observa el comando utilizado para ejecutar el entorno virtual llamado **_whisper_env_**.

![Texto alternativo](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/4.%20ActivacionEntorno.png)

<p align="center">Figura 4. Ejecucion del entorno virtual</p>

### 5. Ejecución del servidor escrito en nodejs.

En la figura 5, se realizan las siguiente acciones.

- Ingresar a la carpeta Chatbot_3.2-Llama-vision.
- Ejecutar el servidor escrito en nodejs.

![Texto alternativo](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/6.EjecucionServidorNode.png)

<p align="center">Figura 5. Ejecucion del servidor servidor escrito en nodejs.

Después de realizar los pasos anteriores, el siguiente paso es mostrar la interfaz web desde un navegador para comunicarse con el modelo de lenguaje multimodal llama 3.2.

## Pruebas realizadas.

A continuación se muestran los pasos a seguir para ejecutar el chatbot multimodal.

### 1. Cargar la interfaz web en un navegador.

Cabe comentar que para la realización del proyecto se utilizó un servidor del laboratorio de postgrado, por dicha razón se escribe en el navegador la dirección ip del servidor.

`http://192.168.241.18:3000/`

Nota: En el código del servidor escrito en nodejs, se indica que va a estar escuchando las peticiones web en el puerto 3000.

En la figura 6, se observa la interfaz del chatbot multimodal corriendo en el servidor del laboratorio de postgrado.

![Interfaz Web](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/7.%20InterfazWeb.png)

<p align="center">Figura 6. Interfaz web del chatbot multimodal</p>

### 2. Cargar imagen.

En la figura 7 se muestra el proceso para cargar una imagen. Dicha imagen será enviada al llama3.2.
![Selección de Imagen](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/8.seleccionImagen.png)

<p align="center">Figura 7. Cargar imagen</p>

En la figura 8 se muestra el resultado de cargar la imagen.
![Buscar Imagen](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/9.BuscarImagen.png)

<p align="center">Figura 8. Imagen cargada</p>

### 3. Crear prompt con texto e imagen.

En la figura 9 se muestra de qué manera se puede crear un prompt con texto e imagen. En este caso el prompt consiste en pedirle a llama3.2 que describa el contenido de la imagen.

![Prompt de Texto 1](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/10.PronptTexto1.png)

<p align="center">Figura 9. Preparar prompt con texto e imagen</p>

En la figura 10 se observa la respuesta del modelo del lenguaje. Cabe comentar que el prompt consistió en enviar texto e imagen. Posteriormente el modelo contesta solamente texto.

![Descripción Prompt 1 - Parte 2](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/12.descripcionProm1_p2.png)

<p align="center">Figura 10. Respuesta del modelo.</p>

### 4. Crear prompt con audio.

Antes de realizar la creación de un prompt con audio, es necesario realizar una configuración de seguridad en el navegador.

Recordemos que para cargar la interfaz web, se escribe la siguiente dirección.

`http://192.168.241.18:3000/`

La URL anterior carece de seguridad, y por consecuencia el navegador lo considera inseguro y no permite que el botón iniciar la grabación del audio se active en la interfaz web.

Por lo tanto a continuación se muestra la configuración a realizar para que el navegador considere la URL segura y en consecuencia permita activar el botón de grabar.

`chrome://flags/#unsafely-treat-insecure-origin-as-secure`

La configuración antes mencionada, hace referencia a una opción experimental en las flags de Chrome que permite tratar ciertos orígenes inseguros como si fueran seguros. Esta opción puede ser útil en entornos de desarrollo, pero no se recomienda activarla en situaciones normales debido a los riesgos de seguridad.

#### Configuración del navegador chrome.

Para realizar la configuración es necesario copiar y pegar en el url del navegador la siguiente instrucción, ver figura 11:
`chrome://flags/#unsafely-treat-insecure-origin-as-secure`

Posteriormente elegir la opción enabled del botón, que se encuentra en lado derecho de la pantalla. Posteriormente hay que dar click en el mensaje que dice **_reset all_**.

![Seguridad](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/13.seguridad.png)

<p align="center">Figura 11. Tratar sitio inseguro como seguro</p>

### 1. Crear prompt con audio.

Si todo salió bien deberá mostrarse un micrófono activado del lado izquierdo de la URL, ver figura 11.

![Micrófono Activado](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/microfonoActivado.png)

<p align>Figura 11. Micrófono activado </p>

#### El siguiente paso consiste en crear el prompt de audio.

Para iniciar la grabación del audio se debe dar click sobre el botón de **_iniciar grabación_** y para finalizar la grabación se da click sobre botón **_Detener grabación_**.

![Resultado Configuración Audio](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/14.resultadoConfiguracinAudio.png)

<p align="center">Figura 12.Grabar audio</p>

En la figura 13, se observa que después de grabar el audio, se debe presionar el botón de play para que transcriba a texto el audio grabado, tal como se muestra en la figura.

En este caso el audio grabado es "explica que es un LLM". Posteriormente en envía la petición al modelo de lenguaje.
![Comando Voz](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/15.comandoVoz.png)

<p align="center">Figura 13. Transcribir audio a texto</p>
