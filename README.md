## IMPLEMENTACIÓN DE UN CHATBOT MULTIMODAL CON MODELOS DE LENGUAJE AVANZADOS Y TRANSCRIPCIÓN DE AUDIO: USO DE LLAMA 3.2-VISION Y WHISPER

## Introducción.

En este documento se presenta la implementación de un chatbot multimodal que combina capacidades de procesamiento de lenguaje natural (NLP) y visión por computadora, utilizando el modelo Llama 3.2-Vision-Instruct-11B desarrollado por Meta. Además, se integra el modelo Whisper-base de OpenAI para la transcripción de audio a texto, permitiendo una interacción más completa y versátil con el usuario. El sistema se desarrolla en un entorno Node.js, donde se gestionan las solicitudes HTTP, se procesan imágenes y audio, y se comunican con modelos de inteligencia artificial locales a través de la API de Ollama. El documento detalla el flujo de trabajo, la configuración del servidor, la transcripción de audio mediante Python, y la técnica de fine-tuning con QLoRA para optimizar el rendimiento de los modelos de lenguaje. Este proyecto demuestra cómo las tecnologías de IA avanzadas pueden integrarse en aplicaciones web interactivas, ofreciendo soluciones innovadoras para la interpretación de texto, imágenes y audio.

## Diagrama de flujo de los procesos internos del Chatbot.

En el presente proyecto se utiliza un Modelo de Lenguaje para responder en base a un prompt.

![Descripción de la imagen](https://raw.githubusercontent.com/academicomfc/LLM_imagenes/main/EstructuraChat2.png)

## Descripción de los modelos utilizados:

### [Llama 3.2-Vision-Instruct-11B.] (https://ollama.com/library/llama3.2-vision:11b)

**_Descripción:_** Llama (Large Language Model Meta AI) es una serie de modelos de lenguaje desarrollados por Meta (anteriormente conocida como Facebook). El modelo Llama 3.2-Vision-Instruct-11B es una versión avanzada que combina capacidades de lenguaje con visión artificial. "3.2" hace referencia a la versión del modelo, "Vision" indica que el modelo tiene capacidades de procesamiento visual, y "Instruct" sugiere que está optimizado para realizar tareas bajo instrucciones.

**_Uso:_** Este modelo se utiliza para tareas que combinan procesamiento de lenguaje natural (NLP) y visión por computadora. Esto incluye la capacidad de analizar imágenes y describirlas con texto, interpretar instrucciones relacionadas con imágenes, y generar respuestas a partir de contenidos visuales. Ideal para aplicaciones como la interpretación de imágenes, generación de descripciones automáticas de escenas visuales, y asistentes de inteligencia artificial que necesitan procesar tanto texto como imágenes.

### [Whisper.](https://openai.com/index/whisper/)

**_Descripción:_** Whisper es un modelo de OpenAI diseñado para la transcripción de audio a texto, también conocido como un sistema de reconocimiento de voz. Está entrenado para manejar una variedad de idiomas y es capaz de realizar tareas de transcripción, traducción y detección de idiomas en tiempo real.

**_Uso:_** Se utiliza principalmente en aplicaciones de reconocimiento de voz, como la transcripción automática de reuniones, subtitulado de videos, asistencia por voz, traducción de audio a texto y muchas otras aplicaciones que requieren convertir audio en texto escrito. Además, Whisper es bastante robusto y puede entender diferentes acentos y sonidos, lo que lo hace muy útil en contextos multilingües y diversos.

## Descripción del código del servidor.

El código define un servidor web construido con Node.js y Express.js que proporciona dos funcionalidades principales:

- Un sistema de chat con capacidades de análisis de imágenes.

- Un servicio de transcripción de audio utilizando un modelo de IA (probablemente Whisper).

Este servidor se comunica con un API local llamado Ollama y ejecuta scripts de Python en un entorno Conda para la transcripción de audio.

### 1. Dependencias y Configuración Inicial

```javascript
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");
```

- **_express:_** Framework web para manejar rutas y peticiones HTTP.

- **_cors:_** Permite la comunicación entre diferentes orígenes (Cross-Origin Resource Sharing).

- **_axios:_** Cliente HTTP para hacer solicitudes a APIs externas.

- **_multer:_** Middleware para manejar archivos en peticiones multipart/form-data.

- **_path y fs:_** Módulos para manejo de rutas de archivos y operaciones en el sistema de archivos.

- **_child_process (exec):_** Ejecuta comandos del sistema, en este caso para correr scripts de Python.

Configuración básica del servidor:

```javascript
const app = express();
const PORT = 3000;
const OLLAMA_URL = "http://127.0.0.1:11434/api/chat";
const CONDA_ENV = "whisper_env";
```

### 2. Middleware y Configuración de Rutas.

```javascript
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
```

- **_CORS_** habilitado para permitir peticiones de diferentes dominios.

- **_express.json()_** para parsear cuerpos JSON en solicitudes HTTP.

- **_Archivos estáticos_** servidos desde la carpeta public (HTML, CSS, JS).

### 3. Ruta Principal (/).

```javascript
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});
```

- **_Funcionalidad:_** Sirve el archivo index.html ubicado en la carpeta public.

- **_Propósito:_** Proporciona una interfaz de usuario básica para interactuar con el servidor.

### 4. API de Chat con Soporte de Imágenes (/api/chat).

```javascript
app.post("/api/chat", upload.single("image"), async (req, res) => {
  const { message } = req.body;
  const imageFile = req.file;

  if (!message) {
    return res
      .status(400)
      .json({ error: 'El campo "message" es obligatorio.' });
  }

  try {
    const payload = {
      model: "llama3.2-vision",
      messages: [{ role: "user", content: message }],
    };

    if (imageFile) {
      const imageBase64 = imageFile.buffer.toString("base64");
      payload.messages[0].images = [imageBase64];
    }

    const response = await axios.post(OLLAMA_URL, payload, {
      headers: { "Content-Type": "application/json" },
    });

    res.json(response.data);
  } catch (error) {
    console.error("Error al consultar Ollama:", error.message);
    res
      .status(500)
      .json({ error: "Error al procesar la consulta con Ollama." });
  }
});
```

**_Flujo de Funcionamiento:_**

- **_Recepción de Datos:_** Acepta mensajes de texto y opcionalmente una imagen.

- **_Validación:_** Verifica que el campo message esté presente.

- **_Procesamiento de Imágenes:_** Si hay una imagen, se convierte a Base64.

- **_Comunicación con Ollama:_** Envía los datos a la API de Ollama para obtener una respuesta.

- **_Respuesta:_** Devuelve la respuesta del modelo de IA al cliente.

**_Errores Comunes:_**

- Falta del campo message (error 400).

- Problemas al conectar con Ollama (error 500).

### 5. API de Transcripción de Audio (/api/transcribe).

```javascript
app.post("/api/transcribe", upload.single("audio"), (req, res) => {
  if (!req.file) {
    return res
      .status(400)
      .json({ error: "No se recibió ningún archivo de audio." });
  }

  const audioPath = path.join(__dirname, "temp_audio.wav");
  fs.writeFileSync(audioPath, req.file.buffer);

  const command = `conda run -n ${CONDA_ENV} python transcribe.py "${audioPath}"`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error("Error ejecutando Whisper:", stderr);
      return res
        .status(500)
        .json({ error: "Error al procesar la transcripción de audio." });
    }

    const transcription = stdout.trim();
    fs.unlinkSync(audioPath);

    res.json({ transcription });
  });
});
```

**_Flujo de Funcionamiento:_**

- **_Recepción del Archivo:_** Acepta un archivo de audio.

- **_Almacenamiento Temporal:_** Guarda el archivo temporalmente en el servidor.

- **_Ejecución de Script de Python:_** Utiliza exec para ejecutar transcribe.py en un entorno Conda.

**_Respuesta:_** Devuelve la transcripción al cliente.

- **_Limpieza:_** Elimina el archivo de audio después de la transcripción.

**_Posibles Errores:_**

- No se recibe archivo de audio (error 400).

- Fallo en la ejecución del script de Python (error 500).

### 6. Ruta Catch-All (\*).

```javascript
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});
```

**_Funcionalidad:_** Redirige cualquier ruta no definida a index.html.

**_Propósito:_** Soporte para aplicaciones SPA (Single Page Applications).

### 7. Inicialización del Servidor.

```javascript
app.listen(PORT, () => {
  console.log(`Servidor corriendo en http://localhost:${PORT}`);
});
```

- El servidor escucha en el puerto 3000.

- Muestra un mensaje de confirmación en la consola.

Este servidor Node.js está diseñado para integrar capacidades de IA, permitiendo tanto interacciones basadas en texto e imágenes (mediante Ollama) como la transcripción de audio (usando Whisper). Su estructura modular y uso de tecnologías modernas lo hace adecuado para aplicaciones web interactivas y dinámicas.

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

### Introducción a Q LoRA y Fine-tuning para Modelos de Lenguaje de Gran Tamaño (LLMs)

**Q LoRA (Quantized Low-Rank Adaptation)** es una técnica innovadora que permite adaptar y optimizar modelos de lenguaje de gran tamaño (LLMs) de manera eficiente. Combina los enfoques de cuantización y adaptación de baja dimensionalidad, logrando una notable reducción en los recursos computacionales necesarios para el ajuste fino (fine-tuning). Esto es crucial en un contexto donde los LLMs demandan gran cantidad de memoria y poder de procesamiento.

### Elementos principales de Q LoRA

1. **Cuantización (Quantization):**  
   Convierte los pesos del modelo a una representación de menor precisión (como 8 bits) para reducir el uso de memoria y acelerar los cálculos, introduciendo mínimas pérdidas de precisión.

2. **Adaptación de Baja Dimensionalidad (LoRA):**  
   Ajusta matrices de bajo rango agregadas al modelo original, mientras los parámetros preentrenados permanecen fijos. Esto optimiza el proceso al reducir drásticamente el número de parámetros entrenables.

### Fine-tuning con Q LoRA

El fine-tuning es el proceso de ajustar modelos preentrenados para tareas específicas. Gracias a Q LoRA, este proceso es más accesible y eficiente, ya que se reduce la necesidad de hardware costoso y se optimizan los tiempos de entrenamiento sin comprometer el rendimiento del modelo.

### Sobre el uso de QLoRA para LLM

QLoRA es una técnica revolucionaria que permite ajustar eficientemente modelos de lenguaje grandes (LLM) utilizando menos recursos computacionales y optimizando el tiempo de entrenamiento. Su enfoque combina estrategias como la cuantización a 4 bits y el ajuste fino eficiente de parámetros (LoRA), lo que la convierte en una herramienta ideal para trabajar con modelos de gran tamaño, incluso en hardware limitado.

### Beneficios clave

- **Reducción de costos computacionales**: Al utilizar cuantización a 4 bits, QLoRA minimiza el uso de memoria sin sacrificar la precisión del modelo.
- **Ajuste fino eficiente**: LoRA permite modificar únicamente un subconjunto de parámetros del modelo, lo que acelera el proceso de entrenamiento y reduce la necesidad de grandes recursos.
- **Escalabilidad**: Es posible aplicar QLoRA a modelos desde millones hasta miles de millones de parámetros, haciendo viable el uso de LLM en más aplicaciones.
- **Flexibilidad**: Compatible con múltiples frameworks y adaptable a diversas tareas, desde generación de texto hasta comprensión del lenguaje.

### Limitaciones

- Aunque QLoRA es eficiente, el uso de hardware adecuado (como GPUs modernas) sigue siendo clave para aprovechar al máximo sus beneficios.
- La implementación puede requerir ajustes técnicos específicos para ciertas arquitecturas de modelos o tipos de datos.
---

### Instalación de librerías necesarias

Para implementar Q LoRA y realizar el fine-tuning, se utilizan diversas bibliotecas especializadas:

1. **Accelerate:**  
   Biblioteca de Hugging Face que simplifica el entrenamiento e inferencia de modelos en múltiples dispositivos (GPUs, TPUs).

2. **PEFT:**  
   Biblioteca para el ajuste fino eficiente de parámetros. Optimiza el ajuste fino en modelos grandes sin necesidad de actualizar todo el modelo.

3. **Bitsandbytes:**  
   Herramienta diseñada para operaciones de precisión reducida y eficiente, útil para ahorrar memoria en modelos de aprendizaje profundo.

4. **Transformers:**  
   Biblioteca de Hugging Face que permite trabajar con modelos basados en Transformers de manera flexible y eficiente.

5. **TRL (Transformers Reinforcement Learning):**  
   Extensión para integrar el aprendizaje por refuerzo en modelos de Transformers

 El código de como se pueden instalar las librerias y la importación de bibliotecas se muestra a continuación:

 ### Código para instalar librerías

```python
 !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
 
```
### Código importación de bibliotecas

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

#peft: Parameter-Efficient Fine-Tuning (PEFT) es una técnica que permite ajustar modelos de lenguaje grandes sin la necesidad de modificar todos sus parámetros.
#LoraConfig: Esta clase se usa para definir la configuración de LoRA (Low-Rank Adaptation), una técnica de ajuste fino eficiente.
from peft import LoraConfig, PeftModel

#SFTTrainer: Es un entrenador especializado en el ajuste fino supervisado (Supervised Fine-Tuning) de modelos.
from trl import SFTTrainer

```
### Base de datos

El conjunto de datos **mlabonne/guanaco-llama2-1k** es una colección de 1,000 muestras extraídas del destacado dataset **timdettmers/openassistant-guanaco**. Este subconjunto ha sido procesado para alinearse con el formato de prompts de **Llama 2**, según lo descrito en [este artículo](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k/blob/main/README.md).

El dataset está disponible en formato **Parquet** y contiene datos textuales que facilitan el ajuste fino de modelos de lenguaje, especialmente en tareas de generación y comprensión de texto. Su tamaño compacto lo convierte en una opción ideal para experimentos y pruebas en entornos con recursos computacionales limitados.

### Modelo NousResearch/Llama-2-7b-chat-hf

El modelo NousResearch/Llama-2-7b-chat-hf es un modelo de lenguaje grande (LLM) desarrollado por Nous Research, basado en la arquitectura Llama 2 de Meta. Este modelo en particular tiene 7 mil millones de parámetros y está optimizado para conversaciones interactivas. Ha sido ajustado para mejorar su capacidad de diálogo y comprensión del lenguaje natural.

**Características principales:**

*   **Tamaño:** 7 mil millones de parámetros, lo que lo hace relativamente pequeño en comparación con otros LLMs, pero aún capaz de generar texto de alta calidad.
*   **Optimización para diálogo:** Ha sido entrenado específicamente para mantener conversaciones coherentes y relevantes, respondiendo a preguntas y siguiendo el hilo de la discusión.
*   **Basado en Llama 2:** Se beneficia de la arquitectura y el entrenamiento de Llama 2, lo que le proporciona una base sólida en comprensión y generación del lenguaje natural.
*   **Formato Hugging Face:** Está disponible en el formato Transformers de Hugging Face, lo que facilita su uso en proyectos de aprendizaje automático y procesamiento del lenguaje natural.

**Posibles usos:**

*   **Asistentes virtuales:** Puede ser utilizado para crear chatbots y asistentes virtuales capaces de interactuar de manera natural con los usuarios.
*   **Generación de contenido:** Puede generar texto creativo, como poemas, artículos o guiones, así como responder preguntas y proporcionar información.
*   **Investigación en PLN:** Sirve como una herramienta valiosa para investigadores en el campo del procesamiento del lenguaje natural, permitiéndoles estudiar y mejorar las capacidades de los LLMs.

**Consideraciones:**

*   **Sesgos:** Como todos los LLMs, puede contener sesgos presentes en los datos con los que fue entrenado. Es importante ser consciente de esto y tomar precauciones para mitigar los posibles sesgos en sus respuestas.
*   **Uso responsable:** Se recomienda utilizar este modelo de manera ética y responsable, evitando su uso para generar contenido dañino, engañoso o discriminatorio.

En el siguiente apartado se muestra el codigo utilizando la base de datos **mlabonne/guanaco-llama2-1k** y el **Modelo NousResearch/Llama-2-7b-chat-hf**

### Código para reentrenar el modelo con Q LoRA

```python
## El modelo que se va a reentrenar
model_name = "NousResearch/Llama-2-7b-chat-hf"

## La nueva base de datos
dataset_name = "mlabonne/guanaco-llama2-1k"

## El nuevo modelo resultante
new_model = "llama-2-7b-miniguanaco"

################################################################################
## QLoRA parámetros
################################################################################

## LoRA attention dimension
lora_r = 64

## Alpha parameter for LoRA scaling
lora_alpha = 16

## Dropout
lora_dropout = 0.1

################################################################################
## bitsandbytes parámetros
################################################################################

## Activate 4-bit precision base model loading
use_4bit = True

## Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

## Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

## Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
## TrainingArguments parámetros
################################################################################

## Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

## Number of training epochs
num_train_epochs = 1

## Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

## Batch size per GPU for training
per_device_train_batch_size = 4

## Batch size per GPU for evaluation
per_device_eval_batch_size = 4

## Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

## Enable gradient checkpointing
gradient_checkpointing = True

## Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

## Tasa de aprendizaje con ADAMW
learning_rate = 2e-4

## Modificación de los pesos
weight_decay = 0.001

## Optimizador
optim = "paged_adamw_32bit"

## Tasa de aprendizaje
lr_scheduler_type = "cosine"

## Número de pasos máximos
max_steps = -1

## Ratio of steps for a linear warmup
warmup_ratio = 0.03

## Agrupar secuencias en lotes con la misma longitud
## Ahorra memoria y acelera considerablemente el entrenamiento
group_by_length = True

## Guardar los pasos
save_steps = 0

## Pasos de registro
logging_steps = 25

################################################################################
## SFT parámetros
################################################################################

## Tamaño de secuencia máxima
max_seq_length = None

packing = False

## Cargar modelo en GPU
device_map = {"": 0}



```

## Configuración Inicial

En este apartado del código se presenta, la carga de los datos, la configuración inicial, el modelo pre-entrenado, el tokenizer, la configuración para LoRA y finalmente el entrenamiento del modelo

## Código para cargar la base de datos y configurar el modelo con Q LoRA

```python
# Cargar base de datos
dataset = load_dataset(dataset_name, split="train")

# Cargar tokenizer y modelo con configuración de QLoRA
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Trabajar con GPU con bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Cargar modelo
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configuración LoRA
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Parámetros de entrenamiento
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Parámetros del Fine Tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Reentrenar modelo
trainer.train()

```

Una vez realizado el código anterior, se muestra a continuación el código que permite ver un ejemplo de una pregunta que se realiza en el promp y su respuesta obtenida,

```python

# Ignorar las advertencias
logging.set_verbosity(logging.CRITICAL)

# Correr un ejemplo
prompt = "¿Como puedo encontrar trabajo de ingeniero? dame la respuesta en español"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

#Resultado de la ejecución

[INST] ¿Como puedo encontrar trabajo de ingeniero? dame la respuesta en español [/INST] Para encontrar trabajo de ingeniero, puedes buscar en línea en sitios web de empleo, como Indeed, LinkedIn o Glassdoor. everybody knows that the best way to find a job is to network.

Encuentra a personas que trabajen en la industria de la ingeniería y pídele que te recomienden a otros contactos. También puedes buscar en las redes sociales de empresas de la industria y dejar tu currículum vitae en sus sitios web de empleo.

Además, puedes buscar en las asociaciones de ingenieros y en las organizaciones de la industria. Estas pueden tener listas de empleos disponibles y puedes buscar en línea en sitios web de empleo.

```

### Pasos para la ejecución del entorno.

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


## Conclusión

El desarrollo de este chatbot multimodal demuestra la viabilidad de integrar múltiples modalidades de entrada (texto, imágenes y audio) en un solo sistema basado en modelos de lenguaje de gran tamaño. La implementación de QLoRA permitió realizar un ajuste fino eficiente, reduciendo costos computacionales sin afectar el rendimiento del modelo. Además, la combinación de Llama 3.2-Vision y Whisper proporcionó una solución robusta para el análisis de imágenes y la transcripción de audio en tiempo real. A pesar de las ventajas, el sistema depende del hardware disponible, lo que podría limitar su aplicabilidad en entornos con menos recursos. En general, este proyecto representa un avance significativo en la interacción hombre-máquina, ofreciendo un modelo más flexible y adaptativo para futuras aplicaciones de inteligencia artificial.
