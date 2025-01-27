import whisper
import sys

# Obtener el nombre del archivo de audio desde los argumentos
audio_file = sys.argv[1]

# Cargar el modelo de Whisper
model = whisper.load_model("base")

# Transcribir el audio
result = model.transcribe(audio_file, language="es")

# Imprimir el texto transcrito para ser capturado por Node.js
print(result["text"])
