const express = require('express');
const cors = require('cors');
const axios = require('axios');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

const app = express();
const PORT = 3000;
const OLLAMA_URL = 'http://127.0.0.1:11434/api/chat';
const CONDA_ENV = 'whisper_env';  // Nombre de tu entorno virtual de Conda

app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/api/chat', upload.single('image'), async (req, res) => {
    const { message } = req.body;
    const imageFile = req.file;

    if (!message) {
        return res.status(400).json({ error: 'El campo "message" es obligatorio.' });
    }

    try {
        const payload = {
            model: 'llama3.2-vision',
            messages: [{ role: 'user', content: message }],
        };

        if (imageFile) {
            const imageBase64 = imageFile.buffer.toString('base64');
            payload.messages[0].images = [imageBase64];
        }

        const response = await axios.post(OLLAMA_URL, payload, {
            headers: { 'Content-Type': 'application/json' },
        });

        res.json(response.data);
    } catch (error) {
        console.error('Error al consultar Ollama:', error.message);
        res.status(500).json({ error: 'Error al procesar la consulta con Ollama.' });
    }
});

app.post('/api/transcribe', upload.single('audio'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No se recibió ningún archivo de audio.' });
    }

    const audioPath = path.join(__dirname, 'temp_audio.wav');
    fs.writeFileSync(audioPath, req.file.buffer);

    const command = `conda run -n ${CONDA_ENV} python transcribe.py "${audioPath}"`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error('Error ejecutando Whisper:', stderr);
            return res.status(500).json({ error: 'Error al procesar la transcripción de audio.' });
        }

        const transcription = stdout.trim();
        fs.unlinkSync(audioPath);

        res.json({ transcription });
    });
});

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Servidor corriendo en http://localhost:${PORT}`);
});


// conda create --name whisper_env python=3.9
