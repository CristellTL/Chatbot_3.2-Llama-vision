const express = require('express');
const cors = require('cors');
const axios = require('axios');
const multer = require('multer');
const path = require('path');

const app = express();
const PORT = 3000;

// Configurar CORS para permitir solicitudes desde cualquier origen
app.use(cors());

// Middleware para manejar JSON y datos de formularios
app.use(express.json());

// Configuración de multer para manejar archivos (imágenes)
const upload = multer();

// Servir archivos estáticos del frontend
app.use(express.static(path.join(__dirname, 'public')));

// Ruta principal (opcional)
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Ruta para manejar solicitudes a /api/chat
app.post('/api/chat', upload.single('image'), async (req, res) => {
    const { message } = req.body;
    const imageFile = req.file;

    if (!message) {
        return res.status(400).json({ error: 'El campo "message" es obligatorio.' });
    }

    try {
        // Construir la solicitud para Ollama
        const payload = {
            model: 'llama3.2-vision',
            messages: [{ role: 'user', content: message }],
        };

        if (imageFile) {
            // Convertir la imagen a base64 y agregarla al payload
            const imageBase64 = imageFile.buffer.toString('base64');
            payload.messages[0].images = [imageBase64];
        }

        // Enviar la solicitud a Ollama
        const response = await axios.post('http://127.0.0.1:11434/api/chat', payload, {
            headers: { 'Content-Type': 'application/json' },
        });

        // Devolver la respuesta de Ollama al cliente
        res.json(response.data);
    } catch (error) {
        console.error('Error al consultar Ollama:', error.message);
        res.status(500).json({ error: 'Error al procesar la consulta con Ollama.' });
    }
});

// Ruta para manejar cualquier otra solicitud no definida
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Iniciar el servidor
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Servidor corriendo en http://0.0.0.0:${PORT}`);
});
