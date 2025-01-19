document.getElementById('sendBtn').addEventListener('click', async () => {
    const userInput = document.getElementById('userInput').value;
    const response = await fetch(`/api/chat?message=${encodeURIComponent(userInput)}`);
    const data = await response.json();
  
    const chatBox = document.getElementById('chatBox');
    chatBox.innerHTML += `<p><strong>Bot:</strong> ${data.message}</p>`;
  
    if (data.imageUrl) {
      chatBox.innerHTML += `<img src="${data.imageUrl}" alt="Imagen del bot" style="max-width: 300px;" />`;
    }
  });
  
  document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    if (!fileInput.files.length) {
      alert('Selecciona una imagen antes de subir');
      return;
    }
  
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
  
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
  
    const data = await response.json();
    alert(data.message);
  
    const chatBox = document.getElementById('chatBox');
    chatBox.innerHTML += `<p><strong>Usuario:</strong> Imagen subida</p>`;
    chatBox.innerHTML += `<img src="${data.filePath.replace(__dirname, '')}" alt="Imagen subida" style="max-width: 300px;" />`;
  });
  