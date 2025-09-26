import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';

let textEncoder, unet, vaeDecoder;

// Utility: create random latent
function initLatent(size=64) {
  const latents = new Float32Array(size*size*4); // 4 channels for demo
  for(let i=0;i<latents.length;i++) latents[i]=Math.random()*2-1;
  return latents;
}

// Utility: draw latent as image
function drawCanvas(latents, canvas) {
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(canvas.width, canvas.height);
  for(let i=0;i<latents.length;i+=4){
    imgData.data[i] = (latents[i]*127+128)|0;
    imgData.data[i+1] = (latents[i+1]*127+128)|0;
    imgData.data[i+2] = (latents[i+2]*127+128)|0;
    imgData.data[i+3] = 255;
  }
  ctx.putImageData(imgData,0,0);
}

async function loadModels() {
  textEncoder = await ort.InferenceSession.create('https://huggingface.co/onnxruntime/sd-turbo/resolve/main/text_encoder/model.onnx');
  unet = await ort.InferenceSession.create('https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx');
  vaeDecoder = await ort.InferenceSession.create('https://huggingface.co/onnxruntime/sd-turbo/resolve/main/vae_decoder/model.onnx');
  console.log("Models loaded!");
}

async function generateImage(prompt){
  if(!textEncoder || !unet || !vaeDecoder) await loadModels();

  // 1. Tokenize prompt (simplified demo, random embeddings)
  const embeddings = new Float32Array(1*77*768).fill(0.01); // batch x seq_len x hidden

  // 2. Initialize latent
  let latent = initLatent(64*64*4);

  // 3. Diffusion steps (simplified demo)
  for(let step=0;step<5;step++){
    const unetOut = await unet.run({ sample: latent, timestep: new Float32Array([step]), encoder_hidden_states: embeddings });
    // naive update
    for(let i=0;i<latent.length;i++) latent[i]-=0.1*unetOut.sample[i];
  }

  // 4. Decode latent to image
  const decoded = await vaeDecoder.run({ latents: latent });
  drawCanvas(decoded.latents, document.getElementById('canvas'));
}

if('serviceWorker' in navigator){
  navigator.serviceWorker.register('/sw.js').then(()=>console.log('SW registered'));
}

document.getElementById('generateBtn').addEventListener('click',()=>{
  const prompt = document.getElementById('prompt').value;
  generateImage(prompt);
});
