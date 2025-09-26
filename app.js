import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/transformers/dist/transformers.min.js';

let textEncoder, unet, vaeDecoder, tokenizer;

// Initialize tokenizer and models
async function loadModels() {
  tokenizer = await AutoTokenizer.from_pretrained('https://huggingface.co/onnxruntime/sd-turbo/resolve/main/tokenizer/');

  textEncoder = await ort.InferenceSession.create(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/text_encoder/model.onnx'
  );
  unet = await ort.InferenceSession.create(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx'
  );
  vaeDecoder = await ort.InferenceSession.create(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/vae_decoder/model.onnx'
  );

  console.log('Models and tokenizer loaded!');
}

// Generate random latent
function initLatent(size=64){
  const latent = new Float32Array(size*size*4);
  for(let i=0;i<latent.length;i++) latent[i]=Math.random()*2-1;
  return latent;
}

// Draw latent tensor to canvas
function drawCanvas(latents, canvas){
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(canvas.width, canvas.height);
  for(let i=0;i<latents.length;i+=4){
    imgData.data[i]   = (latents[i]*127+128)|0;
    imgData.data[i+1] = (latents[i+1]*127+128)|0;
    imgData.data[i+2] = (latents[i+2]*127+128)|0;
    imgData.data[i+3] = 255;
  }
  ctx.putImageData(imgData,0,0);
}

// Main generation
async function generateImage(prompt){
  if(!textEncoder || !unet || !vaeDecoder || !tokenizer) await loadModels();

  // 1. Tokenize prompt
  const encoded = await tokenizer.encode(prompt);
  const tokenIds = encoded.ids.map(x=>BigInt(x));
  const inputIds = new ort.Tensor('int64', BigInt64Array.from(tokenIds), [1, tokenIds.length]);

  // 2. Text embeddings
  const embeddings = (await textEncoder.run({ input_ids: inputIds })).last_hidden_state;

  // 3. Initialize latent
  let latent = initLatent(64*64*4);

  // 4. Simplified diffusion steps (5 steps for demo)
  for(let t=0; t<5; t++){
    const unetOut = await unet.run({
      sample: latent,
      timestep: new Float32Array([t]),
      encoder_hidden_states: embeddings
    });
    for(let i=0;i<latent.length;i++) latent[i]-=0.1*unetOut.sample[i];
  }

  // 5. Decode latent → image
  const decoded = await vaeDecoder.run({ latents: latent });
  drawCanvas(decoded.latents, document.getElementById('canvas'));
}

// Register service worker
if('serviceWorker' in navigator){
  navigator.serviceWorker.register('/sw.js').then(()=>console.log('SW registered'));
}

// Button click
document.getElementById('generateBtn').addEventListener('click',()=>{
  const prompt = document.getElementById('prompt').value;
  generateImage(prompt);
});
