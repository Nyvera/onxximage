import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/transformers/dist/transformers.min.js';

let textEncoder, unet, vaeDecoder, tokenizer;
const statusEl = document.getElementById('status');
const generateBtn = document.getElementById('generateBtn');

async function loadModelWithStatus(name, url) {
  statusEl.innerText = `Loading ${name}...`;
  const model = await ort.InferenceSession.create(url);
  statusEl.innerText = `${name} loaded`;
  return model;
}

async function loadModels() {
  statusEl.innerText = 'Loading tokenizer...';
  tokenizer = await AutoTokenizer.from_pretrained(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/tokenizer/'
  );
  statusEl.innerText = 'Tokenizer loaded';

  textEncoder = await loadModelWithStatus(
    'text_encoder',
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/text_encoder/model.onnx'
  );

  unet = await loadModelWithStatus(
    'unet',
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx'
  );

  vaeDecoder = await loadModelWithStatus(
    'vae_decoder',
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/vae_decoder/model.onnx'
  );

  statusEl.innerText = 'All models loaded!';
  generateBtn.disabled = false;
}

// Generate random latent for demo
function initLatent(size=64){
  const latent = new Float32Array(size*size*4);
  for(let i=0;i<latent.length;i++) latent[i]=Math.random()*2-1;
  return latent;
}

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

// Main generation (simplified)
async function generateImage(prompt){
  if(!textEncoder || !unet || !vaeDecoder || !tokenizer) await loadModels();

  statusEl.innerText = 'Generating embeddings...';
  const encoded = await tokenizer.encode(prompt);
  const tokenIds = encoded.ids.map(x=>BigInt(x));
  const inputIds = new ort.Tensor('int64', BigInt64Array.from(tokenIds), [1, tokenIds.length]);
  const embeddings = (await textEncoder.run({ input_ids: inputIds })).last_hidden_state;

  statusEl.innerText = 'Running diffusion...';
  let latent = initLatent(64*64*4);
  for(let t=0;t<5;t++){
    const unetOut = await unet.run({
      sample: latent,
      timestep: new Float32Array([t]),
      encoder_hidden_states: embeddings
    });
    for(let i=0;i<latent.length;i++) latent[i]-=0.1*unetOut.sample[i];
  }

  statusEl.innerText = 'Decoding image...';
  const decoded = await vaeDecoder.run({ latents: latent });
  drawCanvas(decoded.latents, document.getElementById('canvas'));

  statusEl.innerText = 'Generation complete!';
}

// Register service worker
if('serviceWorker' in navigator){
  navigator.serviceWorker.register('/sw.js').then(()=>console.log('SW registered'));
}

// Button click
generateBtn.addEventListener('click',()=>{
  const prompt = document.getElementById('prompt').value;
  generateImage(prompt);
});

// Load models immediately on page load
loadModels();
