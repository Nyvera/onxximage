import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/transformers/dist/transformers.min.js';

let textEncoder, unet, vaeDecoder, tokenizer;
const statusEl = document.getElementById('status');
const progressEl = document.getElementById('progress');
const generateBtn = document.getElementById('generateBtn');

// Helper: fetch ONNX with progress
async function fetchModelWithProgress(url, name) {
  statusEl.innerText = `Downloading ${name}...`;
  const res = await fetch(url);
  const contentLength = res.headers.get('Content-Length');
  if (!contentLength) {
    const arrayBuffer = await res.arrayBuffer();
    statusEl.innerText = `${name} downloaded`;
    return arrayBuffer;
  }
  const total = parseInt(contentLength, 10);
  const reader = res.body.getReader();
  let received = 0;
  const chunks = [];
  while(true){
    const {done, value} = await reader.read();
    if(done) break;
    chunks.push(value);
    received += value.length;
    progressEl.value = (received/total)*100;
  }
  const arrayBuffer = new Uint8Array(chunks.reduce((acc, val)=>acc.concat(Array.from(val)), []));
  statusEl.innerText = `${name} downloaded`;
  progressEl.value = 0;
  return arrayBuffer.buffer;
}

// Load ONNX model from fetched array buffer
async function loadModelFromURL(url, name){
  const buffer = await fetchModelWithProgress(url, name);
  const model = await ort.InferenceSession.create(buffer);
  statusEl.innerText = `${name} loaded`;
  return model;
}

// Load all models + tokenizer
async function loadModels() {
  statusEl.innerText = 'Loading tokenizer...';
  tokenizer = await AutoTokenizer.from_pretrained(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/tokenizer/'
  );
  statusEl.innerText = 'Tokenizer loaded';

  textEncoder = await loadModelFromURL(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/text_encoder/model.onnx',
    'text_encoder'
  );
  unet = await loadModelFromURL(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx',
    'unet'
  );
  vaeDecoder = await loadModelFromURL(
    'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/vae_decoder/model.onnx',
    'vae_decoder'
  );

  statusEl.innerText = 'All models loaded!';
  generateBtn.disabled = false;
}

// Remaining functions (latent init, drawCanvas, generateImage) are same as before
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

if('serviceWorker' in navigator){
  navigator.serviceWorker.register('/sw.js').then(()=>console.log('SW registered'));
}

generateBtn.addEventListener('click',()=>{
  const prompt = document.getElementById('prompt').value;
  generateImage(prompt);
});

loadModels();
