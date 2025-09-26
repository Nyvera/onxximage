const CACHE_NAME = 'sd-turbo-cache-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/app.js',

  // ONNX models
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/text_encoder/model.onnx',
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/unet/model.onnx',
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/vae_encoder/model.onnx',
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/vae_decoder/model.onnx',

  // Tokenizer files
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/tokenizer/vocab.json',
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/tokenizer/merges.txt',
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/tokenizer/tokenizer_config.json',
  'https://huggingface.co/onnxruntime/sd-turbo/resolve/main/tokenizer/special_tokens_map.json',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});
