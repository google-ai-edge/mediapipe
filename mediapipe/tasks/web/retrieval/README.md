# MediaPipe Tasks Retrieval Package

This package contains the retrieval tasks for MediaPipe.

## Semantic Retriever

The MediaPipe Semantic Retriever automates vector similarity search and index
building on the web. It combines in-memory vector storage with embedding-based
retrieval, allowing you to quickly build multi-modal semantic search into your
web apps.

For more information, refer to the [Semantic Retriever](https://developers.google.com/mediapipe/solutions/retrieval/semantic_retriever/web_js)
documentation.

```javascript
import { SemanticRetriever, InMemoryVectorStore, UniversalEmbedder, FilesetResolver } from "@mediapipe/tasks-retrieval";

const retrieval = await FilesetResolver.forRetrievalTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-retrieval@latest/wasm");

const embedder = await UniversalEmbedder.createFromOptions(retrieval, {
  baseOptions: {
    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/universal_embedder/embededder.litertlm"
  }
});

const semanticRetriever = await SemanticRetriever.createFromComponents(retrieval, {
  vectorStore: new InMemoryVectorStore(),
  providers: [embedder.getProvider()]
});

await semanticRetriever.insertDocument("doc1", "How to compute semantic distance on the web");
await semanticRetriever.insertDocument("doc2", "Best vector search libraries for JavaScript");

const results = await semanticRetriever.retrieve("What is vector search in JS?", { limit: 1 });
console.log(`Similarity Score: ${results[0].score}, Text Data: ${results[0].textData}`);
```

## Universal Embedder

The MediaPipe Universal Embedder extracts versatile feature embeddings from
multi-modal inputs – including text, image, and audio - all within the same
embedding space.

For more information, refer to the [Universal Embedder](https://developers.google.com/mediapipe/solutions/retrieval/universal_embedder/web_js)
documentation.

```javascript
import { UniversalEmbedder, FilesetResolver } from "@mediapipe/tasks-retrieval";

const retrieval = await FilesetResolver.forRetrievalTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-retrieval@latest/wasm");
const universalEmbedder = await UniversalEmbedder.createFromOptions(
  retrieval,
  {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/universal_embedder/embededder.litertlm"
    }
  }
);

// Embed text
const textEmbeddingResult = universalEmbedder.embedText("The quick brown fox jumps over the lazy dog");
console.log("Text embedding:", textEmbeddingResult.embeddings[0].floatEmbedding);

// Embed images directly from the DOM
const image = document.getElementById("myImage");
const imageEmbeddingResult = universalEmbedder.embedImage(image);

// Compute similarity
const similarity = UniversalEmbedder.cosineSimilarity(
  textEmbeddingResult.embeddings[0],
  imageEmbeddingResult.embeddings[0]
);
```

### Privacy Notice

Last modified: June 5, 2026

When you use MediaPipe Tasks, processing of the input data (e.g. images, video,
text) takes place on device, and MediaPipe does not send that input data to
Google servers. As a result, you can use our MediaPipe Tasks APIs for
processing data that should not leave the device.

MediaPipe Tasks APIs send metrics about the performance and utilization of the
APIs in your app to Google. Google uses this metrics data to measure
performance, usage, debug, maintain and improve the MediaPipe Tasks, as further
described in our [Privacy Policy](https://policies.google.com/privacy).

**You are responsible for obtaining informed consent from your app users about
Google's processing of MediaPipe metrics data as required by applicable law.**
