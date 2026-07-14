# Glance ML Internship: Multimodal Fashion & Context Retrieval

## 1. Approaches & Trade-offs
When designing a multimodal retrieval system for fashion, the core challenge is **compositionality** (the "bag-of-words" problem). Models like CLIP are great at general semantic matching but struggle to bind specific attributes (e.g. color) to specific garments (e.g. shirt vs pants) when both exist in an image.

### Approach 1: Vanilla CLIP (Baseline)
- **What it is**: Embed the full query and full image using CLIP and compute cosine similarity.
- **Trade-offs**: Fast and requires no extra processing. However, it fails complex queries. "A red tie and white shirt" will retrieve "A white tie and red shirt" with a near-identical score.

### Approach 2: VLM Post-Reranking
- **What it is**: Use Vanilla CLIP to fetch the top 50 images, then pass them to a zero-shot VLM (like LLaVA or GPT-4V) to score them based on strict adherence to the query.
- **Trade-offs**: Extremely accurate, but slow and expensive at query time. Not scalable for high QPS (Queries Per Second).

### Approach 3: Region-Based Indexing with Semantic Query Decomposition (Chosen)
- **What it is**: 
  1. Parse the natural language query into `context` (environment) and `items` (specific clothing).
  2. Index full images in a `context` vector space.
  3. Crop individual garments using bounding boxes (e.g. from Fashionpedia or GroundingDINO) and index them in an `item` vector space.
  4. At query time, search both spaces and aggregate the scores.
- **Trade-offs**: Requires more storage (multiple vectors per image) and upfront processing (cropping). However, it entirely solves the compositionality problem at the vector level without incurring the latency of a VLM at query time.

## 2. Chosen Architecture: Region-Based Indexing
We implemented **Approach 3**. 

### Indexing Pipeline
- We parse the Fashionpedia dataset to extract bounding boxes for all annotated clothing items.
- We generate a `Context Embedding` using CLIP for the full image.
- We crop the bounding boxes, generate `Item Embeddings` using CLIP for each garment, and store them in FAISS.

### Retrieval Pipeline
- We use a local LLM (e.g. Mistral 7B quantized) to extract structured JSON from the user query: `{"context": "modern office", "items": ["professional business attire"]}`. 
- We search the FAISS `context_index` with the context string, and the `item_index` with each item string.
- The scores are aggregated. This ensures that an image ranks highly ONLY if its overall scene matches the context AND it contains isolated garments that match the item descriptions.

## 3. Codebase Structure
- `src/indexer/`: Scripts for parsing COCO annotations, cropping images, and generating FAISS indices.
- `src/retriever/`: Logic for the LLM Query Parser and the Multi-Index Search aggregator.
- `src/vector_store/`: Wrappers for FAISS.
- `run_indexer.py` & `run_retriever.py`: Entry points.

## 4. Future Work

### a. Adding Locations and Weather
- **Location**: We can add an external API (like Google Places) to parse entities in the query. If a specific city is mentioned ("New York"), we can filter the FAISS search using metadata tags (if the images have geotags), rather than relying purely on visual embeddings.
- **Weather**: We can train a lightweight classifier on top of the CLIP embeddings to predict weather (Sunny, Raining, Snowing). We can add this as structured metadata in FAISS and apply boolean pre-filtering during the search.

### b. Improving Precision
- **Dynamic Weighting**: Currently, we use a fixed `alpha` weight (e.g., 0.5 for context, 0.5 for items). We can use an ML model (or the LLM parser) to dynamically assign weights based on the query. If the query is "a red hat", the context weight should be 0.
- **Zero-Shot Crop Generation**: In a production environment without ground-truth bounding boxes, we would deploy a fast zero-shot object detector like GroundingDINO or FastSAM at ingestion time to automatically generate the garment crops.
