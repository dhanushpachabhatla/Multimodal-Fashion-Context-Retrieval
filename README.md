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
  3. Crop individual garments using bounding boxes (e.g. from Fashionpedia) and index them in an `item` vector space.
  4. At query time, search both spaces and aggregate the scores.
- **Trade-offs**: Requires more storage (multiple vectors per image) and upfront processing (cropping). However, it entirely solves the compositionality problem at the vector level without incurring the latency of a VLM at query time.

## 2. Chosen Architecture: Region-Based Indexing
We implemented **Approach 3**. 

### Indexing Pipeline (`run_indexer.py`)
- We parsed a random subset of 1,158 images from the Fashionpedia dataset to extract bounding boxes for annotated clothing items.
- Generated `Context Embeddings` using `openai/clip-vit-base-patch32` for the full image.
- Generated `Item Embeddings` for each cropped garment and stored them independently in FAISS indices.

### Retrieval Pipeline (`run_retriever.py` & `interactive_search.py`)
- We use a local LLM (Mistral 7B) to extract structured JSON from the user query: `{"context": "modern office", "items": ["professional business attire"]}`. 
- We search the FAISS `context_index` with the context string, and the `item_index` with each item string.
- The scores are aggregated. We implemented **Dynamic Weighting**: if a user does not ask for specific items, the system shifts 100% of the mathematical weight to the context score (and vice versa) to prevent 0.0 scores from deflating exact matches.

## 3. Evaluation & Metrics
To move beyond manual "vibe" checks, we built an automated evaluation pipeline (`evaluate.py`) that strictly checks if the top 5 retrieved vectors actually correspond to the correct ground-truth Fashionpedia categories.

**The Strict Combinatorial Test:**
We formulated 19 test queries ranging from Easy (1 item) to Hard (3+ items). We explicitly generated these queries based on combinations that are **guaranteed to exist** in our 1,158-image subset (e.g., we know for an absolute fact that an image with a scarf, a blouse, and a skirt exists in our subset).

**Results:**
- **Overall Category Recall@5:** ~58% (11/19)
- **Analysis:** The system achieved a near-perfect 100% on 1-item queries, but failed on the 3+ item queries (e.g. "A person with a scarf, a blouse, and a skirt carrying a bag"). Because we mathematically guaranteed that the target image exists in the subset, these failures explicitly highlight the limitations of independent region scoring (the Attribute Binding Problem). The FAISS vector search simply took the average of the maximum independent crop scores, allowing an image with incredibly high scores for just 2 items to mathematically beat out the single correct image that actually contained all 4 items!

## 4. Experiments & Engineering Hurdles
Throughout the assignment, we faced and solved several real-world ML engineering problems:
- **Transformers CVE-2025-32434 Crash:** HuggingFace recently banned loading `.bin` weights on older PyTorch versions due to security vulnerabilities, crashing the indexer. We solved this by explicitly fetching `revision="refs/pr/66"` to force the pipeline to use `.safetensors`.
- **LLM Instruction Hallucination:** Our local quantized LLM originally hallucinated items into queries that didn't specify any (e.g., hallucinating "yellow raincoat" just because it saw it in a few-shot prompt). We solved this via strict negative prompting and json formatting constraints.

## 5. Future Work
### a. Adding Locations and Weather
- **Location**: We can add an external API (like Google Places) to parse entities in the query. If a specific city is mentioned ("New York"), we can filter the FAISS search using metadata tags (if the images have geotags), rather than relying purely on visual embeddings.
- **Weather**: We can train a lightweight classifier on top of the CLIP embeddings to predict weather (Sunny, Raining, Snowing). We can add this as structured metadata in FAISS and apply boolean pre-filtering during the search.

### b. Improving Precision in Production
- **Zero-Shot Crop Generation**: In a production environment without ground-truth bounding boxes, we would deploy a fast zero-shot object detector like GroundingDINO or FastSAM at ingestion time to automatically generate the garment crops.
- **Contrastive Reranker**: To solve the 58% bottleneck on multi-item queries, we would deploy an ALBEF reranker or a lightweight Vision-Language Model at the very end of the pipeline to explicitly re-score the Top 50 images for exact attribute binding.
