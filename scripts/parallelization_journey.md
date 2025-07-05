# The Journey to a stable parallelization

This whitepaper discusses some of the hiccups that I got while developing a stable parallelization structure for the `build_vector_db.py` script.

The initial task was to build a vector database from a dataset of 5,000 documents. A simple, sequential script took 2-3 hours approximately. This motivated me to leverage multi-core CPU threads to parallelize the pipeline and drastically reduce this build time.

## Attempt 1: The Leaked Semaphore Warning
The first attempt involved using two different parallel processing libraries back-to-back:

- Optical character recognition (OCR) phase: Use [Python's `concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor) to run OCR on multiple files at once.
- Encoding Phase: Use the [`sentence-transformers` library's built-in `start_multi_process_pool()`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.start_multi_process_pool) for encoding.

Result: The script worked but produced a `UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown`.

Root Cause: This warning was a symptom of a resource conflict. The `ProcessPoolExecutor` did not perfectly clean up its low-level system resources (semaphores) before the `sentence-transformers` pool tried to initialize its own. This conflict between two different multiprocessing implementations created an unstable state.

## Attempt 2: The CUDA out of memory Error
To solve the semaphore leak, the next approach was to use a single, consistent `ProcessPoolExecutor` for both phases. This involved creating a new `SentenceTransformer` model instance inside each worker process for the encoding phase.

Result: A fatal `RuntimeError: CUDA error: out of memory.`

Root Cause: This approach was too aggressive for my local machine. Each worker process tried to load a full copy of the GPU-accelerated model into the GPU's limited VRAM simultaneously. If the model required 1GB of VRAM and we had 4 workers, the script instantly demanded 4GB of VRAM, overwhelming the GPU's capacity and causing a crash.

## Attempt 3: The BrokenProcessPool and HTTP 429 Errors
The next logical step was to control the number of workers with a `MAX_WORKERS` setting. However, this led to two new problems:

**BrokenProcessPool**: This error indicated that a worker process was being terminated abruptly by the operating system, almost always because the system was running out of main memory (RAM). Pre-loading all 5,000 documents into memory before starting the workers was the cause.

**HTTP Error 429**: Too Many Requests: Even with fewer workers, initializing the SentenceTransformer model in each one caused all workers to ping the Hugging Face servers simultaneously to check for model updates, leading to IP throttling.

## The final, stable solution: The Worker-Initializer Pattern
By using Python's core `multiprocessing.Pool` with a special initializer function (init_ocr_worker) that instantiates an OCR model we managed to clear all of the remaining leaked semaphores. While maintaining a vanilla implementation of the `SentenceTransformer` model, which already provides a batch embedding paradigm, this prevailed as the cleanest solution to reducing the build time as of yet.

Why This Works:

No Resource Conflicts: The initializer function loads the complex `EasyOCR` model only once per worker process when it's created. The worker then reuses this single instance for all its tasks. This completely avoids the repeated, conflicting resource allocation that caused the semaphore leaks and memory errors. The workers read their own files, so only a small number of documents are in RAM at any given time, preventing the BrokenProcessPool error.

On top of that no network throttling as the `SentenceTransformer` model is initialized only once in the main process, avoiding the HTTP 429 error.
