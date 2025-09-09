#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // For getline and strdup
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <hs/hs.h>
#include <sys/stat.h>
#include <unistd.h>

// CUDA includes for GPU mode
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// cuDF RAPIDS includes for GPU regex
#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/types.hpp>

// CUDA error checking macro with detailed error reporting
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error detected. %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Additional macro for checking CUDA errors after kernel launches
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Error detected. %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// --- Data Structures ---

// Execution Mode
typedef enum {
    MODE_CPU,
    MODE_GPU
} execution_mode_t;

// Configuration Structure
typedef struct {
    execution_mode_t mode;
    char* rules_file;
    char* input_file;
    int num_threads;  // Only used for CPU mode
    int gpu_streams;  // GPU: number of concurrent pattern workers (logical)
    int gpu_wave;      // GPU: patterns per batch
    int gpu_chunk_mb;  // GPU: per-chunk input size in MB
} config_t;

/**
 * @struct MatchContext
 * @brief Context structure passed to the Hyperscan match event handler.
 */
typedef struct {
    int* matches;           // Array to store IDs of matched rules.
    int match_count;        // Number of matches found for the current line.
    int match_capacity;     // Allocated capacity of the matches array.
} MatchContext;

/**
 * @struct ThreadData
 * @brief Data structure to pass information to each worker thread.
 */
typedef struct {
    int thread_id;                 // Unique identifier for the thread.
    char** lines;                  // Pointer to the array of all input lines.
    unsigned int* line_lengths;    // Pointer to the array of all line lengths.
    long start_line;               // Starting line index for this thread.
    long end_line;                 // Ending line index for this thread.
    hs_database_t* database;       // Pointer to the compiled Hyperscan database.
    hs_scratch_t* scratch;         // Per-thread scratch space for Hyperscan.
    char*** thread_results;        // 2D array: [line_index][match_list] for this thread's lines
    long total_matches;            // Total number of matches found by this thread.
} ThreadData;


// --- Forward Declarations ---
int run_cpu_mode(const config_t* config);
int run_gpu_mode(const config_t* config);

// cuDF helper functions for GPU mode
#ifdef __cplusplus
extern "C" {
#endif

// Build device strings column from host vector<string>
static std::unique_ptr<cudf::column>
make_device_strings(const std::vector<std::string>& h, rmm::cuda_stream_view stream) {
    using size_type = cudf::size_type;
    const size_type n = static_cast<size_type>(h.size());

    // Handle edge case of empty input
    if (n == 0) {
        return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    }

    std::vector<int32_t> h_offsets(n + 1, 0);
    size_t total_chars = 0;
    for (size_t i = 0; i < h.size(); ++i) {
        total_chars += h[i].size();
        h_offsets[i + 1] = static_cast<int32_t>(total_chars);
    }
    
    std::vector<char> h_chars;
    h_chars.reserve(total_chars);
    for (auto& s : h) h_chars.insert(h_chars.end(), s.begin(), s.end());

    // Allocate device memory with explicit error checking
    rmm::device_uvector<int32_t> d_offsets(n + 1, stream);
    rmm::device_uvector<char> d_chars(total_chars, stream);

    CUDA_CHECK(cudaMemcpyAsync(d_offsets.data(), h_offsets.data(),
                               (n + 1) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream.value()));
    if (total_chars > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_chars.data(), h_chars.data(), total_chars,
                                   cudaMemcpyHostToDevice, stream.value()));
    }
    
    // Synchronize to ensure data transfer is complete
    CUDA_CHECK(cudaStreamSynchronize(stream.value()));

    auto null_mask = rmm::device_buffer{0, stream};
    cudf::size_type null_count = 0;

    auto offsets_buf = d_offsets.release();
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        n + 1,
        std::move(offsets_buf),
        rmm::device_buffer{0, stream},
        0);
    auto chars_buf = d_chars.release();
    return cudf::make_strings_column(
        n,
        std::move(offsets_col),
        std::move(chars_buf),
        null_count,
        std::move(null_mask));
}

__global__ void add_true_to_counts(const uint8_t* __restrict__ vals,
                                   int n,
                                   int* __restrict__ counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) counts[i] += (vals[i] != 0);
}

#ifdef __cplusplus
}
#endif

// --- Utility Functions ---

/**
 * @brief Prints an error message and exits the program.
 */
void fail(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(EXIT_FAILURE);
}

/**
 * @brief Print usage information.
 */
void print_usage(const char* program_name) {
    printf("Usage: %s --mode <cpu|gpu> --rules <rules_file> --input <input_file> [--threads <n>] [--gpu-streams <n>] [--gpu-wave <n>] [--gpu-chunk-mb <n>]\n", program_name);
    printf("\nRequired arguments:\n");
    printf("  --mode      <cpu|gpu>       Processing mode (CPU or GPU)\n");
    printf("  --rules     <rules_file>    Path to the rules file\n");
    printf("  --input     <input_file>    Path to the input file\n");
    printf("\nOptional arguments:\n");
    printf("  --threads   <num_threads>   Number of threads (required for CPU mode)\n");
    printf("  --gpu-streams <n>        GPU: logical concurrency across patterns (default 6)\n");
    printf("  --gpu-wave    <n>        GPU: patterns per batch (default 48)\n");
    printf("  --gpu-chunk-mb <n>       GPU: max input MB per chunk (default 128)\n");
    printf("\nOutput files are automatically generated in the results/ directory:\n");
    printf("  Results_HW3_MCC_030402_401106039_{CPU/GPU}_{DataSet}_{NumThreads/Library}.txt\n");
    printf("  Results_HW3_MCC_030402_401106039_{CPU/GPU}_{DataSet}_{Hyperscan/GPULibrary}.csv\n");
    printf("\nExample:\n");
    printf("  %s --mode cpu --rules rules.txt --input set1.txt --threads 4\n", program_name);
    printf("  %s --mode gpu --rules rules.txt --input set1.txt --gpu-streams 6 --gpu-wave 48 --gpu-chunk-mb 128\n", program_name);
    exit(EXIT_FAILURE);
}

/**
 * @brief Generate automatic output filename based on configuration.
 */
char* generate_output_filename(const config_t* config) {
    // Extract dataset name from input file (e.g., "set1.txt" -> "set1")
    const char* input_basename = strrchr(config->input_file, '/');
    if (input_basename) {
        input_basename++; // Skip the '/'
    } else {
        input_basename = config->input_file;
    }
    
    // Remove file extension
    char dataset[256];
    strncpy(dataset, input_basename, sizeof(dataset) - 1);
    dataset[sizeof(dataset) - 1] = '\0';
    char* dot = strrchr(dataset, '.');
    if (dot) {
        *dot = '\0';
    }
    
    // Allocate memory for the filename
    char* filename = (char*)malloc(512);
    if (!filename) {
        fprintf(stderr, "Error: Memory allocation failed for output filename\n");
        exit(EXIT_FAILURE);
    }
    
    if (config->mode == MODE_CPU) {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_CPU_%s_%d.txt", 
                 dataset, config->num_threads);
    } else {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_GPU_%s_CUDA.txt", 
                 dataset);
    }
    
    return filename;
}

/**
 * @brief Generate performance CSV filename based on configuration.
 */
char* generate_performance_filename(const config_t* config, const char* input_filename) {
    // Extract dataset name from input filename
    const char* dataset_name = strrchr(input_filename, '/');
    if (dataset_name) {
        dataset_name++; // Skip the '/'
    } else {
        dataset_name = input_filename;
    }
    
    // Remove extension from dataset name
    char* dataset_clean = strdup(dataset_name);
    char* dot = strrchr(dataset_clean, '.');
    if (dot) *dot = '\0';
    
    char* filename = (char*)malloc(512);
    if (config->mode == MODE_CPU) {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_CPU_%s_Hyperscan.csv", 
                 dataset_clean);
    } else {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_GPU_%s_CUDA.csv", 
                 dataset_clean);
    }
    
    free(dataset_clean);
    return filename;
}

/**
 * @brief Parse command line arguments.
 */
config_t parse_arguments(int argc, char* argv[]) {
    config_t config{};
    // Defaults for GPU tuning
    config.gpu_streams = 6;
    config.gpu_wave    = 48;
    config.gpu_chunk_mb= 128;

    
    if (argc < 5) {  // Minimum required arguments for GPU mode
        print_usage(argv[0]);
    }
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "cpu") == 0) {
                config.mode = MODE_CPU;
            } else if (strcmp(argv[i + 1], "gpu") == 0) {
                config.mode = MODE_GPU;
            } else {
                fprintf(stderr, "ERROR: Invalid mode '%s'. Use 'cpu' or 'gpu'.\n", argv[i + 1]);
                print_usage(argv[0]);
            }
            i++; // Skip next argument
        } else if (strcmp(argv[i], "--rules") == 0 && i + 1 < argc) {
            config.rules_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            config.input_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config.num_threads = atoi(argv[i + 1]); i++;
        } else if (strcmp(argv[i], "--gpu-streams") == 0 && i + 1 < argc) {
            config.gpu_streams = atoi(argv[i + 1]); i++;
        } else if (strcmp(argv[i], "--gpu-wave") == 0 && i + 1 < argc) {
            config.gpu_wave = atoi(argv[i + 1]); i++;
        } else if (strcmp(argv[i], "--gpu-chunk-mb") == 0 && i + 1 < argc) {
            config.gpu_chunk_mb = atoi(argv[i + 1]); i++;
            config.num_threads = atoi(argv[i + 1]);
            if (config.num_threads <= 0) {
                fprintf(stderr, "ERROR: Number of threads must be a positive integer.\n");
                print_usage(argv[0]);
            }
            i++;
        }
    }
    
    // Validate required arguments
    if (!config.rules_file || !config.input_file) {
        fprintf(stderr, "ERROR: Missing required arguments.\n");
        print_usage(argv[0]);
    }
    
    if (config.mode == MODE_CPU && config.num_threads == 0) {
        fprintf(stderr, "ERROR: --threads argument is required for CPU mode.\n");
        print_usage(argv[0]);
    }
    
    return config;
}

/**
 * @brief Reads all lines from a file into a dynamically allocated array.
 */
char** read_lines_from_file(const char* filename, long* line_count, unsigned int** line_lengths, long* total_bytes) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("fopen failed");
        fail("Could not open file.");
    }

    // Get file size for total_bytes metric
    struct stat st;
    if (stat(filename, &st) == 0) {
        *total_bytes = st.st_size;
    } else {
        *total_bytes = 0; // Fallback
    }

    long capacity = 1024;
    char** lines = (char**)malloc(capacity * sizeof(char*));
    if (!lines) fail("Failed to allocate memory for lines.");

    *line_count = 0;
    char* line_buffer = NULL;
    size_t buffer_size = 0;

    while (getline(&line_buffer, &buffer_size, file) != -1) {
        if (*line_count >= capacity) {
            capacity *= 2;
            lines = (char**)realloc(lines, capacity * sizeof(char*));
            if (!lines) fail("Failed to reallocate memory for lines.");
        }
        // Strip newline characters
        line_buffer[strcspn(line_buffer, "\r\n")] = 0;
        lines[*line_count] = strdup(line_buffer);
        if (!lines[*line_count]) fail("Failed to duplicate line.");
        (*line_count)++;
    }

    free(line_buffer);
    fclose(file);

    // Create the line lengths array
    *line_lengths = (unsigned int*)malloc(*line_count * sizeof(unsigned int));
    if (!*line_lengths) fail("Failed to allocate memory for line lengths.");
    for (long i = 0; i < *line_count; i++) {
        (*line_lengths)[i] = strlen(lines[i]);
    }

    return lines;
}

// --- Hyperscan Match Callback ---

/**
 * @brief Hyperscan match event handler.
 */
static int onMatch(unsigned int id, unsigned long long from, unsigned long long to,
                   unsigned int flags, void* ctx) {
    (void)from;   // Suppress unused parameter warning
    (void)to;     // Suppress unused parameter warning
    (void)flags;  // Suppress unused parameter warning
    
    MatchContext* context = (MatchContext*)ctx;

    // Resize matches array if needed
    if (context->match_count >= context->match_capacity) {
        context->match_capacity *= 2;
        context->matches = (int*)realloc(context->matches, context->match_capacity * sizeof(int));
        if (!context->matches) {
            fail("Failed to reallocate memory for matches in callback.");
        }
    }

    context->matches[context->match_count++] = id;
    return 0; // Continue scanning
}


// --- Worker Thread ---

/**
 * @brief The main function for each worker thread.
 */
void* worker_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    data->total_matches = 0;

    // Allocate scratch space for this thread
    hs_error_t scratch_err = hs_alloc_scratch(data->database, &data->scratch);
    if (scratch_err != HS_SUCCESS) {
        fprintf(stderr, "Thread %d: Failed to allocate scratch space. Error: %d\n", data->thread_id, scratch_err);
        return NULL;
    }

    // Allocate 2D result array for this thread's lines
    long thread_line_count = data->end_line - data->start_line;
    data->thread_results = (char***)malloc(thread_line_count * sizeof(char**));
    if (!data->thread_results) {
        fprintf(stderr, "Thread %d: Failed to allocate thread results array.\n", data->thread_id);
        return NULL;
    }

    for (long i = data->start_line; i < data->end_line; i++) {
        long local_index = i - data->start_line; // Local index within this thread's range
        
        // Initialize context for this line's scan
        MatchContext context;
        context.match_capacity = 16; // Initial capacity
        context.matches = (int*)malloc(context.match_capacity * sizeof(int));
        if (!context.matches) {
             data->thread_results[local_index] = (char**)malloc(sizeof(char*));
             data->thread_results[local_index][0] = strdup(""); // Store empty result on failure
             continue;
        }
        context.match_count = 0;

        // Perform the scan
        hs_error_t err = hs_scan(data->database, data->lines[i], data->line_lengths[i], 0,
                                 data->scratch, onMatch, &context);

        if (err != HS_SUCCESS) {
            free(context.matches);
            data->thread_results[local_index] = (char**)malloc(sizeof(char*));
            data->thread_results[local_index][0] = strdup(""); // Store empty result on error
            continue;
        }

        data->total_matches += context.match_count;

        // Format the result string with ZERO-INDEXED pattern numbers (e.g., "0,3,9")
        if (context.match_count > 0) {
            // A rough estimation for buffer size: 10 chars per match ID + commas
            size_t buffer_size = context.match_count * 10;
            char* result_buffer = (char*)malloc(buffer_size);
            if (!result_buffer) {
                data->thread_results[local_index] = (char**)malloc(sizeof(char*));
                data->thread_results[local_index][0] = strdup("");
            } else {
                int offset = 0;
                for (int j = 0; j < context.match_count; j++) {
                    // Use ZERO-INDEXED pattern numbers (Hyperscan IDs start from 0)
                    offset += snprintf(result_buffer + offset, buffer_size - offset,
                                       "%d%s", context.matches[j], (j == context.match_count - 1) ? "" : ",");
                }
                data->thread_results[local_index] = (char**)malloc(sizeof(char*));
                data->thread_results[local_index][0] = result_buffer;
            }
        } else {
            // If no matches, store an empty string
            data->thread_results[local_index] = (char**)malloc(sizeof(char*));
            data->thread_results[local_index][0] = strdup("");
        }

        free(context.matches);
    }

    // Free scratch space allocated by this thread
    if (data->scratch) {
        hs_free_scratch(data->scratch);
    }

    return NULL;
}


// --- CPU Mode Implementation ---

int run_cpu_mode(const config_t* config) {
    // --- 1. Read and Compile Rules ---
    printf("Reading and compiling regex rules from '%s'...\n", config->rules_file);
    long pattern_count = 0;
    long ignored_total_bytes;
    unsigned int* ignored_lengths;
    char** patterns = read_lines_from_file(config->rules_file, &pattern_count, &ignored_lengths, &ignored_total_bytes);
    free(ignored_lengths);

    unsigned int* ids = (unsigned int*)malloc(pattern_count * sizeof(unsigned int));
    unsigned int* flags = (unsigned int*)malloc(pattern_count * sizeof(unsigned int));
    if (!ids || !flags) fail("Failed to allocate memory for rule IDs/flags.");

    for (long i = 0; i < pattern_count; i++) {
        ids[i] = i; // Hyperscan uses 0-indexed IDs
        flags[i] = 0; // No flags
    }

    hs_database_t* database;
    hs_compile_error_t* compile_err;
    hs_platform_info_t platform;
    
    // Populate platform information for optimal compilation
    hs_error_t platform_err = hs_populate_platform(&platform);
    if (platform_err != HS_SUCCESS) {
        printf("Warning: Could not populate platform info, using default settings.\n");
    }
    
    hs_error_t err = hs_compile_multi((const char* const*)patterns, flags, ids, pattern_count,
                                      HS_MODE_BLOCK, (platform_err == HS_SUCCESS) ? &platform : NULL, 
                                      &database, &compile_err);

    if (err != HS_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to compile pattern: %s\n", compile_err->message);
        hs_free_compile_error(compile_err);
        fail("Hyperscan compilation failed.");
    }
    
    if (!database) {
        fail("Database compilation succeeded but database is NULL.");
    }
    
    printf("Compilation successful. %ld rules loaded.\n", pattern_count);

    // --- 2. Read Input Data ---
    printf("Reading input data from '%s'...\n", config->input_file);
    long line_count = 0;
    long total_bytes = 0;
    unsigned int* line_lengths;
    char** lines = read_lines_from_file(config->input_file, &line_count, &line_lengths, &total_bytes);
    printf("Read %ld lines, total size: %.2f MB.\n", line_count, (double)total_bytes / (1024 * 1024));

    // --- 3. Setup and Run Threads ---
    printf("Processing with %d worker thread(s)...\n", config->num_threads);
    pthread_t* threads = (pthread_t*)malloc(config->num_threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(config->num_threads * sizeof(ThreadData));
    if (!threads || !thread_data) {
        fail("Failed to allocate memory for thread management.");
    }

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    long lines_per_thread = line_count / config->num_threads;
    long remaining_lines = line_count % config->num_threads;
    long current_line = 0;

    for (int i = 0; i < config->num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].lines = lines;
        thread_data[i].line_lengths = line_lengths;
        thread_data[i].database = database;
        thread_data[i].thread_results = NULL; // Will be allocated by each thread
        thread_data[i].total_matches = 0;
        thread_data[i].scratch = NULL; // Let each thread allocate its own scratch

        // Distribute lines
        thread_data[i].start_line = current_line;
        long chunk_size = lines_per_thread + (i < remaining_lines ? 1 : 0);
        thread_data[i].end_line = current_line + chunk_size;
        current_line += chunk_size;

        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }

    // --- 4. Join Threads and Collect Results ---
    long total_matches = 0;
    for (int i = 0; i < config->num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_matches += thread_data[i].total_matches;
    }

    // --- 5. Merge Thread Results into Final Output Array ---
    char** all_results = (char**)malloc(line_count * sizeof(char*));
    if (!all_results) {
        fail("Failed to allocate memory for final results.");
    }

    // Copy results from each thread's 2D array to the final output array
    for (int i = 0; i < config->num_threads; i++) {
        long thread_line_count = thread_data[i].end_line - thread_data[i].start_line;
        for (long j = 0; j < thread_line_count; j++) {
            long global_index = thread_data[i].start_line + j;
            all_results[global_index] = strdup(thread_data[i].thread_results[j][0]);
            
            // Free the thread's result memory
            free(thread_data[i].thread_results[j][0]);
            free(thread_data[i].thread_results[j]);
        }
        free(thread_data[i].thread_results);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    printf("Processing completed.\n");

    // --- 6. Calculate Performance Metrics ---
    double elapsed_seconds = (end_time.tv_sec - start_time.tv_sec) +
                             (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    double throughput_input_per_sec = line_count / elapsed_seconds;
    double throughput_mbytes_per_sec = (total_bytes / (1024.0 * 1024.0)) / elapsed_seconds;
    double throughput_match_per_sec = total_matches / elapsed_seconds;
    double latency_ms = (elapsed_seconds * 1000.0) / line_count;

    printf("Performance Metrics:\n");
    printf("  Total Time: %.4f seconds\n", elapsed_seconds);
    printf("  Total Matches: %ld\n", total_matches);
    printf("  Throughput (Input/sec): %.2f\n", throughput_input_per_sec);
    printf("  Throughput (MBytes/sec): %.2f\n", throughput_mbytes_per_sec);
    printf("  Throughput (Match/sec): %.2f\n", throughput_match_per_sec);
    printf("  Latency (ms/input): %.4f\n", latency_ms);

    // --- 7. Write Output Files ---
    char* output_filename = generate_output_filename(config);
    printf("Writing results to '%s'...\n", output_filename);

    // Write match results
    FILE* out_file = fopen(output_filename, "w");
    if (!out_file) fail("Could not open output file for writing.");
    for (long i = 0; i < line_count; i++) {
        fprintf(out_file, "%s\n", all_results[i]);
    }
    fclose(out_file);

    // Write performance metrics
    char* perf_filename = generate_performance_filename(config, config->input_file);
    FILE* perf_file = fopen(perf_filename, "a");
    if (!perf_file) fail("Could not open performance file for writing.");

    // Check if file is empty (new file) to write header
    fseek(perf_file, 0, SEEK_END);
    long file_size = ftell(perf_file);
    if (file_size == 0) {
        // File is empty, write header
        fprintf(perf_file, "threads,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency_ms\n");
    }
    
    fprintf(perf_file, "%d,%.2f,%.2f,%.2f,%.4f\n",
            config->num_threads,
            throughput_input_per_sec,
            throughput_mbytes_per_sec,
            throughput_match_per_sec,
            latency_ms);
    fclose(perf_file);
    
    printf("Results written to '%s' and '%s'\n\n", output_filename, perf_filename);
    free(output_filename);
    free(perf_filename);

    // --- 8. Cleanup ---
    hs_free_database(database);
    for (long i = 0; i < pattern_count; i++) free(patterns[i]);
    free(patterns);
    free(ids);
    free(flags);
    for (long i = 0; i < line_count; i++) {
        free(lines[i]);
        free(all_results[i]);
    }
    free(lines);
    free(line_lengths);
    free(all_results);
    free(threads);
    free(thread_data);

    return EXIT_SUCCESS;
}


// --- GPU Mode Implementation ---

int run_gpu_mode(const config_t* config) {
    printf("Starting GPU mode processing with cuDF/RAPIDS...\n");

    // Memory resources (kept alive for entire scope)
    std::shared_ptr<rmm::mr::cuda_memory_resource> cuda_mr;
    std::shared_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_mr;

    try {
        // --- 1) CUDA & RMM init ---
        CUDA_CHECK(cudaSetDevice(0));
        cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
        // 512MB initial pool (grow as needed)
        pool_mr = std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(cuda_mr.get(), 512UL*1024*1024);
        rmm::mr::set_current_device_resource(pool_mr.get());

        cudaDeviceProp device_prop;
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
        printf("Using GPU: %s\n", device_prop.name);

        // --- 2) Read patterns ---
        printf("Reading regex patterns from '%s'...\n", config->rules_file);
        long pattern_count = 0;
        unsigned int* tmp_len = nullptr;
        long tmp_bytes = 0;
        char** patterns_c = read_lines_from_file(config->rules_file, &pattern_count, &tmp_len, &tmp_bytes);
        free(tmp_len);
        printf("Loaded %ld patterns.\n", pattern_count);

        std::vector<std::string> patterns;
        patterns.reserve(pattern_count);
        for (long i = 0; i < pattern_count; ++i) patterns.emplace_back(patterns_c[i]);

        // --- 3) Read input lines ---
        printf("Reading input data from '%s'...\n", config->input_file);
        long line_count = 0;
        long total_bytes = 0;
        unsigned int* line_lengths = nullptr;
        char** lines = read_lines_from_file(config->input_file, &line_count, &line_lengths, &total_bytes);
        printf("Read %ld lines, total size: %.2f MB.\n", line_count, (double)total_bytes / (1024.0*1024.0));

        // Global results holder: for each line list of matched pattern IDs
        std::vector<std::vector<int>> line_matches(line_count);

        // --- Timers ---
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        double acc_h2d = 0.0, acc_kernel = 0.0, acc_d2h = 0.0;
        long long total_matches = 0;

        // --- 4) Chunking plan ---
        const long max_mb = (config->gpu_chunk_mb > 0) ? config->gpu_chunk_mb : 128;
        const size_t budget = (size_t)max_mb * 1024ULL * 1024ULL;

        std::vector<std::pair<long,long>> chunks;
        chunks.reserve((size_t)( (total_bytes / (budget?budget:1)) + 2 ));
        long s = 0; size_t acc = 0;
        for (long i = 0; i < line_count; ++i) {
            size_t L = (size_t)line_lengths[i];
            if (i> s && (acc + L) > budget) {
                chunks.emplace_back(s, i);
                s = i; acc = 0;
            }
            acc += L;
        }
        if (s < line_count) chunks.emplace_back(s, line_count);

        printf("Chunking: %zu chunk(s), ~%ld MB each (target).\n", chunks.size(), max_mb);

        // Create one CUDA stream for transfers; compute ops will implicitly use the same stream in this build.
        cudaStream_t gpu_stream;
        CUDA_CHECK(cudaStreamCreate(&gpu_stream));
        auto stream = rmm::cuda_stream_view{gpu_stream};

        const int wave = (config->gpu_wave > 0) ? config->gpu_wave : 48;

        // --- 5) Process each chunk ---
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            long lo = chunks[ci].first, hi = chunks[ci].second;
            // Host staging for this chunk
            std::vector<std::string> h_lines;
            h_lines.reserve((size_t)(hi - lo));
            for (long i = lo; i < hi; ++i) h_lines.emplace_back(lines[i]);

            // H2D: build cuDF strings column
            struct timespec h2d_s, h2d_e;
            clock_gettime(CLOCK_MONOTONIC, &h2d_s);
            auto sentences_col = make_device_strings(h_lines, stream);
            cudf::strings_column_view sview{sentences_col->view()};
            int nrows = (int)sview.size();
            CUDA_CHECK(cudaStreamSynchronize(stream.value()));
            clock_gettime(CLOCK_MONOTONIC, &h2d_e);
            acc_h2d += (h2d_e.tv_sec - h2d_s.tv_sec) + (h2d_e.tv_nsec - h2d_s.tv_nsec)/1e9;

            // Patterns in waves
            for (long p0 = 0; p0 < pattern_count; p0 += wave) {
                long p1 = std::min(p0 + (long)wave, pattern_count);

                for (long p = p0; p < p1; ++p) {
                    const std::string& pat = patterns[p];
                    if (pat.empty() || pat.size() > 2048) continue;

                    try {
                        // Compile regex program
                        auto prog = cudf::strings::regex_program::create(pat);

                        // Kernel-ish time: contains_re (GPU)
                        struct timespec k_s, k_e;
                        clock_gettime(CLOCK_MONOTONIC, &k_s);
                        auto bool_col = cudf::strings::contains_re(sview, *prog);
                        CUDA_CHECK(cudaStreamSynchronize(stream.value()));
                        clock_gettime(CLOCK_MONOTONIC, &k_e);
                        acc_kernel += (k_e.tv_sec - k_s.tv_sec) + (k_e.tv_nsec - k_s.tv_nsec)/1e9;

                        // D2H: bring matches back
                        struct timespec d2h_s, d2h_e;
                        clock_gettime(CLOCK_MONOTONIC, &d2h_s);
                        auto bv = bool_col->view();
                        const uint8_t* d = bv.data<uint8_t>();
                        std::vector<uint8_t> h(bv.size());
                        CUDA_CHECK(cudaMemcpyAsync(h.data(), d, h.size(), cudaMemcpyDeviceToHost, stream.value()));
                        CUDA_CHECK(cudaStreamSynchronize(stream.value()));
                        clock_gettime(CLOCK_MONOTONIC, &d2h_e);
                        acc_d2h += (d2h_e.tv_sec - d2h_s.tv_sec) + (d2h_e.tv_nsec - d2h_s.tv_nsec)/1e9;

                        // Accumulate results
                        long chunk_matches = 0;
                        for (int i = 0; i < nrows; ++i) {
                            if (h[i]) {
                                line_matches[lo + i].push_back((int)p);
                                chunk_matches++;
                            }
                        }
                        total_matches += chunk_matches;
                    } catch (const std::exception& e) {
                        fprintf(stderr, "Warning: pattern %ld failed on chunk %zu: %s\n", p, ci, e.what());
                        CUDA_CHECK(cudaStreamSynchronize(stream.value()));
                        continue;
                    }
                } // end for p in wave
            } // end waves per chunk

            // Free the big strings column ASAP
            sentences_col.reset();
            CUDA_CHECK(cudaStreamSynchronize(stream.value()));
        } // end chunks

        // --- Assemble all_results like CPU ---
        printf("Formatting results...\n");
        char** all_results = (char**)malloc(line_count * sizeof(char*));
        if (!all_results) fail("Failed to allocate memory for final results.");

        for (long i = 0; i < line_count; ++i) {
            auto& v = line_matches[i];
            if (v.empty()) {
                all_results[i] = strdup("");
            } else {
                // estimate buffer
                size_t buf = v.size()*10;
                char* out = (char*)malloc(buf);
                if (!out) { all_results[i] = strdup(""); continue; }
                int off=0;
                for (size_t j=0;j<v.size();++j) {
                    off += snprintf(out+off, buf-off, "%d%s", v[j], (j+1==v.size())?"":",");
                }
                all_results[i] = out;
            }
        }

        struct timespec t_end; clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t0.tv_sec) + (t_end.tv_nsec - t0.tv_nsec)/1e9;

        // --- Metrics ---
        double thr_input = line_count / elapsed;
        double thr_mb = (total_bytes / (1024.0*1024.0)) / elapsed;
        double thr_match = (total_matches) / elapsed;
        double latency_ms = (elapsed * 1000.0) / line_count;

        printf("Performance Metrics (GPU):\n");
        printf("  Total Time: %.4f s\n", elapsed);
        printf("  H2D: %.4f s, Kernel: %.4f s, D2H: %.4f s\n", acc_h2d, acc_kernel, acc_d2h);
        printf("  Total Matches: %lld\n", total_matches);
        printf("  Throughput (Input/sec): %.2f\n", thr_input);
        printf("  Throughput (MBytes/sec): %.2f\n", thr_mb);
        printf("  Throughput (Match/sec): %.2f\n", thr_match);
        printf("  Latency (ms/input): %.4f\n", latency_ms);

        // --- Write outputs (same layout) ---
        char* output_filename = generate_output_filename(config);
        printf("Writing results to '%s'...\n", output_filename);
        FILE* out = fopen(output_filename, "w");
        if (!out) fail("Could not open output file for writing.");
        for (long i = 0; i < line_count; ++i) fprintf(out, "%s\\n", all_results[i]);
        fclose(out);

        char* perf_filename = generate_performance_filename(config, config->input_file);
        FILE* pf = fopen(perf_filename, "w");
        if (pf) {
            fprintf(pf, "Mode,DataSet,Library,TotalTime,TotalMatches,InputPerSec,MBPerSec,MatchPerSec,LatencyMs,H2D,Kernel,D2H\\n");
            const char* dataset = strrchr(config->input_file, '/'); dataset = dataset? dataset+1: config->input_file;
            char dataset_clean[256]; strncpy(dataset_clean, dataset, sizeof(dataset_clean)-1); dataset_clean[sizeof(dataset_clean)-1]=0;
            char* dot = strrchr(dataset_clean, '.'); if (dot) *dot = 0;
            fprintf(pf, "GPU,%s,CUDA,%.6f,%lld,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f\\n",
                    dataset_clean, elapsed, total_matches, thr_input, thr_mb, thr_match, latency_ms, acc_h2d, acc_kernel, acc_d2h);
            fclose(pf);
        }

        // Cleanup: free C-strings and arrays
        for (long i = 0; i < pattern_count; ++i) free(patterns_c[i]);
        free(patterns_c);
        for (long i = 0; i < line_count; ++i) { free(all_results[i]); }
        free(all_results);
        free(lines);
        free(line_lengths);

        // Destroy stream and RMM pool
        CUDA_CHECK(cudaStreamDestroy(gpu_stream));
        rmm::mr::set_current_device_resource(cuda_mr.get());
        pool_mr.reset(); cuda_mr.reset();

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU mode error: %s\\n", e.what());
        if (pool_mr) { rmm::mr::set_current_device_resource(cuda_mr.get()); pool_mr.reset(); }
        if (cuda_mr) cuda_mr.reset();
        return EXIT_FAILURE;
    }
}


// --- Main Function ---

int main(int argc, char* argv[]) {
    config_t config = parse_arguments(argc, argv);
    
    printf("High-Performance Regex Matching - Mode: %s\n", config.mode == MODE_CPU ? "CPU" : "GPU");
    
    if (config.mode == MODE_CPU) {
        return run_cpu_mode(&config);
    } else {
        return run_gpu_mode(&config);
    }
}