#include "clahe_modern.h"
#include <iostream>
#include <chrono>
#include <string>
#include <random>
#include <iomanip>

struct BenchmarkConfig {
    int iterations = 100;           // Default iterations for ~15-30 sec runtime
    int image_width = 4096;
    int image_height = 4096;
    int grid_width = 16;
    int grid_height = 16;
    int bins = 256;
    double clip_limit = 2.0;
    std::string input_image = "";   // Empty means generate synthetic
};

cv::Mat generate_synthetic_image(int width, int height) {
    cv::Mat image(height, width, CV_8UC1);
    
    // Create a combination of gradient and noise for realistic testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> noise_dist(0, 50);
    
    for (int y = 0; y < height; ++y) {
        unsigned char* row = image.ptr<unsigned char>(y);
        for (int x = 0; x < width; ++x) {
            // Create radial gradient from center
            double dx = (x - width/2.0) / (width/2.0);
            double dy = (y - height/2.0) / (height/2.0);
            double distance = std::sqrt(dx*dx + dy*dy);
            
            // Base gradient value
            int base_value = static_cast<int>(127 * (1.0 - std::min(1.0, distance)));
            
            // Add some structured patterns
            int wave_x = static_cast<int>(30 * std::sin(x * 0.01));
            int wave_y = static_cast<int>(30 * std::cos(y * 0.01));
            
            // Add random noise
            int noise = noise_dist(gen) - 25;
            
            // Combine all components
            int final_value = base_value + wave_x + wave_y + noise;
            row[x] = static_cast<unsigned char>(std::max(0, std::min(255, final_value)));
        }
    }
    
    return image;
}

void print_usage(const char* program_name) {
    std::cout << "CLAHE Benchmarking Harness\n"
              << "Usage: " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  --iterations N          Number of CLAHE iterations (default: 100)\n"
              << "  --image-size WxH        Image dimensions (default: 4096x4096)\n"
              << "  --grid-size GxG         Grid dimensions (default: 16x16)\n"
              << "  --bins B                Histogram bins (default: 256)\n"
              << "  --clip-limit C          Contrast clip limit (default: 2.0)\n"
              << "  --input <file>          Use input image instead of synthetic\n"
              << "  --help                  Show this help\n\n"
              << "Examples:\n"
              << "  " << program_name << " --iterations 150 --image-size 2048x2048\n"
              << "  " << program_name << " --grid-size 8x8 --clip-limit 3.0\n"
              << "  " << program_name << " --input test_image.png --iterations 50\n";
}

bool parse_size_arg(const std::string& arg, int& width, int& height) {
    size_t x_pos = arg.find('x');
    if (x_pos == std::string::npos) return false;
    
    try {
        width = std::stoi(arg.substr(0, x_pos));
        height = std::stoi(arg.substr(x_pos + 1));
        return width > 0 && height > 0;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.iterations = std::stoi(argv[++i]);
            if (config.iterations <= 0) {
                std::cerr << "Error: iterations must be positive\n";
                return 1;
            }
        } else if (arg == "--image-size" && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], config.image_width, config.image_height)) {
                std::cerr << "Error: invalid image size format. Use WxH (e.g., 2048x2048)\n";
                return 1;
            }
        } else if (arg == "--grid-size" && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], config.grid_width, config.grid_height)) {
                std::cerr << "Error: invalid grid size format. Use GxG (e.g., 16x16)\n";
                return 1;
            }
        } else if (arg == "--bins" && i + 1 < argc) {
            config.bins = std::stoi(argv[++i]);
            if (config.bins < 2 || config.bins > 256) {
                std::cerr << "Error: bins must be between 2 and 256\n";
                return 1;
            }
        } else if (arg == "--clip-limit" && i + 1 < argc) {
            config.clip_limit = std::stod(argv[++i]);
            if (config.clip_limit < 0) {
                std::cerr << "Error: clip limit must be non-negative\n";
                return 1;
            }
        } else if (arg == "--input" && i + 1 < argc) {
            config.input_image = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    try {
        // Load or generate test image
        cv::Mat test_image;
        if (!config.input_image.empty()) {
            std::cout << "Loading input image: " << config.input_image << std::endl;
            test_image = cv::imread(config.input_image, cv::IMREAD_GRAYSCALE);
            if (test_image.empty()) {
                std::cerr << "Error: Could not load image " << config.input_image << std::endl;
                return 1;
            }
            std::cout << "Loaded image: " << test_image.cols << "x" << test_image.rows << std::endl;
        } else {
            std::cout << "Generating synthetic test image: " 
                      << config.image_width << "x" << config.image_height << std::endl;
            test_image = generate_synthetic_image(config.image_width, config.image_height);
        }
        
        // Setup CLAHE configuration
        clahe::CLAHEConfig clahe_config;
        clahe_config.grid_width = config.grid_width;
        clahe_config.grid_height = config.grid_height;
        clahe_config.bins = config.bins;
        clahe_config.clip_limit = config.clip_limit;
        clahe_config.collect_metrics = false; // Disable for pure performance testing
        
        clahe::ModernCLAHE processor(clahe_config);
        
        // Print benchmark configuration
        std::cout << "\n=== CLAHE Benchmark Configuration ===\n";
        std::cout << "Image size:       " << test_image.cols << "x" << test_image.rows << "\n";
        std::cout << "Grid size:        " << config.grid_width << "x" << config.grid_height << "\n";
        std::cout << "Histogram bins:   " << config.bins << "\n";
        std::cout << "Clip limit:       " << config.clip_limit << "\n";
        std::cout << "Iterations:       " << config.iterations << "\n";
        std::cout << "Expected runtime: ~" << std::fixed << std::setprecision(1) 
                  << (config.iterations * 0.2) << "-" << (config.iterations * 0.3) << " seconds\n\n";
        
        // Warm-up run
        std::cout << "Performing warm-up run..." << std::flush;
        cv::Mat warmup_result = processor.process(test_image);
        std::cout << " done\n\n";
        
        // Main benchmark loop
        std::cout << "Running " << config.iterations << " iterations...\n";
        std::cout << "Progress: [";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < config.iterations; ++i) {
            // Process CLAHE
            cv::Mat result = processor.process(test_image);
            
            // Progress indicator every 10% of iterations
            if (i % (config.iterations / 10) == 0) {
                std::cout << "=" << std::flush;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "]\n\n";
        
        // Calculate and display results
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double total_seconds = total_duration.count() / 1000.0;
        double avg_ms_per_iteration = total_duration.count() / static_cast<double>(config.iterations);
        
        std::cout << "=== Benchmark Results ===\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total elapsed time:     " << total_seconds << " seconds\n";
        std::cout << "Average per iteration:  " << avg_ms_per_iteration << " ms\n";
        std::cout << "Throughput:             " << std::setprecision(2) 
                  << (config.iterations / total_seconds) << " iterations/second\n";
        
        // Performance scaling estimates
        std::cout << "\nScaling estimates:\n";
        std::cout << "For 15 seconds:  ~" << static_cast<int>(15.0 / avg_ms_per_iteration * 1000) << " iterations\n";
        std::cout << "For 30 seconds:  ~" << static_cast<int>(30.0 / avg_ms_per_iteration * 1000) << " iterations\n";
        
        // Memory usage estimate
        size_t image_memory = test_image.total() * test_image.elemSize();
        size_t histogram_memory = config.grid_width * config.grid_height * config.bins * sizeof(unsigned long);
        std::cout << "\nMemory usage:\n";
        std::cout << "Image data:      " << (image_memory / 1024 / 1024) << " MB\n";
        std::cout << "Histogram data:  " << (histogram_memory / 1024) << " KB\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}