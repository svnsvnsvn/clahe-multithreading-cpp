#include "benchmark.cpp"
#include <iostream>

void print_usage() {
    std::cout << "CLAHE Benchmark Suite\n"
              << "Usage: clahe_benchmark [options]\n\n"
              << "Options:\n"
              << "  --iterations <n>        Number of iterations per test (default: 5)\n"
              << "  --no-opencv            Skip OpenCV comparison\n"
              << "  --quick                Run quick benchmark (fewer configurations)\n"
              << "  --output <filename>     CSV output filename (default: clahe_benchmark_results.csv)\n"
              << "  --help                 Show this help\n";
}

int main(int argc, char* argv[]) {
    clahe::benchmark::BenchmarkConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.iterations = std::stoi(argv[++i]);
        } else if (arg == "--no-opencv") {
            config.compare_opencv = false;
        } else if (arg == "--quick") {
            // Reduced configuration for quick testing
            config.image_sizes = {cv::Size(512, 512), cv::Size(1024, 1024)};
            config.grid_sizes = {{8, 8}};
            config.clip_limits = {2.0};
            config.bin_counts = {256};
            config.iterations = 3;
        } else if (arg == "--output" && i + 1 < argc) {
            config.csv_filename = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }
    
    try {
        clahe::benchmark::BenchmarkSuite suite(config);
        suite.run_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
