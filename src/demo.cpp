#include "clahe_modern.h"
#include <iostream>
#include <string>
#include <sys/stat.h>

// Helper function to create directory if it doesn't exist
void create_directory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        #ifdef _WIN32
            _mkdir(path.c_str());
        #else
            mkdir(path.c_str(), 0755);
        #endif
    }
}

// Helper function to extract filename without extension
std::string get_base_filename(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of('.');
    
    std::string filename = (last_slash != std::string::npos) ? 
                          path.substr(last_slash + 1) : path;
    
    if (last_dot != std::string::npos && last_dot > last_slash) {
        filename = filename.substr(0, last_dot - (last_slash != std::string::npos ? last_slash + 1 : 0));
    }
    
    return filename;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_image> [output_image] [options]\n"
              << "\nOptions:\n"
              << "  --clip-limit <value>    Contrast limit (default: 2.0)\n"
              << "  --grid <width>x<height> Grid dimensions (default: 8x8)\n"
              << "  --bins <count>          Histogram bins (default: 256)\n"
              << "  --compare               Compare with OpenCV implementation\n"
              << "  --metrics               Show performance metrics\n"
              << "  --help                  Show this help\n"
              << "\nExample:\n"
              << "  " << program_name << " input.jpg output.jpg --clip-limit 3.0 --grid 16x16 --metrics\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    std::string input_path = argv[1];
    
    // Create output directory
    create_directory("output");
    
    // Generate default output path if not specified
    std::string base_filename = get_base_filename(input_path);
    std::string default_output = "output/" + base_filename + "_clahe_output.png";
    std::string output_path = (argc > 2 && argv[2][0] != '-') ? argv[2] : default_output;
    
    clahe::CLAHEConfig config;
    config.collect_metrics = false;
    bool show_comparison = false;
    
    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--clip-limit" && i + 1 < argc) {
            config.clip_limit = std::stod(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            std::string grid_str = argv[++i];
            size_t x_pos = grid_str.find('x');
            if (x_pos != std::string::npos) {
                config.grid_width = std::stoi(grid_str.substr(0, x_pos));
                config.grid_height = std::stoi(grid_str.substr(x_pos + 1));
            }
        } else if (arg == "--bins" && i + 1 < argc) {
            config.bins = std::stoi(argv[++i]);
        } else if (arg == "--compare") {
            show_comparison = true;
        } else if (arg == "--metrics") {
            config.collect_metrics = true;
        }
    }
    
    // Validate configuration
    if (!config.is_valid()) {
        std::cerr << "Error: " << config.validation_error() << std::endl;
        return 1;
    }
    
    try {
        // Load input image
        std::cout << "Loading image: " << input_path << std::endl;
        cv::Mat src = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
        
        if (src.empty()) {
            std::cerr << "Error: Could not load image " << input_path << std::endl;
            return 1;
        }
        
        std::cout << "Image size: " << src.cols << "x" << src.rows << std::endl;
        std::cout << "CLAHE Configuration:\n"
                  << "  Grid: " << config.grid_width << "x" << config.grid_height << "\n"
                  << "  Clip limit: " << config.clip_limit << "\n"
                  << "  Bins: " << config.bins << "\n"
                  << "  Full range: " << (config.use_full_range ? "Yes" : "No") << std::endl;
        
        if (show_comparison) {
            // Compare implementations
            std::cout << "\n=== Running Comparison ===\n";
            auto comparison = clahe::comparison::compare_implementations(src, config);
            comparison.print_summary();
            
            // Save both results to output directory with descriptive names
            std::string modern_path = "output/" + base_filename + "_modern.png";
            std::string opencv_path = "output/" + base_filename + "_opencv.png";
            
            cv::imwrite(modern_path, comparison.modern_result);
            cv::imwrite(opencv_path, comparison.opencv_result);
            
            std::cout << "\nResults saved:\n"
                      << "  Modern implementation: " << modern_path << "\n"
                      << "  OpenCV implementation: " << opencv_path << std::endl;
                      
        } else {
            // Process with our implementation only
            std::cout << "\n=== Processing with Modern CLAHE ===\n";
            
            clahe::ModernCLAHE clahe_processor(config);
            cv::Mat result = clahe_processor.process(src);
            
            if (config.collect_metrics) {
                std::cout << "\n";
                clahe_processor.get_metrics().print();
            }
            
            // Save result
            cv::imwrite(output_path, result);
            std::cout << "\nResult saved to: " << output_path << std::endl;
        }
        
        std::cout << "\nProcessing completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
