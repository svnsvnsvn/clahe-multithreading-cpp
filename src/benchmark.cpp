#include "clahe_modern.h"
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace clahe {
namespace benchmark {

struct BenchmarkConfig {
    std::vector<cv::Size> image_sizes = {
        cv::Size(256, 256),   // Small
        cv::Size(512, 512),   // Medium  
        cv::Size(1024, 1024), // Large
        cv::Size(2048, 2048), // Very Large
        cv::Size(4096, 4096)  // Huge
    };
    
    std::vector<std::pair<int, int>> grid_sizes = {
        {4, 4},    // Coarse
        {8, 8},    // Standard
        {16, 16}   // Fine
    };
    
    std::vector<double> clip_limits = {1.0, 2.0, 4.0, 8.0};
    std::vector<unsigned int> bin_counts = {64, 128, 256};
    
    int iterations = 5; // Number of iterations per test
    bool compare_opencv = true;
    bool save_csv = true;
    std::string csv_filename = "clahe_benchmark_results.csv";
};

struct BenchmarkResult {
    cv::Size image_size;
    std::pair<int, int> grid_size;
    double clip_limit;
    unsigned int bins;
    
    // Timing results (microseconds)
    double modern_time_avg;
    double modern_time_std;
    double opencv_time_avg;
    double opencv_time_std;
    double speedup;
    
    // Accuracy metrics
    double mean_absolute_error;
    double peak_snr;
    
    // Memory usage (bytes)
    size_t memory_usage;
    
    // Detailed timing breakdown
    clahe::PerformanceMetrics detailed_metrics;
};

class BenchmarkSuite {
private:
    BenchmarkConfig config_;
    std::vector<BenchmarkResult> results_;
    
    cv::Mat generate_test_image(cv::Size size) {
        cv::Mat image(size, CV_8UC1);
        cv::randu(image, 0, 255);
        
        // Add some structure to make it more realistic
        for (int y = 0; y < size.height; y += 50) {
            for (int x = 0; x < size.width; x += 50) {
                cv::rectangle(image, 
                    cv::Point(x, y), 
                    cv::Point(std::min(x + 25, size.width-1), std::min(y + 25, size.height-1)),
                    cv::Scalar(cv::theRNG().uniform(100, 200)), 
                    -1);
            }
        }
        
        // Add some gaussian noise
        cv::Mat noise(size, CV_8UC1);
        cv::randn(noise, 0, 10);
        image += noise;
        
        return image;
    }
    
    std::pair<double, double> calculate_stats(const std::vector<double>& times) {
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        double std_dev = std::sqrt(sq_sum / times.size() - mean * mean);
        return {mean, std_dev};
    }
    
public:
    explicit BenchmarkSuite(const BenchmarkConfig& config = BenchmarkConfig{}) 
        : config_(config) {}
    
    void run_benchmark() {
        std::cout << "=== CLAHE Benchmarking Suite ===\n\n";
        std::cout << "Configuration:\n"
                  << "  Image sizes: " << config_.image_sizes.size() << " different sizes\n"
                  << "  Grid configurations: " << config_.grid_sizes.size() << " different grids\n"
                  << "  Clip limits: " << config_.clip_limits.size() << " different limits\n"
                  << "  Bin counts: " << config_.bin_counts.size() << " different bin counts\n"
                  << "  Iterations per test: " << config_.iterations << "\n"
                  << "  Compare with OpenCV: " << (config_.compare_opencv ? "Yes" : "No") << "\n\n";
        
        int total_tests = config_.image_sizes.size() * config_.grid_sizes.size() * 
                         config_.clip_limits.size() * config_.bin_counts.size();
        
        std::cout << "Total tests to run: " << total_tests << "\n\n";
        
        int current_test = 0;
        
        for (const auto& size : config_.image_sizes) {
            std::cout << "Testing image size " << size.width << "x" << size.height << "...\n";
            cv::Mat test_image = generate_test_image(size);
            
            for (const auto& grid : config_.grid_sizes) {
                for (double clip_limit : config_.clip_limits) {
                    for (unsigned int bins : config_.bin_counts) {
                        run_single_benchmark(test_image, grid, clip_limit, bins, current_test++, total_tests);
                    }
                }
            }
        }
        
        std::cout << "\nBenchmarking completed!\n";
        print_summary();
        
        if (config_.save_csv) {
            save_results_csv();
        }
    }
    
private:
    void run_single_benchmark(const cv::Mat& test_image, 
                             const std::pair<int, int>& grid,
                             double clip_limit,
                             unsigned int bins,
                             int current_test,
                             int total_tests) {
        
        std::cout << "\r[" << std::setw(3) << (current_test * 100 / total_tests) << "%] "
                  << "Grid: " << grid.first << "x" << grid.second 
                  << ", Clip: " << clip_limit
                  << ", Bins: " << bins
                  << std::flush;
        
        // Setup configuration
        CLAHEConfig config;
        config.grid_width = grid.first;
        config.grid_height = grid.second;
        config.clip_limit = clip_limit;
        config.bins = bins;
        config.collect_metrics = true;
        
        BenchmarkResult result;
        result.image_size = test_image.size();
        result.grid_size = grid;
        result.clip_limit = clip_limit;
        result.bins = bins;
        
        // Benchmark our modern implementation
        std::vector<double> modern_times;
        clahe::PerformanceMetrics accumulated_metrics;
        
        for (int iter = 0; iter < config_.iterations; ++iter) {
            ModernCLAHE clahe(config);
            
            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat modern_result = clahe.process(test_image);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            modern_times.push_back(duration.count());
            
            if (iter == 0) {
                result.detailed_metrics = clahe.get_metrics();
                result.memory_usage = clahe.get_metrics().memory_allocated;
                
                // Compare with OpenCV if requested
                if (config_.compare_opencv) {
                    auto opencv_clahe = cv::createCLAHE(clip_limit, cv::Size(grid.first, grid.second));
                    cv::Mat opencv_result;
                    opencv_clahe->apply(test_image, opencv_result);
                    
                    // Calculate accuracy metrics
                    cv::Mat diff;
                    cv::absdiff(modern_result, opencv_result, diff);
                    result.mean_absolute_error = cv::mean(diff)[0];
                    
                    cv::Mat diff_sq;
                    diff.convertTo(diff_sq, CV_32F);
                    diff_sq = diff_sq.mul(diff_sq);
                    double mse = cv::mean(diff_sq)[0];
                    result.peak_snr = (mse == 0) ? 100.0 : 20.0 * std::log10(255.0 / std::sqrt(mse));
                }
            }
        }
        
        auto [modern_avg, modern_std] = calculate_stats(modern_times);
        result.modern_time_avg = modern_avg;
        result.modern_time_std = modern_std;
        
        // Benchmark OpenCV implementation if requested
        if (config_.compare_opencv) {
            std::vector<double> opencv_times;
            auto opencv_clahe = cv::createCLAHE(clip_limit, cv::Size(grid.first, grid.second));
            
            for (int iter = 0; iter < config_.iterations; ++iter) {
                cv::Mat opencv_result;
                
                auto start = std::chrono::high_resolution_clock::now();
                opencv_clahe->apply(test_image, opencv_result);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                opencv_times.push_back(duration.count());
            }
            
            auto [opencv_avg, opencv_std] = calculate_stats(opencv_times);
            result.opencv_time_avg = opencv_avg;
            result.opencv_time_std = opencv_std;
            result.speedup = opencv_avg / modern_avg;
        } else {
            result.opencv_time_avg = 0;
            result.opencv_time_std = 0;
            result.speedup = 0;
            result.mean_absolute_error = 0;
            result.peak_snr = 0;
        }
        
        results_.push_back(result);
    }
    
    void print_summary() {
        std::cout << "\n\n=== Benchmark Summary ===\n\n";
        
        // Find best and worst performing configurations
        auto best_speedup = std::max_element(results_.begin(), results_.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.speedup < b.speedup;
            });
        
        auto worst_speedup = std::min_element(results_.begin(), results_.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.speedup < b.speedup;
            });
            
        if (config_.compare_opencv && best_speedup != results_.end()) {
            std::cout << "Best speedup: " << std::fixed << std::setprecision(2) 
                      << best_speedup->speedup << "x\n"
                      << "  Configuration: " << best_speedup->image_size.width << "x" << best_speedup->image_size.height
                      << ", Grid: " << best_speedup->grid_size.first << "x" << best_speedup->grid_size.second
                      << ", Clip: " << best_speedup->clip_limit << "\n\n";
                      
            std::cout << "Worst speedup: " << std::fixed << std::setprecision(2) 
                      << worst_speedup->speedup << "x\n"
                      << "  Configuration: " << worst_speedup->image_size.width << "x" << worst_speedup->image_size.height
                      << ", Grid: " << worst_speedup->grid_size.first << "x" << worst_speedup->grid_size.second
                      << ", Clip: " << worst_speedup->clip_limit << "\n\n";
        }
        
        // Calculate average speedup for single-threaded implementation
        if (config_.compare_opencv && !results_.empty()) {
            double total_speedup = 0;
            for (const auto& result : results_) {
                total_speedup += result.speedup;
            }
            double avg_speedup = total_speedup / results_.size();
            
            std::cout << "Single-threaded Performance vs OpenCV:\n"
                      << "  Average speedup: " << std::fixed << std::setprecision(2) 
                      << avg_speedup << "x\n\n";
        }
    }
    
    void save_results_csv() {
        std::ofstream csv_file(config_.csv_filename);
        
        // Write header
        csv_file << "Image_Width,Image_Height,Grid_Width,Grid_Height,Clip_Limit,Bins,"
                << "Modern_Time_Avg_us,Modern_Time_Std_us,OpenCV_Time_Avg_us,OpenCV_Time_Std_us,Speedup,"
                << "Mean_Absolute_Error,Peak_SNR_dB,Memory_Usage_bytes,"
                << "Histogram_Time_us,Clipping_Time_us,Mapping_Time_us,Interpolation_Time_us\n";
        
        // Write data
        for (const auto& result : results_) {
            csv_file << result.image_size.width << "," << result.image_size.height << ","
                    << result.grid_size.first << "," << result.grid_size.second << ","
                    << result.clip_limit << "," << result.bins << ","
                    << std::fixed << std::setprecision(2) 
                    << result.modern_time_avg << "," << result.modern_time_std << ","
                    << result.opencv_time_avg << "," << result.opencv_time_std << ","
                    << result.speedup << "," << result.mean_absolute_error << ","
                    << result.peak_snr << "," << result.memory_usage << ","
                    << result.detailed_metrics.histogram_time.count() << ","
                    << result.detailed_metrics.clipping_time.count() << ","
                    << result.detailed_metrics.mapping_time.count() << ","
                    << result.detailed_metrics.interpolation_time.count() << "\n";
        }
        
        csv_file.close();
        std::cout << "Results saved to: " << config_.csv_filename << "\n";
    }
};

} // namespace benchmark
} // namespace clahe
