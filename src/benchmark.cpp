#include "clahe_modern.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <thread>
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

struct BenchmarkResults {
    double mean_ms;
    double std_dev_ms;
    double min_ms;
    double max_ms;
    
    // Component times
    double histogram_time_us;
    double clipping_time_us;
    double mapping_time_us;
    double interpolation_time_us;
    
    // CPU metrics
    int num_threads;
    double cpu_time_us;
};

class PerformanceBenchmark {
public:
    PerformanceBenchmark(int iterations = 20, int warmup_runs = 3) 
        : iterations_(iterations), warmup_runs_(warmup_runs) {}
    
    BenchmarkResults run_benchmark(const cv::Mat& image, bool use_threading, int num_threads = 1) {
        clahe::CLAHEConfig config;
        config.clip_limit = 2.0;
        config.grid_width = 8;
        config.grid_height = 8;
        config.collect_metrics = true;
        config.use_threading = use_threading;
        
        clahe::ModernCLAHE clahe(config);
        
        std::vector<double> execution_times;
        clahe::PerformanceMetrics final_metrics;
        
        // Warm-up runs to stabilize cache and CPU frequency
        std::cout << "  Warming up (" << warmup_runs_ << " runs)..." << std::flush;
        for (int i = 0; i < warmup_runs_; ++i) {
            cv::Mat result = clahe.process(image);
        }
        std::cout << " Done!" << std::endl;
        
        // Actual benchmark runs with progress indicator
        std::cout << "  Running " << iterations_ << " iterations: [" << std::flush;
        int progress_step = std::max(1, iterations_ / 20);
        
        for (int i = 0; i < iterations_; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat result = clahe.process(image);
            auto end = std::chrono::high_resolution_clock::now();
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            execution_times.push_back(time_ms);
            
            final_metrics = clahe.get_metrics();
            
            // Progress indicator
            if (i % progress_step == 0) {
                std::cout << "=" << std::flush;
            }
        }
        std::cout << "] Done!" << std::endl;
        
        // Calculate statistics
        BenchmarkResults results;
        results.mean_ms = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / execution_times.size();
        
        double sq_sum = 0.0;
        for (double time : execution_times) {
            sq_sum += (time - results.mean_ms) * (time - results.mean_ms);
        }
        results.std_dev_ms = std::sqrt(sq_sum / execution_times.size());
        
        results.min_ms = *std::min_element(execution_times.begin(), execution_times.end());
        results.max_ms = *std::max_element(execution_times.begin(), execution_times.end());
        
        // Component times (from last run)
        results.histogram_time_us = final_metrics.histogram_time.count();
        results.clipping_time_us = final_metrics.clipping_time.count();
        results.mapping_time_us = final_metrics.mapping_time.count();
        results.interpolation_time_us = final_metrics.interpolation_time.count();
        
        results.num_threads = num_threads;
        results.cpu_time_us = results.histogram_time_us + results.clipping_time_us + 
                              results.mapping_time_us + results.interpolation_time_us;
        
        return results;
    }
    
    void print_comparison_report(const std::string& image_path, 
                                const BenchmarkResults& single, 
                                const BenchmarkResults& multi) {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        
        // Extract base filename for output naming
        std::string base_filename = get_base_filename(image_path);
        
        // Create output directory
        create_directory("output");
        
        // Process images with both configurations
        clahe::CLAHEConfig config;
        config.clip_limit = 2.0;
        config.grid_width = 8;
        config.grid_height = 8;
        config.collect_metrics = false;
        
        // Single-threaded result
        config.use_threading = false;
        clahe::ModernCLAHE clahe_single(config);
        cv::Mat result_single = clahe_single.process(image);
        
        // Multi-threaded result
        config.use_threading = true;
        clahe::ModernCLAHE clahe_multi(config);
        cv::Mat result_multi = clahe_multi.process(image);
        
        // Save outputs with descriptive names
        std::string output_single = "output/" + base_filename + "_clahe_single_threaded.png";
        std::string output_multi = "output/" + base_filename + "_clahe_multi_threaded.png";
        std::string output_original = "output/" + base_filename + "_original.png";
        
        cv::imwrite(output_single, result_single);
        cv::imwrite(output_multi, result_multi);
        cv::imwrite(output_original, image);
        
        std::cout << "╔════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║       CLAHE PERFORMANCE ANALYSIS: SINGLE vs MULTI-THREADED        ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Test Configuration:" << std::endl;
        std::cout << "  Image: " << image_path << std::endl;
        std::cout << "  Size: " << image.cols << "x" << image.rows << " pixels" << std::endl;
        std::cout << "  Iterations: " << iterations_ << " (with " << warmup_runs_ << " warmup runs)" << std::endl;
        std::cout << "  Grid Size: 8x8" << std::endl;
        std::cout << "  Clip Limit: 2.0" << std::endl;
        std::cout << std::endl;
        
        // Overall performance comparison
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "OVERALL EXECUTION TIME" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        
        std::cout << "\nSingle-threaded (1 thread):" << std::endl;
        std::cout << "  Mean:     " << std::setw(8) << single.mean_ms << " ms" << std::endl;
        std::cout << "  Std Dev:  " << std::setw(8) << single.std_dev_ms << " ms" << std::endl;
        std::cout << "  Min:      " << std::setw(8) << single.min_ms << " ms" << std::endl;
        std::cout << "  Max:      " << std::setw(8) << single.max_ms << " ms" << std::endl;
        
        std::cout << "\nMulti-threaded (4 threads):" << std::endl;
        std::cout << "  Mean:     " << std::setw(8) << multi.mean_ms << " ms" << std::endl;
        std::cout << "  Std Dev:  " << std::setw(8) << multi.std_dev_ms << " ms" << std::endl;
        std::cout << "  Min:      " << std::setw(8) << multi.min_ms << " ms" << std::endl;
        std::cout << "  Max:      " << std::setw(8) << multi.max_ms << " ms" << std::endl;
        
        double speedup = single.mean_ms / multi.mean_ms;
        double efficiency = (speedup / 4.0) * 100.0;
        
        std::cout << "\n" << std::string(67, '-') << std::endl;
        std::cout << "Overall Speedup:     " << std::setw(8) << speedup << "x" << std::endl;
        std::cout << "Parallel Efficiency: " << std::setw(8) << efficiency << "%" << std::endl;
        std::cout << "Time Reduction:      " << std::setw(8) << ((1.0 - multi.mean_ms/single.mean_ms) * 100.0) << "%" << std::endl;
        std::cout << std::endl;
        
        // Component-wise analysis
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "COMPONENT-WISE PERFORMANCE BREAKDOWN" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << std::setprecision(1);
        
        std::cout << "\n┌─────────────────────────┬──────────┬──────────┬──────────┬────────┐" << std::endl;
        std::cout << "│ Component               │  Single  │  Multi   │ Speedup  │ % Time │" << std::endl;
        std::cout << "│                         │   (μs)   │   (μs)   │          │ Saved  │" << std::endl;
        std::cout << "├─────────────────────────┼──────────┼──────────┼──────────┼────────┤" << std::endl;
        
        auto print_row = [](const std::string& name, double single_val, double multi_val) {
            double speedup = single_val / multi_val;
            double saved = ((single_val - multi_val) / single_val) * 100.0;
            std::cout << "│ " << std::left << std::setw(23) << name << " │ "
                      << std::right << std::setw(8) << std::fixed << std::setprecision(1) << single_val << " │ "
                      << std::setw(8) << multi_val << " │ "
                      << std::setw(8) << std::setprecision(2) << speedup << " │ "
                      << std::setw(6) << std::setprecision(1) << saved << " │" << std::endl;
        };
        
        print_row("Histogram Computation", single.histogram_time_us, multi.histogram_time_us);
        print_row("Histogram Clipping", single.clipping_time_us, multi.clipping_time_us);
        print_row("Mapping Computation", single.mapping_time_us, multi.mapping_time_us);
        print_row("Image Interpolation", single.interpolation_time_us, multi.interpolation_time_us);
        
        std::cout << "└─────────────────────────┴──────────┴──────────┴──────────┴────────┘" << std::endl;
        std::cout << std::endl;
        
        // Threading efficiency analysis
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "THREADING EFFICIENCY ANALYSIS" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        
        double hist_speedup = single.histogram_time_us / multi.histogram_time_us;
        double clip_speedup = single.clipping_time_us / multi.clipping_time_us;
        double interp_speedup = single.interpolation_time_us / multi.interpolation_time_us;
        
        std::cout << std::setprecision(2);
        std::cout << "\nHistogram Computation:" << std::endl;
        std::cout << "  Threads: 4" << std::endl;
        std::cout << "  Speedup: " << hist_speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << (hist_speedup / 4.0 * 100.0) << "%" << std::endl;
        std::cout << "  Analysis: ";
        if (hist_speedup > 3.0) std::cout << "Excellent parallelization";
        else if (hist_speedup > 2.0) std::cout << "Good parallelization";
        else if (hist_speedup > 1.5) std::cout << "Moderate parallelization";
        else std::cout << "Limited by synchronization overhead";
        std::cout << std::endl;
        
        std::cout << "\nHistogram Clipping:" << std::endl;
        std::cout << "  Threads: 4" << std::endl;
        std::cout << "  Speedup: " << clip_speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << (clip_speedup / 4.0 * 100.0) << "%" << std::endl;
        std::cout << "  Analysis: ";
        if (clip_speedup < 1.0) std::cout << "Threading overhead exceeds benefit (too small workload)";
        else if (clip_speedup > 2.0) std::cout << "Good parallelization";
        else std::cout << "Moderate benefit, small workload per region";
        std::cout << std::endl;
        
        std::cout << "\nImage Interpolation:" << std::endl;
        std::cout << "  Threads: 4" << std::endl;
        std::cout << "  Speedup: " << interp_speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << (interp_speedup / 4.0 * 100.0) << "%" << std::endl;
        std::cout << "  Analysis: ";
        if (interp_speedup > 3.0) std::cout << "Excellent parallelization (largest workload)";
        else if (interp_speedup > 2.0) std::cout << "Good parallelization";
        else std::cout << "Moderate parallelization";
        std::cout << std::endl;
        
        std::cout << "\n" << std::string(67, '-') << std::endl;
        std::cout << "\nKey Findings:" << std::endl;
        std::cout << "• Best parallelization in: ";
        if (interp_speedup > hist_speedup && interp_speedup > clip_speedup) {
            std::cout << "Image Interpolation (largest per-pixel workload)" << std::endl;
        } else if (hist_speedup > interp_speedup && hist_speedup > clip_speedup) {
            std::cout << "Histogram Computation (good work distribution)" << std::endl;
        } else {
            std::cout << "Multiple components show good scaling" << std::endl;
        }
        
        std::cout << "• Threading overhead most visible in: Histogram Clipping (small workload)" << std::endl;
        std::cout << "• Overall parallel efficiency: " << efficiency << "% (ideal = 100%)" << std::endl;
        
        std::cout << std::endl;
        
        // CPU utilization estimate
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "CPU UTILIZATION ESTIMATE" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << std::endl;
        
        double single_cpu_usage = 100.0; // 1 thread = 100% of 1 core
        double multi_theoretical_cpu = 400.0; // 4 threads = 400% theoretical
        double multi_actual_cpu = speedup * 100.0; // actual CPU usage based on speedup
        
        std::cout << "Single-threaded:" << std::endl;
        std::cout << "  CPU cores used: 1" << std::endl;
        std::cout << "  CPU utilization: ~" << single_cpu_usage << "% (1 core)" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Multi-threaded:" << std::endl;
        std::cout << "  CPU cores used: 4" << std::endl;
        std::cout << "  Theoretical CPU: " << multi_theoretical_cpu << "% (4 cores)" << std::endl;
        std::cout << "  Effective CPU: ~" << multi_actual_cpu << "% (" << speedup << " cores)" << std::endl;
        std::cout << "  Efficiency: " << (multi_actual_cpu / multi_theoretical_cpu * 100.0) << "%" << std::endl;
        std::cout << std::endl;
        
        // Summary and output file info
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "OUTPUT FILES" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << std::endl;
        std::cout << "Results saved to output directory:" << std::endl;
        std::cout << "  Original:         " << output_original << std::endl;
        std::cout << "  Single-threaded:  " << output_single << std::endl;
        std::cout << "  Multi-threaded:   " << output_multi << std::endl;
        std::cout << std::endl;
    }
    
private:
    int iterations_;
    int warmup_runs_;
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return -1;
    }
    
    std::string input_path = argv[1];
    
    cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << input_path << std::endl;
        return -1;
    }
    
    std::cout << "\nInitializing benchmark suite..." << std::endl;
    std::cout << "This will take approximately 40-60 seconds.\n" << std::endl;
    
    PerformanceBenchmark benchmark(20, 3);  // 20 iterations, 3 warmup runs
    
    std::cout << "=== Phase 1: Single-threaded Baseline ===" << std::endl;
    BenchmarkResults single_results = benchmark.run_benchmark(image, false, 1);
    
    std::cout << "\n=== Phase 2: Multi-threaded Implementation ===" << std::endl;
    BenchmarkResults multi_results = benchmark.run_benchmark(image, true, 4);
    
    benchmark.print_comparison_report(input_path, single_results, multi_results);
    
    return 0;
}
