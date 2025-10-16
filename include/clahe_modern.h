#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <chrono>

/**
 * Modern CLAHE Implementation for OpenCV 4.x
 * 
 * Key improvements over legacy version:
 * 1. Uses cv::Mat instead of IplImage
 * 2. RAII for automatic memory management
 * 3. Template-based for different pixel types
 * 4. Optimized memory layout and access patterns
 * 5. Built-in performance profiling
 */

namespace clahe {

// Performance profiling structure
struct PerformanceMetrics {
    std::chrono::microseconds histogram_time{0};
    std::chrono::microseconds clipping_time{0};
    std::chrono::microseconds mapping_time{0};
    std::chrono::microseconds interpolation_time{0};
    std::chrono::microseconds total_time{0};
    size_t memory_allocated{0};
    
    void print() const;
    void reset();
};

// Configuration for CLAHE algorithm
struct CLAHEConfig {
    unsigned int grid_width = 8;     // Number of contextual regions in X direction (2-16)
    unsigned int grid_height = 8;    // Number of contextual regions in Y direction (2-16) 
    unsigned int bins = 256;         // Number of histogram bins (2-256)
    double clip_limit = 2.0;         // Contrast limit (>= 0.0, negative for AHE)
    bool use_full_range = true;      // Use full pixel range vs input range
    bool collect_metrics = false;    // Collect performance metrics
    
    // Validation
    bool is_valid() const;
    std::string validation_error() const;
};

/**
 * Modern CLAHE class with optimized implementation
 */
class ModernCLAHE {
public:
    explicit ModernCLAHE(const CLAHEConfig& config = CLAHEConfig{});
    
    // Main processing function - supports 8-bit grayscale images
    cv::Mat process(const cv::Mat& src);
    
    // In-place processing for memory efficiency
    void process_inplace(cv::Mat& image);
    
    // Get performance metrics from last operation
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    
    // Update configuration
    void set_config(const CLAHEConfig& config);
    const CLAHEConfig& get_config() const { return config_; }

private:
    CLAHEConfig config_;
    mutable PerformanceMetrics metrics_;
    
    // Pre-allocated working memory to avoid repeated allocations
    mutable std::vector<std::vector<unsigned long>> histograms_;
    mutable std::vector<unsigned char> lut_;
    
    // Core algorithm functions - optimized versions of legacy code
    void compute_histograms(const cv::Mat& image);
    void clip_histograms();
    void compute_mappings(unsigned char min_val, unsigned char max_val);
    void interpolate_image(cv::Mat& image) const;
    
    // Utility functions
    void prepare_working_memory(const cv::Mat& image);
    void make_lut(unsigned char min_val, unsigned char max_val);
    std::pair<unsigned char, unsigned char> get_min_max(const cv::Mat& image) const;
    
    // Optimized interpolation with SIMD-friendly layout
    void interpolate_region(cv::Mat& image, int start_x, int start_y, 
                          int width, int height, int region_x, int region_y) const;
};

/**
 * Convenience functions for backward compatibility and ease of use
 */

// Simple interface similar to cv::createCLAHE()
cv::Mat apply_clahe(const cv::Mat& src, double clip_limit = 2.0, 
                   int grid_width = 8, int grid_height = 8);

// Legacy-compatible function signature
void adaptive_equalize(const cv::Mat& src, cv::Mat& dst, 
                      unsigned int x_divs, unsigned int y_divs, 
                      unsigned int bins, bool use_full_range = true);

void contrast_limited_adaptive_equalize(const cv::Mat& src, cv::Mat& dst,
                                      unsigned int x_divs, unsigned int y_divs,
                                      unsigned int bins, double clip_limit,
                                      bool use_full_range = true);

/**
 * Comparison utilities for benchmarking against OpenCV's built-in CLAHE
 */
namespace comparison {
    
struct ComparisonResult {
    cv::Mat modern_result;
    cv::Mat opencv_result;
    PerformanceMetrics modern_metrics;
    std::chrono::microseconds opencv_time;
    double mean_absolute_error;
    double peak_signal_noise_ratio;
    
    void print_summary() const;
};

ComparisonResult compare_implementations(const cv::Mat& src, 
                                       const CLAHEConfig& config = CLAHEConfig{});

} // namespace comparison

} // namespace clahe
