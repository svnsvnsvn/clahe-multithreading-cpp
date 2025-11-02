#include "clahe_modern.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>

namespace clahe {

// ============================================================================
// PerformanceMetrics Implementation
// ============================================================================

void PerformanceMetrics::print() const {
    std::cout << "CLAHE Performance Metrics:\n"
              << "  Histogram computation: " << histogram_time.count() << " μs\n"
              << "  Histogram clipping:    " << clipping_time.count() << " μs\n"
              << "  Mapping computation:   " << mapping_time.count() << " μs\n"
              << "  Image interpolation:   " << interpolation_time.count() << " μs\n"
              << "  Total time:           " << total_time.count() << " μs\n"
              << "  Memory allocated:     " << memory_allocated << " bytes\n";
}

void PerformanceMetrics::reset() {
    histogram_time = clipping_time = mapping_time = interpolation_time = total_time = 
        std::chrono::microseconds{0};
    memory_allocated = 0;
}

// ============================================================================
// CLAHEConfig Implementation  
// ============================================================================

bool CLAHEConfig::is_valid() const {
    return grid_width >= 2 && grid_width <= 16 &&
           grid_height >= 2 && grid_height <= 16 &&
           bins >= 2 && bins <= 256;
}

std::string CLAHEConfig::validation_error() const {
    if (grid_width < 2 || grid_width > 16) 
        return "Grid width must be between 2 and 16";
    if (grid_height < 2 || grid_height > 16)
        return "Grid height must be between 2 and 16"; 
    if (bins < 2 || bins > 256)
        return "Bins must be between 2 and 256";
    return "";
}

// ============================================================================
// ModernCLAHE Implementation
// ============================================================================

ModernCLAHE::ModernCLAHE(const CLAHEConfig& config) : config_(config) {
    if (!config_.is_valid()) {
        throw std::invalid_argument("Invalid CLAHE configuration: " + config_.validation_error());
    }
}

cv::Mat ModernCLAHE::process(const cv::Mat& src) {
    // Input validation
    if (src.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("Only 8-bit single-channel images are supported");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (config_.collect_metrics) {
        metrics_.reset();
    }
    
    // Create output image
    cv::Mat dst = src.clone();
    
    // Process in-place for efficiency
    process_inplace(dst);
    
    if (config_.collect_metrics) {
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics_.total_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
    }
    
    return dst;
}

void ModernCLAHE::process_inplace(cv::Mat& image) {
    // Ensure image dimensions are compatible with grid
    int orig_width = image.cols;
    int orig_height = image.rows;
    
    // Calculate required padding to make dimensions divisible by grid size
    int pad_x = (config_.grid_width - (orig_width % config_.grid_width)) % config_.grid_width;
    int pad_y = (config_.grid_height - (orig_height % config_.grid_height)) % config_.grid_height;
    
    cv::Mat working_image;
    bool needs_resize = (pad_x > 0 || pad_y > 0);
    
    if (needs_resize) {
        // Pad image to make it compatible with grid
        cv::copyMakeBorder(image, working_image, 0, pad_y, 0, pad_x, 
                          cv::BORDER_REFLECT_101);
    } else {
        working_image = image;
    }
    
    // Prepare working memory
    prepare_working_memory(working_image);
    
    // Get min/max values for output range
    auto [min_val, max_val] = get_min_max(working_image);
    if (!config_.use_full_range) {
        // Use input range instead of full 0-255 range
        double min_d, max_d;
        cv::minMaxLoc(working_image, &min_d, &max_d);
        min_val = static_cast<unsigned char>(min_d);
        max_val = static_cast<unsigned char>(max_d);
    }
    
    // Core CLAHE algorithm steps
    compute_histograms(working_image, min_val, max_val);
    clip_histograms();
    compute_mappings(min_val, max_val);
    interpolate_image(working_image);
    
    // Copy result back if we had to resize
    if (needs_resize) {
        working_image(cv::Rect(0, 0, orig_width, orig_height)).copyTo(image);
    }
}

void ModernCLAHE::prepare_working_memory(const cv::Mat& image) {
    // Pre-allocate histograms for all contextual regions
    size_t num_regions = config_.grid_width * config_.grid_height;
    histograms_.resize(num_regions);
    
    for (auto& hist : histograms_) {
        hist.assign(config_.bins, 0);
    }
    
    // Pre-allocate lookup table
    lut_.resize(256);
    
    if (config_.collect_metrics) {
        metrics_.memory_allocated = 
            num_regions * config_.bins * sizeof(unsigned long) + 256;
    }
}

std::pair<unsigned char, unsigned char> ModernCLAHE::get_min_max(const cv::Mat& image) const {
    if (config_.use_full_range) {
        return {0, 255};
    } else {
        double min_val, max_val;
        cv::minMaxLoc(image, &min_val, &max_val);
        return {static_cast<unsigned char>(min_val), static_cast<unsigned char>(max_val)};
    }
}

void ModernCLAHE::make_lut(unsigned char min_val, unsigned char max_val) {
    // Create lookup table to map input range to histogram bins
    // This is a key optimization from the original algorithm
    const int range = max_val - min_val + 1;
    const double bin_size = static_cast<double>(range) / config_.bins;
    
    for (int i = 0; i < 256; ++i) {
        if (i < min_val) {
            lut_[i] = 0;
        } else if (i > max_val) {
            lut_[i] = config_.bins - 1;
        } else {
            int bin = static_cast<int>((i - min_val) / bin_size);
            lut_[i] = std::min(bin, static_cast<int>(config_.bins - 1));
        }
    }
}

void ModernCLAHE::compute_histograms(const cv::Mat& image, unsigned char min_val, unsigned char max_val) {
    // Begin timing for this section
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Creates LUT, it helps determine pixel shade values faster
    make_lut(min_val, max_val);
    
    // Store image dimensions for use in clip_histograms
    image_width_ = image.cols;
    image_height_ = image.rows;
    
    // Calculating dimensions of tiles
    const int region_width = image.cols / config_.grid_width;
    const int region_height = image.rows / config_.grid_height;
    
    // Number of desired threads
    const int THREAD_COUNT = 4;
    // Determine the total amount of tiles needed
    const size_t tile_total = config_.grid_width * config_.grid_height;
    
    // FIXED: Handle remainder properly for even work distribution
    const size_t tiles_per_thread = tile_total / THREAD_COUNT;
    const size_t remainder = tile_total % THREAD_COUNT;
    
    // Threads individually computing tiles
    auto histogram_For_Range = [this, &image, region_width, region_height](size_t starting_Tile, size_t end_Tile) {
        for (size_t t = starting_Tile; t < end_Tile; ++t) {
            unsigned int gx = t % config_.grid_width;
            unsigned int gy = t / config_.grid_width;
            size_t hist_idx = gy * config_.grid_width + gx;
            
            // Clear histogram
            std::fill(histograms_[hist_idx].begin(), histograms_[hist_idx].end(), 0);
            
            // Compute histogram for this region
            int start_x = gx * region_width;
            int start_y = gy * region_height;
            
            for (int y = start_y; y < start_y + region_height; ++y) {
                // This line below is how the function accesses the actual pixel data from the image
                const unsigned char* row = image.ptr<unsigned char>(y);
                for (int x = start_x; x < start_x + region_width; ++x) {
                    unsigned char bin = lut_[row[x]];
                    ++histograms_[hist_idx][bin];
                }
            }
        }
    };
    
    // FIXED: Create threads with balanced work distribution (handles remainder)
    std::vector<std::thread> threads;
    size_t current_start = 0;
    
    for (int t = 0; t < THREAD_COUNT; ++t) {
        size_t work_size = tiles_per_thread + (t < static_cast<int>(remainder) ? 1 : 0);
        size_t end_tile = current_start + work_size;
        
        threads.emplace_back(histogram_For_Range, current_start, end_tile);
        current_start = end_tile;
    }
    
    // Synchronization Barrier (Waiting for all threads to finish)
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Recording timing
    if (config_.collect_metrics) {
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics_.histogram_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
    }
}

void ModernCLAHE::clip_histograms() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (config_.clip_limit <= 0) {
        // No clipping - standard AHE
        if (config_.collect_metrics) {
            auto end_time = std::chrono::high_resolution_clock::now();
            metrics_.clipping_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);
        }
        return;
    }
    
    // Calculate clip limit based on actual image region size
    const int region_width = image_width_ / config_.grid_width;
    const int region_height = image_height_ / config_.grid_height; 
    const unsigned long pixels_per_region = region_width * region_height;
    const unsigned long clip_limit = static_cast<unsigned long>(
        config_.clip_limit * pixels_per_region / config_.bins);
    const unsigned long actual_clip_limit = std::max(1UL, clip_limit);
    
    if (config_.use_threading) {
        // Multi-threaded implementation
        const size_t num_histograms = histograms_.size();
        const size_t NUM_THREADS = 4;
        
        const size_t histograms_per_thread = num_histograms / NUM_THREADS;
        const size_t remainder = num_histograms % NUM_THREADS;
        
        auto clip_range = [this, actual_clip_limit](size_t start_idx, size_t end_idx) {
            for (size_t i = start_idx; i < end_idx && i < histograms_.size(); ++i) {
                auto& histogram = histograms_[i];
                
                unsigned long excess = 0;
                for (unsigned long& bin : histogram) {
                    if (bin > actual_clip_limit) {
                        excess += bin - actual_clip_limit;
                        bin = actual_clip_limit;
                    }
                }
                
                if (excess > 0) {
                    const unsigned long redistribution_per_bin = excess / config_.bins;
                    const unsigned long remainder_pixels = excess % config_.bins;
                    
                    for (size_t j = 0; j < histogram.size(); ++j) {
                        histogram[j] += redistribution_per_bin;
                    }
                    
                    unsigned long remaining_to_distribute = remainder_pixels;
                    for (size_t j = 0; j < histogram.size() && remaining_to_distribute > 0; ++j) {
                        if (histogram[j] < actual_clip_limit) {
                            ++histogram[j];
                            --remaining_to_distribute;
                        }
                    }
                    
                    for (unsigned long& bin : histogram) {
                        bin = std::min(bin, actual_clip_limit);
                    }
                }
            }
        };
        
        std::vector<std::thread> threads;
        size_t current_start = 0;
        
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            size_t work_size = histograms_per_thread + (t < remainder ? 1 : 0);
            size_t end_idx = current_start + work_size;
            
            threads.emplace_back(clip_range, current_start, end_idx);
            current_start = end_idx;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Single-threaded baseline implementation
        for (auto& histogram : histograms_) {
            unsigned long excess = 0;
            for (unsigned long& bin : histogram) {
                if (bin > actual_clip_limit) {
                    excess += bin - actual_clip_limit;
                    bin = actual_clip_limit;
                }
            }
            
            unsigned long redistribution_per_bin = excess / config_.bins;
            unsigned long remainder = excess % config_.bins;
            
            for (size_t j = 0; j < histogram.size(); ++j) {
                histogram[j] += redistribution_per_bin;
                if (j < remainder) {
                    ++histogram[j];
                }
                histogram[j] = std::min(histogram[j], actual_clip_limit);
            }
        }
    }
    
    if (config_.collect_metrics) {
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics_.clipping_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
    }
}

void ModernCLAHE::compute_mappings(unsigned char min_val, unsigned char max_val) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int region_width = image_width_ / config_.grid_width;
    const int region_height = image_height_ / config_.grid_height;
    const unsigned long pixels_per_region = region_width * region_height;
    const double scale = static_cast<double>(max_val - min_val) / pixels_per_region;
    
    // Single-threaded mapping computation
    for (auto& histogram : histograms_) {
        // Convert histogram to cumulative distribution and map to output range
        unsigned long cumsum = 0;
        for (unsigned long& bin : histogram) {
            cumsum += bin;
            bin = min_val + static_cast<unsigned long>(cumsum * scale);
            bin = std::min(static_cast<unsigned long>(max_val), bin);
        }
    }
    
    if (config_.collect_metrics) {
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics_.mapping_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
    }
}

void ModernCLAHE::interpolate_image(cv::Mat& image) const {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int region_width = image.cols / config_.grid_width;
    const int region_height = image.rows / config_.grid_height;
    
    if (config_.use_threading) {
        // Multi-threaded implementation
        const int NUM_THREADS = 4;
        const int total_rows = (config_.grid_height + 1);
        const int rows_per_thread = total_rows / NUM_THREADS;
        const int remainder_rows = total_rows % NUM_THREADS;
        
        auto process_row_range = [this, &image, region_width, region_height](int start_row, int end_row) {
            for (int gy = start_row; gy < end_row; ++gy) {
                for (int gx = 0; gx <= static_cast<int>(config_.grid_width); ++gx) {
                    interpolate_region(image, gx, gy, region_width, region_height, gx, gy);
                }
            }
        };
        
        std::vector<std::thread> threads;
        int current_start = 0;
        
        for (int t = 0; t < NUM_THREADS; ++t) {
            int work_size = rows_per_thread + (t < remainder_rows ? 1 : 0);
            int end_row = current_start + work_size;
            
            end_row = std::min(end_row, total_rows);
            
            if (current_start < end_row) {
                threads.emplace_back(process_row_range, current_start, end_row);
            }
            current_start = end_row;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Single-threaded baseline implementation
        for (int gy = 0; gy <= static_cast<int>(config_.grid_height); ++gy) {
            for (int gx = 0; gx <= static_cast<int>(config_.grid_width); ++gx) {
                interpolate_region(image, gx, gy, region_width, region_height, gx, gy);
            }
        }
    }
    
    if (config_.collect_metrics) {
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics_.interpolation_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
    }
}

void ModernCLAHE::interpolate_region(cv::Mat& image, int grid_x, int grid_y,
                                   int region_width, int region_height, 
                                   int /* region_x */, int /* region_y */) const {
    // Calculate subregion boundaries
    int sub_x, sub_y;
    int x_left, x_right, y_upper, y_bottom;
    
    if (grid_x == 0) {
        sub_x = region_width / 2;
        x_left = x_right = 0;
    } else if (grid_x == static_cast<int>(config_.grid_width)) {
        sub_x = region_width / 2;
        x_left = x_right = config_.grid_width - 1;
    } else {
        sub_x = region_width;
        x_left = grid_x - 1;
        x_right = grid_x;
    }
    
    if (grid_y == 0) {
        sub_y = region_height / 2;
        y_upper = y_bottom = 0;
    } else if (grid_y == static_cast<int>(config_.grid_height)) {
        sub_y = region_height / 2;
        y_upper = y_bottom = config_.grid_height - 1;
    } else {
        sub_y = region_height;
        y_upper = grid_y - 1;
        y_bottom = grid_y;
    }
    
    // Get pointers to the four surrounding histograms
    const auto& hist_lu = histograms_[y_upper * config_.grid_width + x_left];   // Left-Upper
    const auto& hist_ru = histograms_[y_upper * config_.grid_width + x_right];  // Right-Upper  
    const auto& hist_lb = histograms_[y_bottom * config_.grid_width + x_left];  // Left-Bottom
    const auto& hist_rb = histograms_[y_bottom * config_.grid_width + x_right]; // Right-Bottom
    
    // Calculate starting position for this subregion
    int start_x = (grid_x == 0) ? 0 : (grid_x * region_width - sub_x / 2);
    int start_y = (grid_y == 0) ? 0 : (grid_y * region_height - sub_y / 2);
    
    start_x = std::max(0, std::min(image.cols - sub_x, start_x));
    start_y = std::max(0, std::min(image.rows - sub_y, start_y));
    
    const int norm_factor = sub_x * sub_y;
    
    // OPTIMIZATION: Check if normalization factor is power of 2 for bit shift
    int shift_amount = 0;
    bool use_shift = (norm_factor & (norm_factor - 1)) == 0; // Check if power of 2
    if (use_shift) {
        int temp = norm_factor;
        while (temp >>= 1) ++shift_amount;
    }
    
    // Bilinear interpolation for each pixel in the subregion
    for (int y = 0; y < sub_y && (start_y + y) < image.rows; ++y) {
        unsigned char* row = image.ptr<unsigned char>(start_y + y);
        const int y_coeff = y;
        const int y_inv_coeff = sub_y - y;
        
        for (int x = 0; x < sub_x && (start_x + x) < image.cols; ++x) {
            const int pixel_idx = start_x + x;
            const unsigned char grey_val = lut_[row[pixel_idx]];
            const int x_coeff = x;
            const int x_inv_coeff = sub_x - x;
            
            // Bilinear interpolation between four histogram mappings
            unsigned long interpolated;
            if (use_shift) {
                // OPTIMIZATION: Use bit shift instead of division when possible
                interpolated = (
                    y_inv_coeff * (x_inv_coeff * hist_lu[grey_val] + x_coeff * hist_ru[grey_val]) +
                    y_coeff * (x_inv_coeff * hist_lb[grey_val] + x_coeff * hist_rb[grey_val])
                ) >> shift_amount;
            } else {
                interpolated = (
                    y_inv_coeff * (x_inv_coeff * hist_lu[grey_val] + x_coeff * hist_ru[grey_val]) +
                    y_coeff * (x_inv_coeff * hist_lb[grey_val] + x_coeff * hist_rb[grey_val])
                ) / norm_factor;
            }
            
            row[pixel_idx] = static_cast<unsigned char>(
                std::min(255UL, std::max(0UL, interpolated))
            );
        }
    }
}

void ModernCLAHE::set_config(const CLAHEConfig& config) {
    if (!config.is_valid()) {
        throw std::invalid_argument("Invalid CLAHE configuration: " + config.validation_error());
    }
    config_ = config;
}

// ============================================================================
// Convenience Functions
// ============================================================================

cv::Mat apply_clahe(const cv::Mat& src, double clip_limit, 
                   int grid_width, int grid_height) {
    CLAHEConfig config;
    config.clip_limit = clip_limit;
    config.grid_width = grid_width;
    config.grid_height = grid_height;
    
    ModernCLAHE clahe(config);
    return clahe.process(src);
}

void adaptive_equalize(const cv::Mat& src, cv::Mat& dst,
                      unsigned int x_divs, unsigned int y_divs,
                      unsigned int bins, bool use_full_range) {
    CLAHEConfig config;
    config.grid_width = x_divs;
    config.grid_height = y_divs; 
    config.bins = bins;
    config.clip_limit = -1.0; // Negative for AHE
    config.use_full_range = use_full_range;
    
    ModernCLAHE clahe(config);
    dst = clahe.process(src);
}

void contrast_limited_adaptive_equalize(const cv::Mat& src, cv::Mat& dst,
                                      unsigned int x_divs, unsigned int y_divs,
                                      unsigned int bins, double clip_limit,
                                      bool use_full_range) {
    CLAHEConfig config;
    config.grid_width = x_divs;
    config.grid_height = y_divs;
    config.bins = bins;
    config.clip_limit = clip_limit;
    config.use_full_range = use_full_range;
    
    ModernCLAHE clahe(config);
    dst = clahe.process(src);
}

// ============================================================================
// Comparison Utilities
// ============================================================================

namespace comparison {

void ComparisonResult::print_summary() const {
    std::cout << "\n=== CLAHE Implementation Comparison ===\n";
    std::cout << "Modern Implementation:\n";
    modern_metrics.print();
    std::cout << "\nOpenCV Implementation Time: " << opencv_time.count() << " μs\n";
    std::cout << "\nAccuracy Metrics:\n";
    std::cout << "  Mean Absolute Error: " << mean_absolute_error << "\n";
    std::cout << "  Peak SNR: " << peak_signal_noise_ratio << " dB\n";
    
    double speedup = static_cast<double>(opencv_time.count()) / modern_metrics.total_time.count();
    std::cout << "\nSpeedup: " << speedup << "x ";
    if (speedup > 1.0) {
        std::cout << "(Modern is faster)\n";
    } else if (speedup < 1.0) {
        std::cout << "(OpenCV is faster)\n"; 
    } else {
        std::cout << "(Equivalent performance)\n";
    }
}

ComparisonResult compare_implementations(const cv::Mat& src, const CLAHEConfig& config) {
    ComparisonResult result;
    
    // Test our modern implementation
    ModernCLAHE modern_clahe(config);
    CLAHEConfig modern_config = config;
    modern_config.collect_metrics = true;
    modern_clahe.set_config(modern_config);
    
    result.modern_result = modern_clahe.process(src);
    result.modern_metrics = modern_clahe.get_metrics();
    
    // Test OpenCV's implementation
    auto opencv_clahe = cv::createCLAHE(config.clip_limit, 
        cv::Size(config.grid_width, config.grid_height));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    opencv_clahe->apply(src, result.opencv_result);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    result.opencv_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    // Calculate accuracy metrics
    cv::Mat diff;
    cv::absdiff(result.modern_result, result.opencv_result, diff);
    result.mean_absolute_error = cv::mean(diff)[0];
    
    // Calculate PSNR
    cv::Mat diff_sq;
    diff.convertTo(diff_sq, CV_32F);
    diff_sq = diff_sq.mul(diff_sq);
    double mse = cv::mean(diff_sq)[0];
    result.peak_signal_noise_ratio = (mse == 0) ? 100.0 : 20.0 * std::log10(255.0 / std::sqrt(mse));
    
    return result;
}

} // namespace comparison

} // namespace clahe
