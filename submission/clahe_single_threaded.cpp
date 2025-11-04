/*
 * CLAHE - Contrast Limited Adaptive Histogram Equalization
 * Single-Threaded Implementation
 * 
 * CS441 - Operating Systems
 * Multithreading Programming Project
 * 
 * This file contains the single-threaded baseline implementation of CLAHE.
 * It processes images sequentially without using multiple threads.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Configuration structure for CLAHE algorithm
struct CLAHEConfig {
    int grid_width = 8;      // Number of horizontal tiles
    int grid_height = 8;     // Number of vertical tiles
    int bins = 256;          // Number of histogram bins
    double clip_limit = 2.0; // Contrast limiting factor
};

class CLAHESingleThreaded {
private:
    CLAHEConfig config_;
    std::vector<std::vector<unsigned long>> histograms_;
    std::vector<unsigned char> lut_;
    int image_width_;
    int image_height_;
    
public:
    CLAHESingleThreaded(const CLAHEConfig& config) : config_(config) {}
    
    // Main processing function
    cv::Mat process(const cv::Mat& src) {
        if (src.empty() || src.type() != CV_8UC1) {
            std::cerr << "Error: Input must be 8-bit grayscale image" << std::endl;
            return cv::Mat();
        }
        
        cv::Mat dst = src.clone();
        process_inplace(dst);
        return dst;
    }
    
private:
    void process_inplace(cv::Mat& image) {
        image_width_ = image.cols;
        image_height_ = image.rows;
        
        // Find min/max pixel values
        double min_val, max_val;
        cv::minMaxLoc(image, &min_val, &max_val);
        
        // Step 1: Compute histograms for each tile
        compute_histograms(image, static_cast<unsigned char>(min_val), 
                          static_cast<unsigned char>(max_val));
        
        // Step 2: Clip histograms to limit contrast
        clip_histograms();
        
        // Step 3: Compute cumulative distribution and mapping
        compute_mappings(static_cast<unsigned char>(min_val), 
                        static_cast<unsigned char>(max_val));
        
        // Step 4: Interpolate and apply transformation
        interpolate_image(image);
    }
    
    // Create lookup table for histogram binning
    void make_lut(unsigned char min_val, unsigned char max_val) {
        lut_.resize(256);
        
        if (min_val >= max_val) {
            std::fill(lut_.begin(), lut_.end(), 0);
            return;
        }
        
        const float scale = static_cast<float>(config_.bins - 1) / (max_val - min_val);
        for (int i = 0; i < 256; ++i) {
            if (i < min_val) {
                lut_[i] = 0;
            } else if (i > max_val) {
                lut_[i] = config_.bins - 1;
            } else {
                lut_[i] = static_cast<unsigned char>((i - min_val) * scale);
            }
        }
    }
    
    // Step 1: Build histograms for each grid tile
    void compute_histograms(const cv::Mat& image, unsigned char min_val, unsigned char max_val) {
        make_lut(min_val, max_val);
        
        const int region_width = image_width_ / config_.grid_width;
        const int region_height = image_height_ / config_.grid_height;
        const size_t num_histograms = config_.grid_width * config_.grid_height;
        
        // Initialize histograms
        histograms_.resize(num_histograms);
        for (auto& hist : histograms_) {
            hist.resize(config_.bins, 0);
        }
        
        // Compute histogram for each tile sequentially
        for (int gy = 0; gy < config_.grid_height; ++gy) {
            for (int gx = 0; gx < config_.grid_width; ++gx) {
                size_t hist_idx = gy * config_.grid_width + gx;
                
                int start_x = gx * region_width;
                int start_y = gy * region_height;
                
                // Build histogram for this tile
                for (int y = start_y; y < start_y + region_height; ++y) {
                    const unsigned char* row = image.ptr<unsigned char>(y);
                    for (int x = start_x; x < start_x + region_width; ++x) {
                        unsigned char bin = lut_[row[x]];
                        ++histograms_[hist_idx][bin];
                    }
                }
            }
        }
    }
    
    // Step 2: Apply contrast limiting by clipping histograms
    void clip_histograms() {
        if (config_.clip_limit <= 0) {
            return; // No clipping
        }
        
        const int region_width = image_width_ / config_.grid_width;
        const int region_height = image_height_ / config_.grid_height;
        const unsigned long pixels_per_region = region_width * region_height;
        const unsigned long clip_limit = static_cast<unsigned long>(
            config_.clip_limit * pixels_per_region / config_.bins);
        const unsigned long actual_clip_limit = std::max(1UL, clip_limit);
        
        // Process each histogram sequentially
        for (auto& histogram : histograms_) {
            // Clip excess values
            unsigned long excess = 0;
            for (unsigned long& bin : histogram) {
                if (bin > actual_clip_limit) {
                    excess += bin - actual_clip_limit;
                    bin = actual_clip_limit;
                }
            }
            
            // Redistribute excess uniformly
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
    
    // Step 3: Create cumulative distribution function (mapping)
    void compute_mappings(unsigned char min_val, unsigned char max_val) {
        const int region_width = image_width_ / config_.grid_width;
        const int region_height = image_height_ / config_.grid_height;
        const unsigned long pixels_per_region = region_width * region_height;
        
        if (pixels_per_region == 0) return;
        
        const float scale = (max_val - min_val) / static_cast<float>(pixels_per_region);
        
        // Convert each histogram to CDF and create mapping
        for (auto& histogram : histograms_) {
            unsigned long cumulative = 0;
            for (size_t i = 0; i < histogram.size(); ++i) {
                cumulative += histogram[i];
                histogram[i] = min_val + static_cast<unsigned long>(cumulative * scale);
            }
        }
    }
    
    // Step 4: Apply transformation with bilinear interpolation
    void interpolate_image(cv::Mat& image) {
        const int region_width = image_width_ / config_.grid_width;
        const int region_height = image_height_ / config_.grid_height;
        
        // Process each interpolation region sequentially
        for (int gy = 0; gy <= config_.grid_height; ++gy) {
            for (int gx = 0; gx <= config_.grid_width; ++gx) {
                interpolate_region(image, gx, gy, region_width, region_height);
            }
        }
    }
    
    // Interpolate a single region
    void interpolate_region(cv::Mat& image, int grid_x, int grid_y,
                           int region_width, int region_height) {
        // Determine subregion boundaries
        int sub_x, sub_y;
        int x_left, x_right, y_upper, y_bottom;
        
        if (grid_x == 0) {
            sub_x = region_width / 2;
            x_left = x_right = 0;
        } else if (grid_x == config_.grid_width) {
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
        } else if (grid_y == config_.grid_height) {
            sub_y = region_height / 2;
            y_upper = y_bottom = config_.grid_height - 1;
        } else {
            sub_y = region_height;
            y_upper = grid_y - 1;
            y_bottom = grid_y;
        }
        
        // Get pointers to surrounding histograms
        const auto& hist_lu = histograms_[y_upper * config_.grid_width + x_left];
        const auto& hist_ru = histograms_[y_upper * config_.grid_width + x_right];
        const auto& hist_lb = histograms_[y_bottom * config_.grid_width + x_left];
        const auto& hist_rb = histograms_[y_bottom * config_.grid_width + x_right];
        
        // Calculate starting position
        int start_x = (grid_x == 0) ? 0 : (grid_x * region_width - sub_x / 2);
        int start_y = (grid_y == 0) ? 0 : (grid_y * region_height - sub_y / 2);
        
        start_x = std::max(0, std::min(image_width_ - sub_x, start_x));
        start_y = std::max(0, std::min(image_height_ - sub_y, start_y));
        
        const int norm_factor = sub_x * sub_y;
        
        // Apply bilinear interpolation
        for (int dy = 0; dy < sub_y && (start_y + dy) < image_height_; ++dy) {
            unsigned char* row = image.ptr<unsigned char>(start_y + dy);
            int weight_y_bottom = (dy * sub_y) / norm_factor;
            int weight_y_upper = sub_y - weight_y_bottom;
            
            for (int dx = 0; dx < sub_x && (start_x + dx) < image_width_; ++dx) {
                unsigned char pixel = row[start_x + dx];
                unsigned char bin = lut_[pixel];
                
                int weight_x_right = (dx * sub_x) / norm_factor;
                int weight_x_left = sub_x - weight_x_right;
                
                // Bilinear interpolation of four surrounding mappings
                unsigned long value = 
                    (hist_lu[bin] * weight_x_left * weight_y_upper +
                     hist_ru[bin] * weight_x_right * weight_y_upper +
                     hist_lb[bin] * weight_x_left * weight_y_bottom +
                     hist_rb[bin] * weight_x_right * weight_y_bottom) / norm_factor;
                
                row[start_x + dx] = static_cast<unsigned char>(std::min(255UL, value));
            }
        }
    }
};

// Main function for standalone testing
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        std::cout << "Example: " << argv[0] << " input.jpg output.jpg" << std::endl;
        return -1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    // Load image as grayscale
    cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image " << input_path << std::endl;
        return -1;
    }
    
    std::cout << "Single-Threaded CLAHE Processing" << std::endl;
    std::cout << "Input: " << input_path << std::endl;
    std::cout << "Size: " << image.cols << "x" << image.rows << " pixels" << std::endl;
    
    // Configure CLAHE
    CLAHEConfig config;
    config.grid_width = 8;
    config.grid_height = 8;
    config.clip_limit = 2.0;
    config.bins = 256;
    
    std::cout << "Grid: " << config.grid_width << "x" << config.grid_height << std::endl;
    std::cout << "Clip Limit: " << config.clip_limit << std::endl;
    
    // Process image
    auto start = std::chrono::high_resolution_clock::now();
    
    CLAHESingleThreaded clahe(config);
    cv::Mat result = clahe.process(image);
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (result.empty()) {
        std::cerr << "Error: Processing failed" << std::endl;
        return -1;
    }
    
    // Save result
    cv::imwrite(output_path, result);
    
    std::cout << "Processing time: " << time_ms << " ms" << std::endl;
    std::cout << "Output saved to: " << output_path << std::endl;
    
    return 0;
}
