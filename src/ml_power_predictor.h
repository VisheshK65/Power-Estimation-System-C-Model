#ifndef ML_POWER_PREDICTOR_H
#define ML_POWER_PREDICTOR_H

#include <systemc>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>
#include <chrono>

/**
 * Championship-Grade ML Power Predictor - DVCon Europe 2025
 * Advanced ML Power Prediction with Data-Driven Training
 * 
 * Features:
 * - Proper neural network training from reference data
 * - Comprehensive performance metrics (MAPE, RMSE, R²)
 * - Advanced feature engineering with realistic sensor modeling
 * - Validated state power hierarchy
 * - Competition-optimized architecture
 */

struct TrainingDataPoint {
    double timestamp;
    double power_watts;
    int state;
    std::vector<double> raw_features;
    std::vector<double> engineered_features;
};

struct PredictionFeatures {
    // Core state features
    int current_state;
    int previous_state;
    double time_seconds;
    
    // Temporal features
    double time_of_day_normalized;    // [0,1] for 24h cycle
    double day_of_week_normalized;    // [0,1] for weekly cycle
    double time_since_start;          // Normalized simulation time
    
    // State transition features
    std::vector<int> state_history_5;     // Last 5 states
    std::vector<int> state_history_10;    // Last 10 states
    double state_transition_rate;         // Transitions per hour
    double time_since_last_transition;    // Seconds since state change
    
    // Power consumption features
    double recent_avg_power_1min;         // Average power last minute
    double recent_avg_power_5min;         // Average power last 5 minutes
    double power_trend_slope;             // Linear trend in recent power
    double power_volatility;              // Power variance measure
    
    // Environmental and contextual features
    double ambient_temperature_est;       // Realistic temperature model
    double bluetooth_intensity;           // Bluetooth usage intensity
    double cpu_load_estimate;            // Estimated CPU utilization
    double battery_level_estimate;       // Estimated battery percentage
    bool is_work_hours;                  // 9AM-5PM weekdays
    bool is_weekend;                     // Saturday/Sunday
    
    // Pattern recognition features
    bool periodic_pattern_detected;      // Repetitive behavior
    int pattern_length;                  // Length of detected pattern
    double pattern_confidence;           // Confidence in pattern
    
    // State-specific features
    double bluetooth_state_duration;     // Time spent in BT states
    double work_state_duration;          // Time spent in work states
    double transition_matrix_prob;       // P(current|previous)
};

struct ModelPerformanceMetrics {
    // Core ML metrics
    double mape;                    // Mean Absolute Percentage Error
    double rmse;                    // Root Mean Square Error  
    double mae;                     // Mean Absolute Error
    double r_squared;               // Coefficient of determination
    double adjusted_r_squared;      // Adjusted R²
    
    // Statistical validation
    double mean_residual;           // Bias measure
    double residual_std_dev;        // Prediction uncertainty
    double confidence_interval_95;  // 95% CI width
    
    // Competition metrics
    double execution_time_ms;       // Average prediction time
    double memory_usage_mb;         // Memory footprint
    double training_time_ms;        // Model training time
    
    // Data quality
    int training_samples;
    int validation_samples;
    int test_samples;
    double data_coverage_percent;   // % of state space covered
    
    // Model diagnostics
    std::vector<double> residuals;
    std::vector<double> predictions;
    std::vector<double> actuals;
    std::string model_architecture;
};

class ChampionshipNeuralNetwork {
private:
    struct Layer {
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        std::string activation;  // "relu", "tanh", "sigmoid", "linear"
    };
    
    std::vector<Layer> layers_;
    int input_size_;
    std::vector<int> hidden_sizes_;
    int output_size_;
    double learning_rate_;
    double l2_regularization_;
    std::mt19937 rng_;
    
    // Training state
    std::vector<std::vector<double>> training_inputs_;
    std::vector<double> training_targets_;
    bool is_trained_;
    
public:
    ChampionshipNeuralNetwork(int input_size, const std::vector<int>& hidden_sizes, 
                             int output_size, double learning_rate = 0.001);
    
    // Network operations
    void initialize_xavier_weights();
    double activate(double x, const std::string& activation_type);
    double activate_derivative(double x, const std::string& activation_type);
    
    // Training
    void add_training_data(const std::vector<double>& inputs, double target);
    void train_with_backprop(int epochs = 1000, double validation_split = 0.2);
    double calculate_loss(const std::vector<std::vector<double>>& inputs,
                         const std::vector<double>& targets);
    
    // Prediction
    double predict(const std::vector<double>& inputs);
    std::vector<double> predict_batch(const std::vector<std::vector<double>>& inputs);
    
    // Model persistence
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);
    
    // Diagnostics
    void print_architecture();
    bool is_trained() const { return is_trained_; }
};

class StatePowerAnalyzer {
private:
    struct StateStatistics {
        double mean_power;
        double std_dev_power;
        double min_power;
        double max_power;
        int sample_count;
        std::vector<double> power_samples;
        
        // Temporal patterns
        std::map<int, double> hourly_patterns;  // Hour -> avg power
        std::map<int, double> daily_patterns;   // Day -> avg power
        
        // Transition effects
        std::map<int, double> transition_costs; // From_state -> power delta
    };
    
    std::map<int, StateStatistics> state_stats_;
    bool power_hierarchy_validated_;
    
public:
    void analyze_training_data(const std::vector<TrainingDataPoint>& data);
    void validate_bluetooth_hierarchy();
    void validate_work_state_hierarchy();
    
    double get_state_baseline(int state);
    double get_state_variance(int state);
    double get_transition_cost(int from_state, int to_state);
    
    // Power hierarchy validation
    bool is_hierarchy_valid();
    std::string get_hierarchy_report();
};

class AdvancedFeatureExtractor {
private:
    std::vector<TrainingDataPoint> historical_data_;
    std::map<std::pair<int,int>, double> transition_matrix_;
    
    // Environmental modeling
    double base_temperature_;
    std::vector<double> temperature_history_;
    
    // Pattern detection
    struct Pattern {
        std::vector<int> state_sequence;
        double avg_power;
        double confidence;
        int occurrences;
    };
    std::vector<Pattern> detected_patterns_;
    
public:
    AdvancedFeatureExtractor();
    
    // Feature extraction
    PredictionFeatures extract_comprehensive_features(
        int current_state, double current_time,
        const std::vector<std::pair<int, double>>& recent_history);
    
    // Environmental modeling
    double model_realistic_temperature(double time_seconds);
    double estimate_cpu_load(int state, double time_seconds);
    double estimate_battery_level(double time_seconds, 
                                 const std::vector<int>& recent_states);
    
    // Pattern detection
    void detect_behavioral_patterns();
    bool matches_known_pattern(const std::vector<int>& recent_states);
    double get_pattern_confidence(const std::vector<int>& recent_states);
    
    // Contextual features
    bool is_work_hours(double time_seconds);
    bool is_weekend(double time_seconds);
    double calculate_bluetooth_intensity(const std::vector<int>& recent_states);
    
    // Data management
    void add_historical_data(const TrainingDataPoint& data);
    void build_transition_matrix();

    // Utilities
    std::mt19937& get_rng() { return rng_; }

private:
    std::mt19937 rng_;
};

class ChampionshipMLPredictor {
private:
    // Core ML components
    std::unique_ptr<ChampionshipNeuralNetwork> neural_net_;
    std::unique_ptr<StatePowerAnalyzer> state_analyzer_;
    std::unique_ptr<AdvancedFeatureExtractor> feature_extractor_;
    
    // Training data management
    std::vector<TrainingDataPoint> training_data_;
    std::vector<TrainingDataPoint> validation_data_;
    std::vector<TrainingDataPoint> test_data_;
    
    // Model state
    ModelPerformanceMetrics performance_;
    bool is_trained_;
    bool is_validated_;
    
    // Ensemble components
    struct EnsembleComponent {
        std::string name;
        double weight;
        std::function<double(const PredictionFeatures&)> predictor;
    };
    std::vector<EnsembleComponent> ensemble_models_;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point training_start_;
    std::chrono::high_resolution_clock::time_point last_prediction_;
    
public:
    ChampionshipMLPredictor();
    ~ChampionshipMLPredictor() = default;
    
    // === TRAINING PIPELINE ===
    bool load_and_parse_reference_data(const std::string& csv_path);
    void preprocess_and_clean_data();
    void split_data_strategically(double validation_ratio = 0.15, double test_ratio = 0.15);
    void train_comprehensive_model();
    void validate_model_performance();
    
    // === PREDICTION ===
    double predict_power_championship(int state, double time_seconds,
                                    const std::vector<std::pair<int, double>>& recent_history);
    
    // === ENSEMBLE METHODS ===
    double neural_network_prediction(const PredictionFeatures& features);
    double statistical_baseline_prediction(const PredictionFeatures& features);
    double temporal_pattern_prediction(const PredictionFeatures& features);
    double state_transition_prediction(const PredictionFeatures& features);
    
    // === PERFORMANCE & VALIDATION ===
    void calculate_comprehensive_metrics();
    ModelPerformanceMetrics get_performance_metrics() const { return performance_; }
    void validate_state_power_hierarchy();
    
    // === DIAGNOSTICS & EXPORT ===
    void export_model_diagnostics(const std::string& filename);
    void export_training_curves(const std::string& filename);
    void generate_competition_report(const std::string& filename);
    
    // === UTILITIES ===
    bool is_model_ready() const { return is_trained_ && is_validated_; }
    std::string get_model_summary() const;
    void print_training_progress();

    // === ENVIRONMENTAL MODELING ===
    double get_realistic_noise_std(int state);
    double calculate_environmental_effects(int state, double time_seconds,
                                         const std::vector<std::pair<int, double>>& recent_history);
    std::vector<int> extract_recent_states(const std::vector<std::pair<int, double>>& recent_history);
};

#endif // ML_POWER_PREDICTOR_H