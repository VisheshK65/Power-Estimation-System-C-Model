/**
 * main.cpp
 * DVCON Europe 2025 SystemC Modeling Challenge
 * Smart Engineer-Availability Indicator (SEAI) Power Model
 * Championship ML Implementation with Data-Driven Training
 */

#include <systemc>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <memory>

#include "ml_power_predictor.h"

/**
 * Championship ML Testbench Module
 * Advanced ML Power Prediction with proper training pipeline
 */
SC_MODULE(ChampionshipTestbenchModule) {
    sc_core::sc_port<sc_core::sc_signal_in_if<int>> status_input;
    
    // Power and energy tracking
    double powerEstimation = 0.0;
    double energyEstimation = 0.0;
    double last_time = 0.0;
    int sample_count = 0;
    
    // Championship ML Predictor
    std::unique_ptr<ChampionshipMLPredictor> ml_predictor_;
    
    // Data collection for CSV output
    std::vector<std::tuple<double, double, int>> power_samples; // time, power, status
    std::vector<std::pair<int, double>> recent_history;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point simulation_start_;
    double total_ml_execution_time_ = 0.0;
    int ml_prediction_count_ = 0;
    
    SC_CTOR(ChampionshipTestbenchModule) : status_input("input") {
        SC_THREAD(processing);
        sensitive << status_input;
        dont_initialize();
        
        // Initialize championship ML predictor
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "DVCON Europe 2025 SystemC Modeling Challenge" << std::endl;
        std::cout << "Smart Engineer-Availability Indicator (SEAI) Power Model" << std::endl;
        std::cout << "ML Ensemble Implementation - Championship Edition" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        initialize_championship_ml();
        
        simulation_start_ = std::chrono::high_resolution_clock::now();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ML Ensemble SEAI Power Model Initialized" << std::endl;
        std::cout << "Championship-Level ESP32 Power Prediction" << std::endl;
        std::cout << "6-State Model with Advanced Feature Engineering" << std::endl;
        std::cout << std::string(60, '=') << "\n" << std::endl;
    }
    
    void initialize_championship_ml() {
        ml_predictor_ = std::make_unique<ChampionshipMLPredictor>();
        
        // Competition-Grade ML Training Pipeline
        std::cout << "\n🏆 DVCon Challenge Competition-Grade ML Training! 🏆\n" << std::endl;
        std::cout << "Training advanced neural network on reference data...\n" << std::endl;

        bool data_loaded = ml_predictor_->load_and_parse_reference_data(
            "../ref_csv/DVConChallengeLongTimeMeasurement_States.csv");

        if (data_loaded) {
            std::cout << "\n⚡ Starting Comprehensive ML Training Pipeline..." << std::endl;
            ml_predictor_->preprocess_and_clean_data();
            ml_predictor_->split_data_strategically(0.20, 0.15); // 20% validation, 15% test
            ml_predictor_->train_comprehensive_model();

            // More lenient validation for realistic ML
            ml_predictor_->validate_model_performance();

            if (ml_predictor_->is_model_ready()) {
                std::cout << "\n🏆 Competition-Grade ML Model Successfully Trained! 🏆\n" << std::endl;
            } else {
                std::cout << "\n📡 Model training completed - using best effort predictions\n" << std::endl;
            }
        } else {
            std::cout << "\n⚠️ No reference data - using specification-based fallback\n" << std::endl;
        }
    }

    void processing() {
        std::cout << "ML Ensemble simulation started..." << std::endl;
        std::cout << "Target duration: 1296000 seconds (15 days)" << std::endl;
        std::cout << "Advanced Features: Neural Network + Temporal Patterns + Adaptive Learning" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        
        while (true) {
            int status = status_input->read();
            double current_time = sc_core::sc_time_stamp().to_seconds();
            
            // Championship ML power prediction with performance tracking
            auto ml_start = std::chrono::high_resolution_clock::now();
            
            // DVCon Challenge: Use state-specific baseline power
            double instantaneous_power = get_dvcon_baseline_power(status); // DVCon spec baseline
            if (ml_predictor_) {
                // Always try to use the ML predictor (it has fallback logic)
                double ml_prediction = ml_predictor_->predict_power_championship(
                    status, current_time, recent_history);
                instantaneous_power = ml_prediction;

                // Debug: Print first few predictions
                if (sample_count <= 10) {
                    std::cout << "[DEBUG] State " << status << ": Baseline=" << get_dvcon_baseline_power(status)
                              << "W, ML=" << ml_prediction << "W" << std::endl;
                }
            }
            
            auto ml_end = std::chrono::high_resolution_clock::now();
            double ml_time = std::chrono::duration<double, std::micro>(ml_end - ml_start).count() / 1000.0;
            
            // Update performance tracking
            total_ml_execution_time_ += ml_time;
            ml_prediction_count_++;
            
            // Update power and energy estimations
            double time_delta = current_time - last_time;
            sample_count++;
            
            // Cumulative average power
            powerEstimation = ((powerEstimation * (sample_count - 1)) + instantaneous_power) / sample_count;
            
            // Energy integration
            if (sample_count > 1 && time_delta > 0) {
                energyEstimation += instantaneous_power * time_delta;
            }
            
            last_time = current_time;
            
            // Store for CSV output with intelligent sampling
            bool should_store = false;
            if (sample_count <= 5000) {
                should_store = true; // High frequency for initial samples
            } else if (sample_count % 100 == 0) {
                should_store = true; // Every 100th sample thereafter
            }
            
            if (should_store) {
                power_samples.emplace_back(current_time, instantaneous_power, status);
            }
            
            // Update recent history for ML features (keep last 20 samples)
            recent_history.emplace_back(status, current_time);
            if (recent_history.size() > 20) {
                recent_history.erase(recent_history.begin());
            }
            
            // Display current state with championship formatting
            display_championship_state_info(status, instantaneous_power);

            wait();
        }
    }
    
    void display_championship_state_info(int status, double power) {
        std::string status_text = get_status_text(status);
        std::cout << "State " << status << ": " << status_text 
                  << " (" << std::fixed << std::setprecision(4) << power << "W)" << std::endl;
    }
    
    double get_dvcon_baseline_power(int status) const {
        // DVCon Challenge Specification Exact Values
        switch (status) {
            case 0: return 1.3;   // Boot
            case 1: return 0.96;  // Not at work
            case 2: return 1.25;  // Not at work BT sending
            case 3: return 0.97;  // At work in office
            case 4: return 1.31;  // At work in office BT sending
            case 5: return 1.0;   // At work not in office
            case 6: return 1.2;   // At work not in office BT sending
            default: return 1.05; // Fallback
        }
    }

    std::string get_status_text(int status) const {
        switch(status) {
            case 0: return "Boot";
            case 1: return "Not at Work";
            case 2: return "Not at Work BT Sending";
            case 3: return "At Work (In the Office)";
            case 4: return "At Work (In the Office) BT Sending";
            case 5: return "At Work (Not in the office)";
            case 6: return "At Work (Not in the office) BT Sending";
            default: return "Unknown";
        }
    }
    
    std::string format_time(double seconds) const {
        int total_seconds = static_cast<int>(seconds);
        int hours = total_seconds / 3600;
        int minutes = (total_seconds % 3600) / 60;
        int secs = total_seconds % 60;
        
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << hours << ":"
            << std::setfill('0') << std::setw(2) << minutes << ":"
            << std::setfill('0') << std::setw(2) << secs;
        return oss.str();
    }
    
    void export_championship_results() {
        // Export competition CSV
        export_competition_csv();
        
        // Export championship performance report
        export_performance_report();
        
        // Display final championship results
        display_championship_summary();
    }
    
    void export_competition_csv() {
        std::ofstream csv_file("seai_competition_results.csv");
        if (!csv_file.is_open()) {
            std::cerr << "Error: Could not create competition results CSV file" << std::endl;
            return;
        }
        
        // European CSV format with semicolon separators
        csv_file << "Timings;Voltage [V];Current Corrected [A] ;Power [W];Status;;;;;;" << std::endl;
        
        for (const auto& [time, power, status] : power_samples) {
            std::string time_str = format_time(time);
            
            // Convert power to current (P = V * I, assuming 5V)
            double current = power / 5.0;
            
            // European decimal notation (comma instead of period)
            std::ostringstream power_str, current_str;
            power_str << std::scientific << std::setprecision(2) << power;
            current_str << std::scientific << std::setprecision(2) << current;
            
            std::string power_european = power_str.str();
            std::string current_european = current_str.str();
            std::replace(power_european.begin(), power_european.end(), '.', ',');
            std::replace(current_european.begin(), current_european.end(), '.', ',');
            
            csv_file << time_str << ";5;" << current_european << ";" << power_european 
                    << ";" << get_status_text(status) << ";;;;;;" << std::endl;
        }
        
        csv_file.close();
        std::cout << "Competition results exported to: seai_competition_results.csv" << std::endl;
    }
    
    void export_performance_report() {
        std::ofstream report_file("seai_ml_performance.txt");
        if (!report_file.is_open()) {
            std::cerr << "Error: Could not create performance report file" << std::endl;
            return;
        }
        
        auto simulation_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            simulation_end - simulation_start_);
        
        report_file << "DVCON Europe 2025 - ML Ensemble Power Model Results" << std::endl;
        report_file << "========================================================" << std::endl;
        report_file << std::endl;
        report_file << "Model: ML Ensemble Champion" << std::endl;
        report_file << "Simulation Duration: " << static_cast<int>(last_time) << " s" << std::endl;
        report_file << "Total Samples: " << sample_count << std::endl;
        report_file << "Average Power: " << std::fixed << std::setprecision(6) << powerEstimation << " W" << std::endl;
        report_file << "Total Energy: " << std::fixed << std::setprecision(6) << energyEstimation << " J" << std::endl;
        report_file << "ML Predictions Made: " << ml_prediction_count_ << std::endl;
        
        double avg_ml_time = (ml_prediction_count_ > 0) ? 
                           (total_ml_execution_time_ / ml_prediction_count_) : 0.0;
        report_file << "Avg ML Execution Time: " << std::fixed << std::setprecision(3) << avg_ml_time << " ms" << std::endl;
        
        if (ml_predictor_ && ml_predictor_->is_model_ready()) {
            auto metrics = ml_predictor_->get_performance_metrics();
            report_file << std::endl;
            report_file << "Championship ML Performance Metrics:" << std::endl;
            report_file << "MAPE: " << std::fixed << std::setprecision(4) << metrics.mape << "%" << std::endl;
            report_file << "RMSE: " << std::fixed << std::setprecision(6) << metrics.rmse << " W" << std::endl;
            report_file << "R²: " << std::fixed << std::setprecision(6) << metrics.r_squared << std::endl;
            report_file << "Training Samples: " << metrics.training_samples << std::endl;
            report_file << "Validation Samples: " << metrics.validation_samples << std::endl;
            report_file << "Training Time: " << metrics.training_time_ms << " ms" << std::endl;
        }
        
        report_file << std::endl;
        report_file << "Competition Info:" << std::endl;
        report_file << "Challenge: DVCON Europe 2025 SystemC Modeling Challenge" << std::endl;
        report_file << "System: Smart Engineer-Availability Indicator (SEAI)" << std::endl;
        report_file << "Architecture: Championship ML Ensemble with Neural Network Training" << std::endl;
        report_file << "Total Simulation Time: " << total_duration.count() << " ms" << std::endl;
        
        report_file.close();
        std::cout << "Performance report exported to: seai_ml_performance.txt" << std::endl;
    }
    
    void display_championship_summary() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "CHAMPIONSHIP SIMULATION COMPLETED" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total Simulation Time: " << std::fixed << std::setprecision(1) 
                  << last_time << " seconds (" << (last_time/86400.0) << " days)" << std::endl;
        std::cout << "Total Samples: " << sample_count << std::endl;
        std::cout << "Average Power: " << std::fixed << std::setprecision(6) << powerEstimation << " W" << std::endl;
        std::cout << "Total Energy: " << std::fixed << std::setprecision(3) << energyEstimation/1000.0 << " kJ" << std::endl;
        std::cout << "ML Predictions: " << ml_prediction_count_ << std::endl;
        
        if (ml_prediction_count_ > 0) {
            double avg_ml_time = total_ml_execution_time_ / ml_prediction_count_;
            std::cout << "Avg ML Time: " << std::fixed << std::setprecision(3) << avg_ml_time << " ms/prediction" << std::endl;
        }
        
        if (ml_predictor_ && ml_predictor_->is_model_ready()) {
            auto metrics = ml_predictor_->get_performance_metrics();
            std::cout << "\nChampionship ML Performance:" << std::endl;
            std::cout << "  MAPE: " << std::fixed << std::setprecision(2) << metrics.mape << "%" << std::endl;
            std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << metrics.rmse << " W" << std::endl;
            std::cout << "  R²: " << std::fixed << std::setprecision(4) << metrics.r_squared << std::endl;
        }
        
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Championship Model Results:" << std::endl;
        std::cout << "✓ Competition CSV: seai_competition_results.csv" << std::endl;
        std::cout << "✓ Performance Report: seai_ml_performance.txt" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
};

/**
 * Championship Queue Implementation
 * Cycles through all 6 SEAI states as specified
 */
SC_MODULE(ChampionshipQueue) {
    sc_core::sc_out<int> queue_output;
    
    SC_CTOR(ChampionshipQueue) : queue_output("output") {
        SC_THREAD(queue_generation);
    }
    
    void queue_generation() {
        std::random_device rd;
        std::mt19937 gen(rd());

        // DVCon Challenge: Realistic IoT device state machine
        std::vector<int> operational_states = {1, 2, 3, 4, 5, 6};

        // State transition probabilities (realistic IoT behavior)
        std::map<int, std::vector<std::pair<int, double>>> transition_probs = {
            {1, {{1, 0.7}, {2, 0.05}, {3, 0.2}, {4, 0.03}, {5, 0.015}, {6, 0.005}}}, // Not at Work
            {2, {{1, 0.8}, {2, 0.1}, {3, 0.05}, {4, 0.03}, {5, 0.015}, {6, 0.005}}}, // Not at Work BT
            {3, {{3, 0.6}, {4, 0.1}, {5, 0.15}, {6, 0.05}, {1, 0.09}, {2, 0.01}}}, // At Work Office
            {4, {{3, 0.7}, {4, 0.15}, {5, 0.08}, {6, 0.04}, {1, 0.025}, {2, 0.005}}}, // At Work Office BT
            {5, {{5, 0.5}, {6, 0.15}, {3, 0.25}, {4, 0.08}, {1, 0.018}, {2, 0.002}}}, // At Work Remote
            {6, {{5, 0.6}, {6, 0.2}, {3, 0.12}, {4, 0.05}, {1, 0.025}, {2, 0.005}}} // At Work Remote BT
        };

        // Start with Boot state
        queue_output.write(0); // Boot state
        std::uniform_int_distribution<> boot_duration(5, 15);
        wait(boot_duration(gen), sc_core::SC_SEC);

        // Initial state after boot (typically "Not at Work")
        int current_state = 1;
        queue_output.write(current_state);

        while (true) {
            // Calculate realistic duration for current state
            std::uniform_int_distribution<> duration_dist;

            switch(current_state) {
                case 1: case 2: // Not at Work states - longer durations
                    duration_dist = std::uniform_int_distribution<>(120, 600); // 2-10 minutes
                    break;
                case 3: case 5: // Work states - medium durations
                    duration_dist = std::uniform_int_distribution<>(180, 900); // 3-15 minutes
                    break;
                case 4: case 6: // BT states - shorter durations
                    duration_dist = std::uniform_int_distribution<>(30, 300); // 0.5-5 minutes
                    break;
                default:
                    duration_dist = std::uniform_int_distribution<>(60, 300);
            }

            wait(duration_dist(gen), sc_core::SC_SEC);

            // Probabilistic state transition
            std::uniform_real_distribution<> prob_dist(0.0, 1.0);
            double rand_prob = prob_dist(gen);

            double cumulative_prob = 0.0;
            int next_state = current_state; // Default stay in same state

            if (transition_probs.find(current_state) != transition_probs.end()) {
                for (const auto& [state, prob] : transition_probs[current_state]) {
                    cumulative_prob += prob;
                    if (rand_prob <= cumulative_prob) {
                        next_state = state;
                        break;
                    }
                }
            }

            current_state = next_state;
            queue_output.write(current_state);
        }
    }
};

/**
 * Championship Main Function
 */
int sc_main(int argc, char* argv[]) {
    // Championship signal connections
    sc_core::sc_signal<int> queue_to_testbench;
    
    // Instantiate championship modules
    ChampionshipQueue championship_queue("ChampionshipQueue");
    ChampionshipTestbenchModule championship_testbench("ChampionshipTestbench");
    
    // Connect championship modules
    championship_queue.queue_output(queue_to_testbench);
    championship_testbench.status_input(queue_to_testbench);
    
    std::cout << "\n🏆 DVCON Europe 2025 Championship Simulation Starting 🏆\n" << std::endl;
    
    // Run championship simulation for 15 days
    sc_core::sc_start(15 * 24 * 3600, sc_core::SC_SEC); // 15 days
    
    std::cout << "\n🏁 Championship Simulation Finished 🏁\n" << std::endl;
    
    // Export championship results
    championship_testbench.export_championship_results();
    
    return 0;
}