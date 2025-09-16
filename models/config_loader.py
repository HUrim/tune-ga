import json
import os
from typing import Dict, Any


class GAConfig:
    """Configuration class for Genetic Algorithm parameters"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.instance_type = config_dict.get("instance_type", "unknown")
        self.description = config_dict.get("description", "")
        
        # Basic GA parameters
        self.population_size = config_dict.get("population_size", 100)
        self.generations = config_dict.get("generations", 500)
        self.time_limit_sec = config_dict.get("time_limit_sec", 600)
        
        # Mutation parameters
        mutation = config_dict.get("mutation", {})
        self.mutation_initial_prob = mutation.get("initial_prob", 0.6)
        self.mutation_final_prob = mutation.get("final_prob", 0.25)
        self.mutation_adaptation_type = mutation.get("adaptation_type", "linear_decay")
        self.mutation_decay_factor = mutation.get("decay_factor", 0.95)
        self.mutation_increase_factor = mutation.get("increase_factor", 1.1)
        self.mutation_stagnation_threshold = mutation.get("stagnation_threshold", 0.001)
        
        # Crossover parameters
        crossover = config_dict.get("crossover", {})
        self.crossover_rate = crossover.get("rate", 0.8)
        self.crossover_types = crossover.get("types", ["order_preserving"])
        self.crossover_weights = crossover.get("weights", [1.0])
        
        # Selection parameters
        selection = config_dict.get("selection", {})
        self.tournament_size = selection.get("tournament_size", 10)
        self.selection_methods = selection.get("methods", ["tournament"])
        self.selection_weights = selection.get("weights", [1.0])
        
        # Immigration parameters
        immigration = config_dict.get("immigration", {})
        self.immigration_initial_frac = immigration.get("initial_frac", 0.06)
        self.immigration_max_frac = immigration.get("max_frac", 0.3)
        self.immigration_stagnation_multiplier = immigration.get("stagnation_multiplier", 1.5)
        self.immigration_stagnation_threshold = immigration.get("stagnation_threshold", 10)
        self.immigration_diversity_threshold = immigration.get("diversity_threshold", 0.1)
        
        # Local search parameters
        local_search = config_dict.get("local_search", {})
        self.tweak_initial_steps = local_search.get("initial_tweak_steps", 5)
        self.tweak_final_steps = local_search.get("final_tweak_steps", 2)
        self.tweak_adaptation_schedule = local_search.get("adaptation_schedule", "linear_decay")
        
        # Steady state parameters
        steady_state = config_dict.get("steady_state", {})
        self.steady_state_ratio = steady_state.get("ratio", 0.5)
        self.steady_state_switch_generation = steady_state.get("switch_generation", 250)
        self.steady_state_switch_time_ratio = steady_state.get("switch_time_ratio", 0.5)
        
        # Diversity parameters
        diversity = config_dict.get("diversity", {})
        self.diversity_enabled = diversity.get("enabled", True)
        self.diversity_measurement_interval = diversity.get("measurement_interval", 5)
        self.diversity_min_threshold = diversity.get("min_threshold", 0.1)
        self.diversity_trigger_immigration = diversity.get("trigger_immigration", True)
    
    def get_current_mutation_prob(self, generation: int, fitness_improvement: float, 
                                 current_prob: float) -> float:
        """Calculate current mutation probability based on adaptation strategy"""
        if self.mutation_adaptation_type == "fitness_based":
            if fitness_improvement < self.mutation_stagnation_threshold:
                # Increase exploration when stagnating
                return min(self.mutation_initial_prob, 
                          current_prob * self.mutation_increase_factor)
            else:
                # Increase exploitation when improving
                return max(self.mutation_final_prob, 
                          current_prob * self.mutation_decay_factor)
        
        elif self.mutation_adaptation_type == "linear_decay":
            # Linear interpolation between initial and final
            progress = generation / self.generations if self.generations > 0 else 0
            return self.mutation_initial_prob + progress * (
                self.mutation_final_prob - self.mutation_initial_prob)
        
        else:
            return current_prob
    
    def get_current_tweak_steps(self, generation: int, elapsed_time: float) -> int:
        """Calculate current number of tweak steps based on adaptation schedule"""
        if self.tweak_adaptation_schedule == "linear_decay":
            # Use time-based decay if time limit is set, otherwise use generation-based
            if self.time_limit_sec > 0:
                progress = elapsed_time / self.time_limit_sec
            else:
                progress = generation / self.generations if self.generations > 0 else 0
            
            progress = min(1.0, progress)  # Cap at 1.0
            current_steps = self.tweak_initial_steps + progress * (
                self.tweak_final_steps - self.tweak_initial_steps)
            return max(1, int(round(current_steps)))
        
        return self.tweak_initial_steps
    
    def should_switch_to_steady_state(self, generation: int, elapsed_time: float) -> bool:
        """Determine if should switch to steady-state evolution"""
        time_switch = (self.time_limit_sec > 0 and 
                      elapsed_time >= self.time_limit_sec * self.steady_state_switch_time_ratio)
        gen_switch = generation >= self.steady_state_switch_generation
        return time_switch or gen_switch


class ConfigLoader:
    """Loads and manages GA configuration files"""
    
    @staticmethod
    def load_config(config_path: str) -> GAConfig:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return GAConfig(config_dict)
    
    @staticmethod
    def get_config_for_instance(instance_data) -> GAConfig:
        """Automatically select configuration based on instance characteristics"""
        num_books = instance_data.num_books
        num_libs = instance_data.num_libs
        
        # Determine instance size
        if num_books <= 10000 and num_libs <= 100:
            config_type = "small"
        elif num_books <= 200000:
            config_type = "medium"
        else:
            config_type = "large"
        
        # Load appropriate configuration
        config_path = f"config/ga_config_{config_type}.json"
        
        # Fallback to default if config file doesn't exist
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found, using default parameters")
            return GAConfig({})
        
        return ConfigLoader.load_config(config_path)
    
    @staticmethod
    def save_config(config: GAConfig, config_path: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config.config, f, indent=2)
    
    @staticmethod
    def create_custom_config(base_config_path: str, modifications: Dict[str, Any], 
                           output_path: str) -> GAConfig:
        """Create a custom configuration by modifying an existing one"""
        base_config = ConfigLoader.load_config(base_config_path)
        
        # Apply modifications
        config_dict = base_config.config.copy()
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(config_dict, modifications)
        
        # Create new config and save
        new_config = GAConfig(config_dict)
        ConfigLoader.save_config(new_config, output_path)
        
        return new_config
