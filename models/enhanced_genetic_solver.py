import random
import time
from typing import Tuple, List
from models.solution import Solution
from models.instance_data import InstanceData
from models.config_loader import GAConfig
from models.diversity_manager import DiversityManager
from models.advanced_crossover import AdvancedCrossover
from models.selection_strategies import SelectionStrategies
from models.tweaks import Tweaks


class EnhancedGeneticSolver:
    """Enhanced Genetic Algorithm solver with advanced features"""
    
    def __init__(self, initial_solution: Solution, instance: InstanceData, config: GAConfig):
        self.initial_solution = initial_solution
        self.instance = instance
        self.config = config
        
        # Initialize dynamic parameters
        self.current_mutation_prob = config.mutation_initial_prob
        self.current_immigrant_frac = config.immigration_initial_frac
        self.population = []
        
        # Tracking variables
        self.best_fitness_history = []
        self.diversity_history = []
        self.generation_count = 0
        self.start_time = None
        self.plateau_counter = 0
        self.last_best_fitness = None
        
        # Advanced features
        self.diversity_manager = DiversityManager()
        self.use_steady_state = False
        
    def solve(self) -> Solution:
        """Main solving method with enhanced features"""
        print(f"Starting Enhanced GA with {self.config.instance_type} configuration")
        print(f"Population: {self.config.population_size}, Time limit: {self.config.time_limit_sec}s")
        
        # Initialize population
        self.population = self._initialize_population()
        self.start_time = time.time()
        
        for generation in range(self.config.generations):
            self.generation_count = generation
            elapsed_time = time.time() - self.start_time
            
            # Check time limit
            if elapsed_time >= self.config.time_limit_sec:
                print(f"Stopping at generation {generation} due to time limit ({elapsed_time:.1f}s)")
                break
            
            # Evaluate and sort population
            self.population.sort(key=lambda x: x.fitness_score, reverse=True)
            best_solution = self.population[0]
            
            # Track progress
            self._update_tracking_variables(best_solution, elapsed_time)
            
            # Adapt parameters
            self._adapt_parameters(elapsed_time)
            
            # Check diversity and manage if needed
            if self.config.diversity_enabled and generation % self.config.diversity_measurement_interval == 0:
                try:
                    self._manage_diversity()
                except Exception as e:
                    print(f"Warning: Diversity calculation failed, disabling: {e}")
                    self.config.diversity_enabled = False
            
            # Determine evolution strategy
            self.use_steady_state = self.config.should_switch_to_steady_state(generation, elapsed_time)
            
            # Create new generation
            if self.use_steady_state:
                new_population = self._create_steady_state_generation()
            else:
                new_population = self._create_generational_population()
            
            # Add immigrants if needed
            new_population = self._add_immigrants(new_population)
            
            # Ensure elite preservation
            new_population = self._preserve_elites(new_population, best_solution)
            
            # Update population
            self.population = new_population[:self.config.population_size]
            
            # Print progress
            if generation % 10 == 0:
                diversity = self.diversity_manager.calculate_population_diversity(self.population)
                print(f"Gen {generation}: Best={best_solution.fitness_score:,}, "
                      f"Diversity={diversity:.3f}, Mutation={self.current_mutation_prob:.3f}")
        
        # Return best solution
        final_population = sorted(self.population, key=lambda x: x.fitness_score, reverse=True)
        best_final = final_population[0]
        
        print(f"Final best fitness: {best_final.fitness_score:,}")
        print(f"Total generations: {self.generation_count}")
        print(f"Total time: {time.time() - self.start_time:.1f}s")
        
        return best_final
    
    def _initialize_population(self) -> List[Solution]:
        """Initialize population with diversity"""
        population = [self.initial_solution.shallow_copy()]
        
        # Create diverse variations
        tweak_ratio = 0.6  # Higher ratio for more diversity
        num_tweaked = int(self.config.population_size * tweak_ratio)
        num_clones = self.config.population_size - num_tweaked - 1
        
        # Add tweaked solutions with varying intensity
        for i in range(num_tweaked):
            # Vary tweak intensity
            tweak_steps = random.randint(1, self.config.tweak_initial_steps * 2)
            tweaked = Tweaks.tweak_with_iterations(
                self.initial_solution, self.instance, iterations=tweak_steps)
            population.append(tweaked.shallow_copy())
        
        # Add direct clones
        for _ in range(num_clones):
            population.append(self.initial_solution.shallow_copy())
        
        return population
    
    def _create_generational_population(self) -> List[Solution]:
        """Create new generation using generational replacement"""
        new_population = []
        
        while len(new_population) < self.config.population_size:
            # Select parents
            if self.config.diversity_enabled and random.random() < 0.3:
                # Use diversity-based selection occasionally
                parents = self.diversity_manager.select_diverse_parents(self.population, 2)
            else:
                # Use standard selection
                selection_method = SelectionStrategies.choose_selection_method()
                parents = [selection_method(self.population) for _ in range(2)]
            
            # Crossover
            offspring1, offspring2 = self._perform_crossover(parents[0], parents[1])
            
            # Mutation
            if random.random() < self.current_mutation_prob:
                current_tweak_steps = self.config.get_current_tweak_steps(
                    self.generation_count, time.time() - self.start_time)
                offspring1 = Tweaks.tweak_with_iterations(
                    offspring1, self.instance, iterations=current_tweak_steps)
            
            if random.random() < self.current_mutation_prob:
                current_tweak_steps = self.config.get_current_tweak_steps(
                    self.generation_count, time.time() - self.start_time)
                offspring2 = Tweaks.tweak_with_iterations(
                    offspring2, self.instance, iterations=current_tweak_steps)
            
            new_population.extend([offspring1.shallow_copy(), offspring2.shallow_copy()])
        
        return new_population
    
    def _create_steady_state_generation(self) -> List[Solution]:
        """Create new generation using steady-state replacement"""
        # Keep best solutions
        elite_count = max(1, int(self.config.population_size * 0.1))
        new_population = self.population[:elite_count]
        
        # Generate offspring to fill remaining slots
        while len(new_population) < self.config.population_size:
            # Select parents
            selection_method = SelectionStrategies.choose_selection_method()
            parents = [selection_method(self.population) for _ in range(2)]
            
            # Crossover and mutation
            offspring1, offspring2 = self._perform_crossover(parents[0], parents[1])
            
            if random.random() < self.current_mutation_prob:
                current_tweak_steps = self.config.get_current_tweak_steps(
                    self.generation_count, time.time() - self.start_time)
                offspring1 = Tweaks.tweak_with_iterations(
                    offspring1, self.instance, iterations=current_tweak_steps)
            
            if random.random() < self.current_mutation_prob:
                current_tweak_steps = self.config.get_current_tweak_steps(
                    self.generation_count, time.time() - self.start_time)
                offspring2 = Tweaks.tweak_with_iterations(
                    offspring2, self.instance, iterations=current_tweak_steps)
            
            # Add offspring
            candidates = [offspring1, offspring2]
            for candidate in candidates:
                if len(new_population) < self.config.population_size:
                    new_population.append(candidate.shallow_copy())
        
        # Sort and return best
        new_population.sort(key=lambda x: x.fitness_score, reverse=True)
        return new_population[:self.config.population_size]
    
    def _perform_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Perform crossover using configured method"""
        if random.random() < self.config.crossover_rate:
            # Use advanced crossover methods
            crossover_method = AdvancedCrossover.choose_crossover_method(
                self.config.crossover_types, self.config.crossover_weights)
            return crossover_method(parent1, parent2, self.instance)
        else:
            # Return parents unchanged
            return parent1.shallow_copy(), parent2.shallow_copy()
    
    def _adapt_parameters(self, elapsed_time: float):
        """Adapt parameters based on current state"""
        # Calculate fitness improvement
        current_best = max(self.population, key=lambda x: x.fitness_score).fitness_score
        fitness_improvement = 0.0
        
        if self.last_best_fitness is not None:
            fitness_improvement = (current_best - self.last_best_fitness) / max(1, self.last_best_fitness)
        
        # Update mutation probability
        self.current_mutation_prob = self.config.get_current_mutation_prob(
            self.generation_count, fitness_improvement, self.current_mutation_prob)
        
        # Update plateau counter
        if fitness_improvement < self.config.mutation_stagnation_threshold:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0
            # Reset immigration fraction on improvement
            self.current_immigrant_frac = self.config.immigration_initial_frac
        
        # Increase immigration on stagnation
        if self.plateau_counter > self.config.immigration_stagnation_threshold:
            self.current_immigrant_frac = min(
                self.config.immigration_max_frac,
                self.current_immigrant_frac * self.config.immigration_stagnation_multiplier
            )
        
        self.last_best_fitness = current_best
    
    def _manage_diversity(self):
        """Manage population diversity"""
        current_diversity = self.diversity_manager.calculate_population_diversity(self.population)
        self.diversity_history.append(current_diversity)
        
        if current_diversity < self.config.diversity_min_threshold:
            if self.config.diversity_trigger_immigration:
                # Trigger additional immigration
                self.current_immigrant_frac = min(
                    self.config.immigration_max_frac,
                    self.current_immigrant_frac * 2.0
                )
            
            # Maintain diversity by replacing similar solutions
            self.population = self.diversity_manager.maintain_diversity(
                self.population, self.instance, self.config.diversity_min_threshold)
    
    def _add_immigrants(self, population: List[Solution]) -> List[Solution]:
        """Add immigrant solutions to population"""
        num_immigrants = int(self.current_immigrant_frac * self.config.population_size)
        
        if num_immigrants > 0:
            # Generate diverse immigrants
            immigrants = self.diversity_manager.generate_diverse_immigrants(
                self.initial_solution, self.instance, num_immigrants, diversity_factor=0.5)
            
            # Replace worst individuals
            population.sort(key=lambda x: x.fitness_score, reverse=True)
            population = population[:-num_immigrants] + immigrants
        
        return population
    
    def _preserve_elites(self, population: List[Solution], best_solution: Solution) -> List[Solution]:
        """Ensure best solution is preserved"""
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        if population[0].fitness_score < best_solution.fitness_score:
            population[-1] = best_solution.shallow_copy()
        
        return population
    
    def _update_tracking_variables(self, best_solution: Solution, elapsed_time: float):
        """Update tracking variables for analysis"""
        self.best_fitness_history.append(best_solution.fitness_score)
        
        if self.config.diversity_enabled:
            try:
                diversity = self.diversity_manager.calculate_population_diversity(self.population)
                self.diversity_history.append(diversity)
            except Exception as e:
                print(f"Warning: Diversity tracking failed, disabling: {e}")
                self.config.diversity_enabled = False
                self.diversity_history.append(0.0)
    
    def get_statistics(self) -> dict:
        """Get solver statistics"""
        return {
            'best_fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history,
            'final_mutation_prob': self.current_mutation_prob,
            'final_immigrant_frac': self.current_immigrant_frac,
            'generations_completed': self.generation_count,
            'plateau_counter': self.plateau_counter,
            'used_steady_state': self.use_steady_state
        }
