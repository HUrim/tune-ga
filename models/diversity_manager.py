import random
from typing import List, Set
from models.solution import Solution


class DiversityManager:
    """Manages population diversity measurement and control"""
    
    @staticmethod
    def calculate_hamming_distance(sol1: Solution, sol2: Solution) -> float:
        """Calculate normalized Hamming distance between two solutions"""
        if not sol1.signed_libraries or not sol2.signed_libraries:
            return 1.0  # Maximum distance if one solution is empty
        
        # Convert to sets for comparison
        set1 = set(sol1.signed_libraries)
        set2 = set(sol2.signed_libraries)
        
        # Calculate Jaccard distance (1 - Jaccard similarity)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    
    @staticmethod
    def calculate_order_distance(sol1: Solution, sol2: Solution) -> float:
        """Calculate distance based on library ordering"""
        if not sol1.signed_libraries or not sol2.signed_libraries:
            return 1.0
        
        # Find common libraries
        common_libs = set(sol1.signed_libraries).intersection(set(sol2.signed_libraries))
        
        if len(common_libs) < 2:
            return 1.0
        
        # Create position mappings for common libraries
        pos1 = {lib: i for i, lib in enumerate(sol1.signed_libraries) if lib in common_libs}
        pos2 = {lib: i for i, lib in enumerate(sol2.signed_libraries) if lib in common_libs}
        
        # Calculate normalized Kendall tau distance
        inversions = 0
        total_pairs = 0
        
        common_libs_list = list(common_libs)
        for i in range(len(common_libs_list)):
            for j in range(i + 1, len(common_libs_list)):
                lib_i, lib_j = common_libs_list[i], common_libs_list[j]
                
                # Check if order is different
                order1 = pos1[lib_i] < pos1[lib_j]
                order2 = pos2[lib_i] < pos2[lib_j]
                
                if order1 != order2:
                    inversions += 1
                total_pairs += 1
        
        return inversions / total_pairs if total_pairs > 0 else 0.0
    
    @staticmethod
    def calculate_population_diversity(population: List[Solution]) -> float:
        """Calculate average diversity of the population"""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Combine Hamming and order distances
                hamming_dist = DiversityManager.calculate_hamming_distance(
                    population[i], population[j])
                order_dist = DiversityManager.calculate_order_distance(
                    population[i], population[j])
                
                # Weighted combination
                combined_distance = 0.6 * hamming_dist + 0.4 * order_dist
                total_distance += combined_distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    @staticmethod
    def calculate_diversity_entropy(population: List[Solution]) -> float:
        """Calculate diversity using entropy of library usage"""
        if not population:
            return 0.0
        
        # Count frequency of each library being signed
        library_counts = {}
        total_solutions = len(population)
        
        for solution in population:
            for lib_id in solution.signed_libraries:
                library_counts[lib_id] = library_counts.get(lib_id, 0) + 1
        
        if not library_counts:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in library_counts.values():
            probability = count / total_solutions
            if probability > 0:
                entropy -= probability * (probability ** 0.5)  # Modified entropy
        
        # Normalize by maximum possible entropy
        max_entropy = len(library_counts) ** 0.5
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    @staticmethod
    def is_diverse_enough(population: List[Solution], threshold: float = 0.1) -> bool:
        """Check if population diversity is above threshold"""
        diversity = DiversityManager.calculate_population_diversity(population)
        return diversity >= threshold
    
    @staticmethod
    def select_diverse_parents(population: List[Solution], num_parents: int = 2) -> List[Solution]:
        """Select diverse parents for crossover"""
        if len(population) <= num_parents:
            return population[:num_parents]
        
        # Start with best solution
        selected = [population[0]]
        candidates = population[1:]
        
        # Select remaining parents to maximize diversity
        for _ in range(num_parents - 1):
            best_candidate = None
            best_min_distance = -1
            
            for candidate in candidates:
                # Calculate minimum distance to already selected parents
                min_distance = min(
                    DiversityManager.calculate_hamming_distance(candidate, selected_parent)
                    for selected_parent in selected
                )
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        return selected
    
    @staticmethod
    def generate_diverse_immigrants(base_solution: Solution, instance_data, 
                                  num_immigrants: int, diversity_factor: float = 0.5) -> List[Solution]:
        """Generate diverse immigrant solutions"""
        from models.tweaks import Tweaks
        
        immigrants = []
        
        for _ in range(num_immigrants):
            # Create a copy of base solution
            immigrant = base_solution.shallow_copy()
            
            # Apply more aggressive tweaks for diversity
            num_tweaks = int(5 * (1 + diversity_factor))
            
            for _ in range(num_tweaks):
                tweak_method = Tweaks.choose_tweak_method()
                immigrant = tweak_method(immigrant, instance_data)
            
            immigrants.append(immigrant)
        
        return immigrants
    
    @staticmethod
    def maintain_diversity(population: List[Solution], instance_data, 
                          min_diversity: float = 0.1) -> List[Solution]:
        """Maintain population diversity by replacing similar solutions"""
        current_diversity = DiversityManager.calculate_population_diversity(population)
        
        if current_diversity >= min_diversity:
            return population
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Keep best solutions and replace similar ones
        maintained_population = [population[0]]  # Keep best
        
        for candidate in population[1:]:
            # Check if candidate is too similar to existing solutions
            is_too_similar = any(
                DiversityManager.calculate_hamming_distance(candidate, existing) < 0.05
                for existing in maintained_population
            )
            
            if not is_too_similar:
                maintained_population.append(candidate)
            else:
                # Replace with a diversified version
                diverse_solution = DiversityManager.diversify_solution(
                    candidate, instance_data, diversity_factor=0.3)
                maintained_population.append(diverse_solution)
        
        return maintained_population
    
    @staticmethod
    def diversify_solution(solution: Solution, instance_data, 
                          diversity_factor: float = 0.2) -> Solution:
        """Create a diversified version of a solution"""
        from models.tweaks import Tweaks
        
        diversified = solution.shallow_copy()
        
        # Apply random tweaks
        num_tweaks = max(1, int(3 * diversity_factor))
        
        for _ in range(num_tweaks):
            tweak_method = Tweaks.choose_tweak_method()
            new_solution = tweak_method(diversified, instance_data)
            
            # Accept if it maintains reasonable fitness
            if new_solution.fitness_score >= diversified.fitness_score * 0.9:
                diversified = new_solution
        
        return diversified
