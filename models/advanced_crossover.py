import random
from typing import List, Tuple, Set
from models.solution import Solution


class AdvancedCrossover:
    """Advanced crossover operators for scheduling problems"""
    
    @staticmethod
    def order_crossover(parent1: Solution, parent2: Solution, instance_data) -> Tuple[Solution, Solution]:
        """
        Order Crossover (OX) - preserves relative order from parents
        Specifically designed for permutation/scheduling problems
        """
        if not parent1.signed_libraries or not parent2.signed_libraries:
            return parent1.shallow_copy(), parent2.shallow_copy()
        
        def create_ox_offspring(p1_libs: List[int], p2_libs: List[int]) -> List[int]:
            size = len(p1_libs)
            if size <= 2:
                return p1_libs[:]
            
            # Select two random crossover points
            point1 = random.randint(0, size - 2)
            point2 = random.randint(point1 + 1, size - 1)
            
            # Initialize offspring with None
            offspring = [None] * size
            
            # Copy segment from parent1
            for i in range(point1, point2 + 1):
                offspring[i] = p1_libs[i]
            
            # Fill remaining positions with parent2's order
            p2_filtered = [lib for lib in p2_libs if lib not in offspring[point1:point2 + 1]]
            
            # Fill positions before crossover segment
            p2_idx = 0
            for i in range(point1):
                if p2_idx < len(p2_filtered):
                    offspring[i] = p2_filtered[p2_idx]
                    p2_idx += 1
            
            # Fill positions after crossover segment
            for i in range(point2 + 1, size):
                if p2_idx < len(p2_filtered):
                    offspring[i] = p2_filtered[p2_idx]
                    p2_idx += 1
            
            # Handle any remaining None values (shouldn't happen with valid inputs)
            for i in range(size):
                if offspring[i] is None:
                    # Find unused library
                    used = set(lib for lib in offspring if lib is not None)
                    available = set(p1_libs) - used
                    if available:
                        offspring[i] = random.choice(list(available))
                    else:
                        offspring[i] = p1_libs[i]  # Fallback
            
            return offspring
        
        # Create offspring using OX
        offspring1_libs = create_ox_offspring(parent1.signed_libraries, parent2.signed_libraries)
        offspring2_libs = create_ox_offspring(parent2.signed_libraries, parent1.signed_libraries)
        
        # Build complete solutions
        offspring1 = AdvancedCrossover._build_solution_from_order(offspring1_libs, instance_data)
        offspring2 = AdvancedCrossover._build_solution_from_order(offspring2_libs, instance_data)
        
        return offspring1, offspring2
    
    @staticmethod
    def partially_mapped_crossover(parent1: Solution, parent2: Solution, instance_data) -> Tuple[Solution, Solution]:
        """
        Partially Mapped Crossover (PMX) - maintains permutation validity
        Good for problems where order matters significantly
        """
        if not parent1.signed_libraries or not parent2.signed_libraries:
            return parent1.shallow_copy(), parent2.shallow_copy()
        
        def create_pmx_offspring(p1_libs: List[int], p2_libs: List[int]) -> List[int]:
            size = len(p1_libs)
            if size <= 2:
                return p1_libs[:]
            
            # Select crossover points
            point1 = random.randint(0, size - 2)
            point2 = random.randint(point1 + 1, size - 1)
            
            # Initialize offspring with parent1
            offspring = p1_libs[:]
            
            # Create mapping from crossover segment
            mapping = {}
            for i in range(point1, point2 + 1):
                if p1_libs[i] != p2_libs[i]:
                    mapping[p1_libs[i]] = p2_libs[i]
            
            # Apply mapping outside crossover segment
            for i in range(size):
                if i < point1 or i > point2:
                    current = offspring[i]
                    # Follow mapping chain
                    while current in mapping:
                        current = mapping[current]
                    offspring[i] = current
            
            # Copy crossover segment from parent2
            for i in range(point1, point2 + 1):
                offspring[i] = p2_libs[i]
            
            return offspring
        
        # Create offspring using PMX
        offspring1_libs = create_pmx_offspring(parent1.signed_libraries, parent2.signed_libraries)
        offspring2_libs = create_pmx_offspring(parent2.signed_libraries, parent1.signed_libraries)
        
        # Build complete solutions
        offspring1 = AdvancedCrossover._build_solution_from_order(offspring1_libs, instance_data)
        offspring2 = AdvancedCrossover._build_solution_from_order(offspring2_libs, instance_data)
        
        return offspring1, offspring2
    
    @staticmethod
    def cycle_crossover(parent1: Solution, parent2: Solution, instance_data) -> Tuple[Solution, Solution]:
        """
        Cycle Crossover (CX) - preserves absolute positions from both parents
        """
        if not parent1.signed_libraries or not parent2.signed_libraries:
            return parent1.shallow_copy(), parent2.shallow_copy()
        
        def create_cx_offspring(p1_libs: List[int], p2_libs: List[int]) -> List[int]:
            size = len(p1_libs)
            offspring = [None] * size
            visited = [False] * size
            
            # Alternate between parents for each cycle
            use_parent1 = True
            
            for start in range(size):
                if visited[start]:
                    continue
                
                # Find cycle starting at position 'start'
                current_pos = start
                cycle_positions = []
                
                while not visited[current_pos]:
                    visited[current_pos] = True
                    cycle_positions.append(current_pos)
                    
                    # Find where p2_libs[current_pos] appears in p1_libs
                    target_lib = p2_libs[current_pos]
                    try:
                        current_pos = p1_libs.index(target_lib)
                    except ValueError:
                        break  # Handle case where library doesn't exist
                
                # Assign libraries to cycle positions
                for pos in cycle_positions:
                    if use_parent1:
                        offspring[pos] = p1_libs[pos]
                    else:
                        offspring[pos] = p2_libs[pos]
                
                use_parent1 = not use_parent1
            
            # Handle any None values
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = p1_libs[i]
            
            return offspring
        
        # Create offspring using CX
        offspring1_libs = create_cx_offspring(parent1.signed_libraries, parent2.signed_libraries)
        offspring2_libs = create_cx_offspring(parent2.signed_libraries, parent1.signed_libraries)
        
        # Build complete solutions
        offspring1 = AdvancedCrossover._build_solution_from_order(offspring1_libs, instance_data)
        offspring2 = AdvancedCrossover._build_solution_from_order(offspring2_libs, instance_data)
        
        return offspring1, offspring2
    
    @staticmethod
    def position_based_crossover(parent1: Solution, parent2: Solution, instance_data) -> Tuple[Solution, Solution]:
        """
        Position-Based Crossover - randomly selects positions from one parent,
        fills remaining with other parent's order
        """
        if not parent1.signed_libraries or not parent2.signed_libraries:
            return parent1.shallow_copy(), parent2.shallow_copy()
        
        def create_pbx_offspring(p1_libs: List[int], p2_libs: List[int]) -> List[int]:
            size = len(p1_libs)
            offspring = [None] * size
            
            # Randomly select positions to inherit from parent1
            num_positions = random.randint(1, size - 1)
            selected_positions = random.sample(range(size), num_positions)
            
            # Copy selected positions from parent1
            for pos in selected_positions:
                offspring[pos] = p1_libs[pos]
            
            # Fill remaining positions with parent2's order
            p2_remaining = [lib for lib in p2_libs if lib not in offspring]
            p2_idx = 0
            
            for i in range(size):
                if offspring[i] is None and p2_idx < len(p2_remaining):
                    offspring[i] = p2_remaining[p2_idx]
                    p2_idx += 1
            
            # Handle any remaining None values
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = p1_libs[i]
            
            return offspring
        
        # Create offspring using PBX
        offspring1_libs = create_pbx_offspring(parent1.signed_libraries, parent2.signed_libraries)
        offspring2_libs = create_pbx_offspring(parent2.signed_libraries, parent1.signed_libraries)
        
        # Build complete solutions
        offspring1 = AdvancedCrossover._build_solution_from_order(offspring1_libs, instance_data)
        offspring2 = AdvancedCrossover._build_solution_from_order(offspring2_libs, instance_data)
        
        return offspring1, offspring2
    
    @staticmethod
    def _build_solution_from_order(library_order: List[int], instance_data) -> Solution:
        """Build a complete solution from a library ordering"""
        scanned_books = set()
        scanned_per_lib = {}
        used_libs = []
        
        current_day = 0
        for lib_id in library_order:
            if lib_id >= len(instance_data.libs):
                continue  # Skip invalid library IDs
            
            lib_data = instance_data.libs[lib_id]
            
            # Check if library can be signed up in time
            if current_day + lib_data.signup_days >= instance_data.num_days:
                continue
            
            current_day += lib_data.signup_days
            remaining_days = instance_data.num_days - current_day
            max_books = remaining_days * lib_data.books_per_day
            
            # Select books to scan
            available_books = [b.id for b in lib_data.books if b.id not in scanned_books]
            available_books.sort(key=lambda x: instance_data.scores[x], reverse=True)
            selected = available_books[:max_books]
            
            if selected:
                scanned_books.update(selected)
                scanned_per_lib[lib_id] = selected
                used_libs.append(lib_id)
        
        # Create solution
        all_lib_ids = set(range(instance_data.num_libs))
        unsigned_libs = list(all_lib_ids - set(used_libs))
        
        solution = Solution(
            signed_libs=used_libs,
            unsigned_libs=unsigned_libs,
            scanned_books_per_library=scanned_per_lib,
            scanned_books=scanned_books
        )
        
        solution.calculate_fitness_score(instance_data.scores)
        return solution
    
    @staticmethod
    def choose_crossover_method(crossover_types: List[str], weights: List[float]):
        """Choose a crossover method based on configuration"""
        method_map = {
            "order_preserving": AdvancedCrossover.order_crossover,
            "order_crossover": AdvancedCrossover.order_crossover,
            "partially_mapped": AdvancedCrossover.partially_mapped_crossover,
            "cycle_crossover": AdvancedCrossover.cycle_crossover,
            "position_based": AdvancedCrossover.position_based_crossover
        }
        
        # Select method based on weights
        selected_type = random.choices(crossover_types, weights=weights, k=1)[0]
        return method_map.get(selected_type, AdvancedCrossover.order_crossover)
