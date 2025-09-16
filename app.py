import sys
import os
import json
import time
from models import Parser, ConfigLoader
from models.initial_solution import InitialSolution
from models.enhanced_genetic_solver import EnhancedGeneticSolver
from validator.validator import validate_solution


def run_enhanced_instances(output_dir='enhanced_output', config_dir='config'):
    """Run instances using the enhanced genetic algorithm with JSON configuration"""
    print(f"Running Enhanced GA - Output directory: {output_dir}")
    print("=" * 60)
    
    # Get all .txt files and sort by instance size (small -> medium -> large)
    all_files = [f for f in os.listdir('input') if f.endswith('.txt')]
    
    def get_file_size_category(filename):
        """Determine size category for sorting"""
        try:
            # Save current library counter
            from models.library import Library
            original_counter = Library._id_counter
            
            parser = Parser(f'./input/{filename}')
            instance = parser.parse()
            
            # Restore library counter to avoid ID conflicts
            Library._id_counter = original_counter
            
            if instance.num_books <= 10000 and instance.num_libs <= 100:
                return (0, instance.num_books)  # Small - sort by num_books
            elif instance.num_books <= 200000:
                return (1, instance.num_books)  # Medium - sort by num_books  
            else:
                return (2, instance.num_books)  # Large - sort by num_books
        except:
            return (3, 0)  # Error cases last
    
    # Sort files by size category, then by number of books within category
    directory = sorted(all_files, key=get_file_size_category)
    
    print(f"📋 Processing order (Small → Medium → Large):")
    for i, file in enumerate(directory, 1):
        print(f"  {i:2d}. {file}")
    print()
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Create statistics directory
    stats_dir = os.path.join(output_dir, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    for file_index, file in enumerate(directory, 1):
        if file.endswith('.txt'):
            print(f'\n🔄 Processing: {file}')
            print("-" * 50)
            
            # Parse instance
            parser = Parser(f'./input/{file}')
            instance = parser.parse()
            
            # Get appropriate configuration
            try:
                config = ConfigLoader.get_config_for_instance(instance)
                print(f"📋 Using {config.instance_type} configuration")
                print(f"   Population: {config.population_size}, Time: {config.time_limit_sec}s")
            except Exception as e:
                print(f"⚠️  Warning: Could not load config, using defaults: {e}")
                from models.config_loader import GAConfig
                config = GAConfig({})  # Use defaults
            
            # Generate initial solution
            print("🏗️  Generating initial solution...")
            initial_solution = InitialSolution.generate_initial_solution(instance)
            print(f"   Initial score: {initial_solution.fitness_score:,}")
            
            # Run enhanced genetic solver
            print("🧬 Running Enhanced Genetic Algorithm...")
            enhanced_solver = EnhancedGeneticSolver(
                initial_solution=initial_solution,
                instance=instance,
                config=config
            )
            
            solution = enhanced_solver.solve()
            score = solution.fitness_score
            
            # Get solver statistics
            stats = enhanced_solver.get_statistics()
            
            # Save solution
            output_file = os.path.join(output_dir, file)
            solution.export(output_file)
            
            # Save statistics
            stats_file = os.path.join(stats_dir, f"{file.replace('.txt', '_stats.json')}")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Validate solution
            print(f"🔍 Validating solution...")
            validation_result = validate_solution(f'./input/{file}', output_file, isConsoleApplication=True)
            
            # Record results
            improvement = ((score - initial_solution.fitness_score) / 
                          max(1, initial_solution.fitness_score)) * 100
            results.append((file, initial_solution.fitness_score, score, improvement, validation_result))
            
            print(f"✅ Completed: {file}")
            print(f"   Initial:  {initial_solution.fitness_score:,}")
            print(f"   Final:    {score:,}")
            print(f"   Improvement: {improvement:.2f}%")
            print(f"   Generations: {stats['generations_completed']}")
            print(f"   Final Mutation Rate: {stats['final_mutation_prob']:.3f}")
            
            # Update and display real-time summary after each file
            print(f"\n📊 UPDATING RESULTS SUMMARY ({file_index}/{len(directory)})...")
            _update_realtime_summary(output_dir, results)
            
            # Display current summary
            summary_file = os.path.join(output_dir, 'results_summary.txt')
            if os.path.exists(summary_file):
                print(f"\n📋 CURRENT RESULTS SUMMARY:")
                print("=" * 50)
                with open(summary_file, 'r') as f:
                    print(f.read())
                print("=" * 50)
    
    print("\n" + "=" * 60)
    print("🔍 Final validation complete - individual validations done per file")
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 ENHANCED GA RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Instance':<25} {'Initial':>12} {'Final':>12} {'Improvement':>12} {'Valid':>6}")
    print("-" * 66)
    
    total_improvement = 0
    valid_improvements = 0
    
    for result in results:
        if len(result) >= 5:
            file, initial_score, final_score, improvement, validation = result
        else:
            file, initial_score, final_score, improvement = result
            validation = "Unknown"
        
        valid_mark = "✅" if validation == 'Valid' else "❌"
        print(f"{file:<25} {initial_score:>12,} {final_score:>12,} {improvement:>11.2f}% {valid_mark}")
        if improvement > 0:
            total_improvement += improvement
            valid_improvements += 1
    
    print("-" * 66)
    if valid_improvements > 0:
        avg_improvement = total_improvement / valid_improvements
        print(f"Average improvement: {avg_improvement:.2f}% ({valid_improvements}/{len(results)} instances)")
    
    # Write detailed summary
    summary_file = os.path.join(output_dir, 'enhanced_results_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Enhanced Genetic Algorithm Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Instance':<25} {'Initial':>12} {'Final':>12} {'Improvement':>12}\n")
        f.write("-" * 62 + "\n")
        
        for result in results:
            if len(result) >= 5:
                file, initial_score, final_score, improvement, validation = result
                valid_mark = "✅" if validation == 'Valid' else "❌"
            else:
                file, initial_score, final_score, improvement = result
                valid_mark = "?"
            f.write(f"{file:<25} {initial_score:>12,} {final_score:>12,} {improvement:>11.2f}% {valid_mark}\n")
        
        f.write("-" * 62 + "\n")
        if valid_improvements > 0:
            f.write(f"Average improvement: {avg_improvement:.2f}% ({valid_improvements}/{len(results)} instances)\n")
        
        f.write(f"\nTotal instances processed: {len(results)}\n")
        f.write(f"Statistics saved in: {stats_dir}\n")
    
    print(f"\n📁 Detailed summary saved to: {summary_file}")
    print(f"📊 Individual statistics saved to: {stats_dir}")


def run_comparison(original_output_dir='output', enhanced_output_dir='enhanced_output'):
    """Compare original vs enhanced results"""
    print("\n" + "=" * 60)
    print("🔄 COMPARISON: Original vs Enhanced GA")
    print("=" * 60)
    
    # Read original results if they exist
    original_summary = os.path.join(original_output_dir, 'summary_results.txt')
    enhanced_summary = os.path.join(enhanced_output_dir, 'enhanced_results_summary.txt')
    
    if not os.path.exists(original_summary):
        print("⚠️  Original results not found. Run the original solver first.")
        return
    
    if not os.path.exists(enhanced_summary):
        print("⚠️  Enhanced results not found. Run the enhanced solver first.")
        return
    
    print("📊 Comparison will be implemented in future version")
    print(f"Original results: {original_summary}")
    print(f"Enhanced results: {enhanced_summary}")


def _update_realtime_summary(output_dir, results):
    """Update results summary file after each completed file"""
    summary_file = os.path.join(output_dir, 'results_summary.txt')
    
    if not results:
        return
    
    # Calculate statistics
    total_runs = len(results)
    improvements = [improvement for _, _, _, improvement, _ in results]
    valid_improvements = [imp for imp in improvements if imp > 0]
    valid_solutions = [result for result in results if len(result) > 4 and result[4] == 'Valid']
    
    avg_improvement = sum(valid_improvements) / len(valid_improvements) if valid_improvements else 0
    max_improvement = max(improvements) if improvements else 0
    min_improvement = min(improvements) if improvements else 0
    
    # Write real-time summary
    with open(summary_file, 'w') as f:
        f.write("🧬 ENHANCED GENETIC ALGORITHM - REAL-TIME RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Completed runs: {total_runs}\n")
        f.write(f"Valid solutions: {len(valid_solutions)}/{total_runs}\n\n")
        
        f.write("📊 PERFORMANCE STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average improvement: {avg_improvement:.2f}%\n")
        f.write(f"Best improvement: {max_improvement:.2f}%\n")
        f.write(f"Worst improvement: {min_improvement:.2f}%\n")
        f.write(f"Successful improvements: {len(valid_improvements)}/{total_runs}\n\n")
        
        f.write("📋 DETAILED RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'File':<25} {'Initial':>10} {'Final':>10} {'Improve':>8} {'Valid':>6}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            if len(result) >= 5:
                file, initial_score, final_score, improvement, validation = result
                valid_mark = "✅" if validation == 'Valid' else "❌"
            else:
                file, initial_score, final_score, improvement = result
                valid_mark = "?"
            
            f.write(f"{file:<25} {initial_score:>10,} {final_score:>10,} {improvement:>7.1f}% {valid_mark:>6}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"\n✅ {total_runs} files completed so far...\n")


def _update_cumulative_results(output_dir, result_data):
    """Update cumulative results file with new run data"""
    import csv
    from datetime import datetime
    
    results_file = os.path.join(output_dir, 'cumulative_results.csv')
    results_summary = os.path.join(output_dir, 'results_summary.txt')
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(results_file)
    
    # Append to CSV file
    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'input_file', 'instance_size', 'num_books', 'num_libs', 'num_days',
            'initial_score', 'final_score', 'improvement_percent', 'total_time_seconds',
            'generations', 'final_mutation_rate', 'final_immigration_rate', 'used_steady_state',
            'validation_status', 'output_file', 'stats_file'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result_data)
    
    # Update summary text file
    _update_summary_file(results_summary, results_file)
    
    print(f"📊 Updated cumulative results: {results_file}")
    print(f"📋 Updated summary: {results_summary}")


def _update_summary_file(summary_file, csv_file):
    """Update human-readable summary file"""
    import csv
    
    try:
        # Read all results
        results = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(row)
        
        if not results:
            return
        
        # Calculate statistics
        total_runs = len(results)
        improvements = [float(r['improvement_percent']) for r in results if r['improvement_percent']]
        valid_improvements = [imp for imp in improvements if imp > 0]
        
        avg_improvement = sum(valid_improvements) / len(valid_improvements) if valid_improvements else 0
        max_improvement = max(improvements) if improvements else 0
        min_improvement = min(improvements) if improvements else 0
        
        total_time = sum(float(r['total_time_seconds']) for r in results if r['total_time_seconds'])
        avg_time = total_time / total_runs if total_runs > 0 else 0
        
        # Write summary
        with open(summary_file, 'w') as f:
            f.write("🧬 ENHANCED GENETIC ALGORITHM - CUMULATIVE RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {results[-1]['timestamp']}\n")
            f.write(f"Total runs: {total_runs}\n\n")
            
            f.write("📊 PERFORMANCE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average improvement: {avg_improvement:.2f}%\n")
            f.write(f"Best improvement: {max_improvement:.2f}%\n")
            f.write(f"Worst improvement: {min_improvement:.2f}%\n")
            f.write(f"Successful improvements: {len(valid_improvements)}/{total_runs}\n")
            f.write(f"Average time per run: {avg_time:.2f}s\n")
            f.write(f"Total computation time: {total_time:.2f}s\n\n")
            
            f.write("📋 DETAILED RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'File':<25} {'Initial':>10} {'Final':>10} {'Improve':>8} {'Time':>6} {'Valid':>6}\n")
            f.write("-" * 70 + "\n")
            
            for result in results:
                file_name = result['input_file'][:24]  # Truncate long names
                initial = int(float(result['initial_score'])) if result['initial_score'] else 0
                final = int(float(result['final_score'])) if result['final_score'] else 0
                improve = float(result['improvement_percent']) if result['improvement_percent'] else 0
                time_sec = float(result['total_time_seconds']) if result['total_time_seconds'] else 0
                valid = "✅" if result['validation_status'] == 'Valid' else "❌"
                
                f.write(f"{file_name:<25} {initial:>10,} {final:>10,} {improve:>7.1f}% {time_sec:>5.0f}s {valid:>6}\n")
            
            f.write("-" * 70 + "\n")
            f.write(f"\n💾 Detailed data available in: {os.path.basename(csv_file)}\n")
            f.write(f"📊 Individual statistics in: statistics/ directory\n")
    
    except Exception as e:
        print(f"Warning: Could not update summary file: {e}")


def create_custom_config_example():
    """Create an example of custom configuration"""
    custom_config = {
        "instance_type": "custom_example",
        "description": "Example custom configuration with aggressive parameters",
        "population_size": 300,
        "time_limit_sec": 1200,
        "mutation": {
            "initial_prob": 0.8,
            "final_prob": 0.2,
            "adaptation_type": "fitness_based"
        },
        "crossover": {
            "rate": 0.9,
            "types": ["order_crossover", "partially_mapped"],
            "weights": [0.7, 0.3]
        },
        "diversity": {
            "enabled": True,
            "min_threshold": 0.2,
            "trigger_immigration": True
        }
    }
    
    os.makedirs('config', exist_ok=True)
    custom_path = 'config/ga_config_custom.json'
    
    with open(custom_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    print(f"📝 Custom configuration example created: {custom_path}")


def run_single_instance(input_file_path, config_type=None, output_dir='enhanced_output'):
    """Run enhanced GA on a single instance file"""
    import time
    
    # Validate input file
    if not os.path.exists(input_file_path):
        print(f"❌ Error: Input file not found: {input_file_path}")
        return False
    
    file_name = os.path.basename(input_file_path)
    print(f"🧪 RUNNING ENHANCED GA ON SINGLE INSTANCE")
    print("=" * 60)
    print(f"📁 File: {file_name}")
    print(f"📂 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    try:
        # Parse instance
        print("\n🔍 Parsing instance...")
        parser = Parser(input_file_path)
        instance = parser.parse()
        
        print(f"   📊 Books: {instance.num_books:,}")
        print(f"   📚 Libraries: {instance.num_libs:,}")
        print(f"   📅 Days: {instance.num_days:,}")
        
        # Get appropriate configuration
        try:
            if config_type and config_type.endswith('.json'):
                # Custom config file path provided
                config = ConfigLoader.load_config(config_type)
                print(f"   🎯 Using custom configuration: {config.instance_type}")
            else:
                # Use auto-detection or specified type
                if config_type in ['small', 'medium', 'large']:
                    config = ConfigLoader.load_config(f'config/ga_config_{config_type}.json')
                    print(f"   🎯 Using {config_type} configuration")
                else:
                    config = ConfigLoader.get_config_for_instance(instance)
                    print(f"   🎯 Using auto-detected {config.instance_type} configuration")
            
            print(f"   👥 Population: {config.population_size}, Time: {config.time_limit_sec}s")
            print(f"   🧬 Diversity enabled: {config.diversity_enabled}")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load config, using defaults: {e}")
            from models.config_loader import GAConfig
            config = GAConfig({})
        
        # Generate initial solution
        print("\n🏗️  Generating initial solution...")
        start_time = time.time()
        initial_solution = InitialSolution.generate_initial_solution(instance)
        init_time = time.time() - start_time
        print(f"   ✅ Generated in {init_time:.2f}s, Score: {initial_solution.fitness_score:,}")
        
        # Run enhanced genetic solver
        print(f"\n🧬 Running Enhanced Genetic Algorithm...")
        solver_start = time.time()
        enhanced_solver = EnhancedGeneticSolver(
            initial_solution=initial_solution,
            instance=instance,
            config=config
        )
        
        final_solution = enhanced_solver.solve()
        solver_time = time.time() - solver_start
        
        # Get solver statistics
        stats = enhanced_solver.get_statistics()
        
        # Calculate improvement
        improvement = ((final_solution.fitness_score - initial_solution.fitness_score) / 
                      max(1, initial_solution.fitness_score)) * 100
        
        # Save solution and statistics
        output_file = os.path.join(output_dir, file_name)
        final_solution.export(output_file)
        
        stats_file = os.path.join(stats_dir, f"{file_name.replace('.txt', '_stats.json')}")
        stats.update({
            'input_file': file_name,
            'initial_score': initial_solution.fitness_score,
            'final_score': final_solution.fitness_score,
            'improvement_percent': improvement,
            'total_time_seconds': solver_time
        })
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Validate solution
        print(f"\n🔍 Validating solution...")
        validation_result = validate_solution(input_file_path, output_file, isConsoleApplication=True)
        
        # Update cumulative results file
        _update_cumulative_results(output_dir, {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_file': file_name,
            'instance_size': config.instance_type,
            'num_books': instance.num_books,
            'num_libs': instance.num_libs,
            'num_days': instance.num_days,
            'initial_score': initial_solution.fitness_score,
            'final_score': final_solution.fitness_score,
            'improvement_percent': improvement,
            'total_time_seconds': solver_time,
            'generations': stats['generations_completed'],
            'final_mutation_rate': stats['final_mutation_prob'],
            'final_immigration_rate': stats['final_immigrant_frac'],
            'used_steady_state': stats['used_steady_state'],
            'validation_status': validation_result,
            'output_file': output_file,
            'stats_file': stats_file
        })
        
        # Print results
        print(f"\n✅ RESULTS SUMMARY")
        print("=" * 40)
        print(f"📁 File: {file_name}")
        print(f"🎯 Initial Score: {initial_solution.fitness_score:,}")
        print(f"🚀 Final Score: {final_solution.fitness_score:,}")
        print(f"📈 Improvement: {improvement:.2f}%")
        print(f"⏱️  Time: {solver_time:.2f}s")
        print(f"🔄 Generations: {stats['generations_completed']}")
        print(f"✅ Valid: {validation_result == 'Valid'}")
        print(f"💾 Output: {output_file}")
        print(f"📊 Stats: {stats_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function with intelligent file detection"""
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        
        # Check if first argument is a file path
        if os.path.isfile(first_arg) and first_arg.endswith('.txt'):
            # Single file mode
            input_file = first_arg
            config_type = sys.argv[2] if len(sys.argv) > 2 else None
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'enhanced_output'
            
            print("🎯 Detected single file input - running on single instance")
            success = run_single_instance(input_file, config_type, output_dir)
            
            if success:
                print(f"\n🎉 Single instance test completed successfully!")
            else:
                print(f"\n💥 Single instance test failed!")
                sys.exit(1)
        
        elif first_arg == "all":
            # Run on all instances
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "enhanced_output"
            print("🎯 Running on all instances in input directory")
            run_enhanced_instances(output_dir)
        
        elif first_arg == "compare":
            # Compare original vs enhanced
            original_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
            enhanced_dir = sys.argv[3] if len(sys.argv) > 3 else "enhanced_output"
            run_comparison(original_dir, enhanced_dir)
        
        elif first_arg == "create_config":
            # Create example custom config
            create_custom_config_example()
        
        else:
            print("❌ Invalid command or file not found")
            print("\nUsage:")
            print("  python3 app.py <input_file.txt> [config] [output_dir]  - Run on single file")
            print("  python3 app.py all [output_dir]                       - Run on all files")
            print("  python3 app.py compare [orig_dir] [enh_dir]          - Compare results")
            print("  python3 app.py create_config                         - Create config example")
    
    else:
        print("🧬 Enhanced Genetic Algorithm for Book Scanning")
        print("=" * 50)
        print("📖 INTELLIGENT FILE DETECTION:")
        print("   • Provide a .txt file path → runs on single instance")
        print("   • Use 'all' command → runs on all instances")
        print("")
        print("📋 USAGE EXAMPLES:")
        print("   python3 app.py input/B50_L5_D4.txt")
        print("   python3 app.py input/c_incunabula.txt small")
        print("   python3 app.py input/B1000k_L115_D230.txt large my_output")
        print("   python3 app.py all enhanced_results")
        print("   python3 app.py compare output enhanced_results")
        print("")
        print("🎯 Just provide the file path and it will automatically run on that single instance!")


if __name__ == "__main__":
    main()
