## Tuning strategy for the Genetic Book Scanning solver

This guide gives practical, high‑impact settings and steps to tune the GA in this repo for better scores under a fixed time budget.

### What we optimize
- Maximize total score of uniquely scanned books
- Keep runs time‑bounded per instance size (seconds or minutes)

### Recommended defaults (by instance size)
- Small (≤10k books, ≤100 libs)
  - population_size: 200
  - time_limit_sec: 180–300
  - mutation_prob: start 0.7 → decay to 0.3 after stagnation
  - immigrant_frac: 0.05, x2 after >10 stagnant generations
  - tweak_steps: start 7 → 3 later
  - steady_state_ratio: 0.5
  - tournament size: ~12

- Medium (≤200k books)
  - population_size: 200–300
  - time_limit_sec: 600–900
  - mutation_prob: 0.6 → 0.25
  - immigrant_frac: 0.07 (increase on stagnation)
  - tweak_steps: 5 → 2
  - steady_state_ratio: 0.5–0.6
  - tournament size: 10–12

- Large (≥200k books or many libs)
  - population_size: 100–150
  - time_limit_sec: 900–1800
  - mutation_prob: 0.6 → 0.25
  - immigrant_frac: 0.1 (increase on stagnation)
  - tweak_steps: 3 → 1
  - steady_state_ratio: 0.6
  - tournament size: ~10

Notes:
- Use the time limit to terminate; set generations high enough and let time decide.
- Keep immigrants, mutation, and tweak steps higher early; reduce after plateaus.

### Adaptive controls to lean on (already available)
- Immigrants on plateau (GeneticSolver): increase `immigrant_frac` after stagnation; reset upon improvement. Recommended: start 0.05–0.1; multiply by 1.5–2.0 after 8–12 stagnant gens (cap at 0.3).
- Generational → steady-state switch: raise `steady_state_ratio` to 0.5–0.6 so later search is steady‑state (better exploitation while preserving best).
- Local search intensity: raise `tweak_steps` early (exploration), reduce later (exploitation).

### Local search (Tweaks) guidance
- Keep a mix: swapping signed libs, signed↔unsigned, neighbor swaps, insert, last‑book swaps, and crossover rebuild.
- If you extend the solver: add adaptive operator selection (track average fitness gain per tweak and adjust `Tweaks.WEIGHTS` online).

### Crossover
- Current GA always applies crossover in offspring creation. Consider wiring `crossover_rate` to actually control crossover vs clone+mutate if you want to study its effect. For ordering problems, also try OX/PMX variants in a future improvement.

### Initial solutions
- Keep the existing combination (weighted efficiency + GRASP + sorted) and pick best.
- For further gains, bias library scoring by book rarity using `InstanceData.book_libs`: boost books that appear in fewer libraries.

### Using the existing MetaGeneticOptimizer (quick start)
Use it to pick good hyperparameters for a representative instance set, then run the full solver with those params.

```python
# scripts/tune_example.py (example usage)
from models import Parser
from models.initial_solution import InitialSolution
from models.meta_genetic_optimizer import MetaGeneticOptimizer
from models.genetic_solver import GeneticSolver

FILES = [
    'input/B50_L5_D4.txt',
    'input/c_incunabula.txt',
    'input/d_tough_choices.txt',
]

results = []
for path in FILES:
    instance = Parser(path).parse()
    init_sol = InitialSolution.generate_initial_solution(instance)
    meta = MetaGeneticOptimizer(GeneticSolver, instance, init_sol,
                                meta_pop_size=6, meta_generations=4,
                                inner_generations=30, inner_pop_size=60)
    best_h = meta.optimize()
    results.append(best_h)

# Aggregate simple recommendation (median across instances)
import statistics as st
rec = {
    'mutation_prob': st.median([r['mutation_prob'] for r in results]),
    'crossover_rate': st.median([r['crossover_rate'] for r in results]),
    'immigrant_frac': st.median([r['immigrant_frac'] for r in results]),
}
print('Recommended hyperparameters:', rec)
```

Apply recommended hyperparameters in `app.py` by setting on the solver before `solve()`:

```python
genetic_solver = GeneticSolver(initial_solution=initial_solution, instance=instance)
genetic_solver.mutation_prob = 0.5
genetic_solver.crossover_rate = 0.35  # currently not used to gate crossover frequency
genetic_solver.immigrant_frac = 0.08
genetic_solver.time_limit_sec = 600
genetic_solver.tweak_steps = 5
solution = genetic_solver.solve()
```

### Instance-size presets (copy/paste)
- Small
```python
population_size=200; time_limit_sec=300; mutation_prob=0.7; immigrant_frac=0.05; tweak_steps=7; steady_state_ratio=0.5
```
- Medium
```python
population_size=250; time_limit_sec=900; mutation_prob=0.6; immigrant_frac=0.07; tweak_steps=5; steady_state_ratio=0.55
```
- Large
```python
population_size=120; time_limit_sec=1500; mutation_prob=0.6; immigrant_frac=0.1; tweak_steps=3; steady_state_ratio=0.6
```

### Optional: islands for large instances
- Split the population into 3–5 subpopulations; migrate top individuals every N generations. This improves exploration and parallelizes easily.

### Reproducibility tips
- Set a seed at process start:
```python
import random; random.seed(42)
```
- Keep the same time budget when comparing settings; average over 3–5 runs.


Addition by gemini:
Final Recommendation
Start with Random-Restart Hill Climbing. It gives you the best "bang for your buck" and will solve the most glaring issue with the basic algorithm. For a large number of problems, this is all you will need.

If, after implementing Random-Restart, your solutions are still not good enough, then move on to Simulated Annealing. Be prepared to spend time experimenting with its parameters to get the best performance.