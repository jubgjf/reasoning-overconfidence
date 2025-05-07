import random
import json
from collections import defaultdict
from confidence.data import SubsetSumData


class SubsetSumProblemGenerator:
    def __init__(self, max_n, max_abs_val):
        """
        Initializes the problem generator.

        Args:
            max_n (int): Maximum size of the integer list.
            max_abs_val (int): Maximum absolute value of integers in the list.
        """
        self.max_n = max_n
        self.max_abs_val = max_abs_val

    def generate_problem(self, n, min_val, max_val):
        """
        Generates a random Subset Sum problem instance with unique elements.
        Guarantees at least one solution exists by picking a random subset for the target.

        Args:
            n (int): Size of the integer list.
            min_val (int): Minimum value for integers in the list (inclusive).
            max_val (int): Maximum value for integers in the list (inclusive).

        Returns:
            dict: A dictionary representing the problem {"nums": [...], "target": ...},
                  or None if parameters are invalid or cannot generate enough unique numbers.
        """
        if not (
            1 <= n <= self.max_n
            and abs(min_val) <= self.max_abs_val
            and abs(max_val) <= self.max_abs_val
            and min_val <= max_val
        ):
            print("Warning: Requested parameters exceed maximum limits or are invalid.")
            return None

        # Ensure enough unique integers can be generated in the range
        if (max_val - min_val + 1) < n:
            print(f"Warning: Range [{min_val}, {max_val}] does not contain enough unique numbers for n={n}.")
            return None

        # Generate n unique integers within the range
        nums = random.sample(range(min_val, max_val + 1), n)
        random.shuffle(nums)  # Shuffle to avoid any implicit order bias

        # Generate a target sum by picking a random non-empty subset
        # This guarantees at least one solution exists
        subset_size = random.randint(1, n)
        target_subset = random.sample(nums, subset_size)
        target = sum(target_subset)

        # Optional: Occasionally generate a target that might lead to 0 solutions or many solutions
        # based on a random number NOT from a subset, but within a plausible range.
        # This is more complex if you specifically need problems with 0 solutions.
        # For this code, we guarantee at least one solution. If you need 0-solution problems,
        # generate a random target independently and keep problems where solve_subset_sum returns [].

        return {"nums": nums, "target": target}

    def format_problem_text(self, problem):
        """Formats the problem dictionary into a human-readable text description."""
        if problem is None:
            return "Invalid Problem"

        nums_str = ", ".join(map(str, problem["nums"]))
        text = f"Given the set of unique integers: {{{nums_str}}}\n"
        text += f"Find all subsets that sum exactly to the target: {problem['target']}"

        return text


class SubsetSumSolver:
    def __init__(self):
        self.solutions = set()  # Use a set to store unique solutions (as tuples)
        self.nums = []
        self.target = 0

    def solve(self, problem):
        """
        Finds all unique subsets for a given Subset Sum problem using backtracking.

        Args:
            problem (dict): The problem dictionary {"nums": [...], "target": ...}.

        Returns:
            list: A list of solution lists. Returns an empty list if no solutions found.
                  Solutions are returned as lists of integers, sorted for consistency.
        """
        self.problem = problem
        if self.problem is None:
            return []

        self.nums = sorted(problem["nums"])  # Sorting helps with pruning and canonical representation
        self.target = problem["target"]
        self.solutions = set()  # Reset solutions set

        self._backtrack(0, [], 0)  # Start backtracking from index 0

        # Convert solutions from set of tuples to list of sorted lists
        sorted_solutions = [sorted(list(sol)) for sol in self.solutions]
        # Sort the list of solutions for consistent output order
        sorted_solutions.sort()

        return sorted_solutions

    def _backtrack(self, index, current_subset, current_sum):
        """Recursive backtracking function."""

        # Pruning 1: If current sum exceeds target (only works reliably for non-negative numbers, or if numbers sorted)
        # Since we generate targets from sums of subsets, and usually non-negative numbers, this is safe.
        # If allowing negative numbers in generation, remove this pruning.
        if current_sum > self.target:
            return

        # Base case 1: If target sum is reached
        # It's important to check this *after* a potential recursive call includes the last element
        # or before returning if the last element wasn't included.
        # A common pattern is to check when reaching the end of the numbers list.

        # Base case 2: If all numbers have been considered
        if index == len(self.nums):
            if current_sum == self.target:
                # Found a solution. Store as a tuple to handle uniqueness.
                self.solutions.add(tuple(sorted(current_subset)))  # Sort subset before converting to tuple
            return

        # --- Recursive Steps ---

        # 1. Include self.nums[index]
        self._backtrack(index + 1, current_subset + [self.nums[index]], current_sum + self.nums[index])

        # 2. Exclude self.nums[index]
        # To handle duplicate *solutions* (which shouldn't happen here since input nums are unique,
        # but is relevant if input nums could be duplicates or if solution order matters),
        # one might skip duplicate elements here. But with unique input numbers, skipping
        # is not needed for finding all unique subsets of elements.
        self._backtrack(index + 1, current_subset, current_sum)

    def format_solutions_text(self, solutions):
        """Formats a list of solution lists into human-readable text."""
        if not solutions:
            return "No feasible solutions found."

        text = "Feasible Subsets:\n"
        for i, solution in enumerate(solutions):
            text += f"Solution {i + 1}: {{{', '.join(map(str, solution))}}}\n"
        return text.strip()


# --- Example Usage and Data Generation Loop ---

if __name__ == "__main__":
    # Define parameters for problem generation
    # Choose n such that enumerating 2^n subsets is feasible
    MAX_N = 15  # 2^15 = 32768. Max subsets to explore.
    MAX_ABS_VAL = 200  # Range of numbers

    generator = SubsetSumProblemGenerator(MAX_N, MAX_ABS_VAL)
    solver = SubsetSumSolver()

    # Target number of problems per difficulty level
    TARGET_PROBLEMS_PER_LEVEL = 100
    # Define difficulty levels by number of solutions
    # Adjust these ranges based on your observed distribution of solution counts for subset sum
    DIFFICULTY_LEVELS = [
        (1, 50),
        (51, 100),
        (101, 150),
        (151, 200),
        (201, 250),
        (251, 300),
        (301, float("inf")),
    ]
    NUM_DIFFICULTY_LEVELS = len(DIFFICULTY_LEVELS)
    TOTAL_TARGET_PROBLEMS = TARGET_PROBLEMS_PER_LEVEL * NUM_DIFFICULTY_LEVELS

    # Parameters for generated problem instances (adjust these to get desired solution distribution)
    # Smaller n, smaller range of values, or target sum closer to the middle of possible sums might yield more solutions.
    MIN_N = 10
    MIN_VAL = 1  # Start with non-negative for simpler pruning
    MAX_VAL = 50

    problems_by_difficulty = defaultdict(list)
    generated_count = 0
    attempts = 0  # Track attempts, as some parameter combinations might not easily yield all difficulty levels

    print(
        f"Generating and solving Subset Sum problems until {TOTAL_TARGET_PROBLEMS} feasible problems are found across {NUM_DIFFICULTY_LEVELS} difficulty levels..."
    )

    # Loop until enough problems are collected for each difficulty level
    while generated_count < TOTAL_TARGET_PROBLEMS and attempts < TOTAL_TARGET_PROBLEMS * 100:  # Add a limit to attempts
        attempts += 1
        # Randomly select parameters for the current problem instance
        num_n = random.randint(MIN_N, MAX_N)
        min_v = random.randint(MIN_VAL, MAX_VAL // 2)  # Try to keep numbers somewhat centered
        max_v = random.randint(MAX_VAL // 2 + 1, MAX_VAL)

        # Ensure max_v - min_v + 1 >= num_n
        if (max_v - min_v + 1) < num_n:
            continue  # Skip if range is too small for number of elements

        problem = generator.generate_problem(num_n, min_v, max_v)

        if problem:
            solutions = solver.solve(problem)
            num_solutions = len(solutions)

            # All generated problems from this generator will have >= 1 solution
            # since the target is derived from a subset sum.

            # Determine difficulty level based on number of solutions
            difficulty_level = -1
            for level_idx, (min_sol, max_sol) in enumerate(DIFFICULTY_LEVELS):
                if min_sol <= num_solutions <= max_sol:
                    difficulty_level = level_idx
                    break

            if difficulty_level != -1:
                if len(problems_by_difficulty[difficulty_level]) < TARGET_PROBLEMS_PER_LEVEL:
                    # Store problem text and num_solutions
                    problem_text = generator.format_problem_text(problem)

                    problems_by_difficulty[difficulty_level].append(
                        {
                            "problem_params": (
                                num_n,
                                min_v,
                                max_v,
                                problem["target"],
                            ),  # Store params for potential regeneration/debugging
                            "problem_text": problem_text,
                            "num_solutions": num_solutions,
                            # "solutions_data": solutions # Warning: can be large! Store only if needed.
                        }
                    )
                    generated_count += 1
                    print(
                        f"Generated problem (n={num_n}, target={problem['target']}) with {num_solutions} solutions (Level {difficulty_level + 1}). Total generated: {generated_count}/{TOTAL_TARGET_PROBLEMS}"
                    )
                # else:
                # print(f"Skipping problem with {num_solutions} solutions (Level {difficulty_level+1}) as target for this level reached.")
            # else:
            # print(f"Problem with {num_solutions} solutions did not fit into defined difficulty levels.")

    print("\n--- Generation Complete ---")
    print("Summary of problems generated per difficulty level:")
    for level_idx, count in sorted([(k, len(v)) for k, v in problems_by_difficulty.items()]):
        sol_range = DIFFICULTY_LEVELS[level_idx]
        print(
            f"Level {level_idx + 1} ({sol_range[0]} - {sol_range[1]}{'+' if sol_range[1] == float('inf') else ''} solutions): {count} problems"
        )

    if generated_count < TOTAL_TARGET_PROBLEMS:
        print(
            f"\nWarning: Could only generate {generated_count} problems after {attempts} attempts. Adjusting generation parameters or difficulty levels might be needed."
        )

    # You would typically save problems_by_difficulty to a file (e.g., JSON, pickle)
    # for later use in your LLM experiments.
    with open("./dataset/subsetsum.jsonl", "w") as f:
        question_id = 0
        for level, problems_list in problems_by_difficulty.items():
            for problem_data in problems_list:
                data = SubsetSumData(
                    question_id=question_id,
                    question=problem_data["problem_text"],
                    answer_count=problem_data["num_solutions"],
                )
                question_id += 1
                f.write(json.dumps(data.model_dump(), ensure_ascii=False) + "\n")
    print("\nDataset structure saved (example).")
