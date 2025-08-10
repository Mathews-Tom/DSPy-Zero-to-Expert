#!/usr/bin/env python3
"""
Exercise 4 Solution: Advanced A/B Testing System

This solution demonstrates how to build an advanced A/B testing system with
sequential testing, Bayesian analysis, and multi-armed bandit testing.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import dspy


class TestDecision(Enum):
    """Status of an A/B test."""

    CONTINUE = "continue"
    STOP_VARIANT_A_WINS = "stop_a_wins"
    STOP_VARIANT_B_WINS = "stop_b_wins"
    STOP_NO_DIFFERENCE = "stop_no_difference"


@dataclass
class BayesianResult:
    """Result of Bayesian A/B test analysis."""

    posterior_a_alpha: float
    posterior_a_beta: float
    posterior_b_alpha: float
    posterior_b_beta: float
    probability_b_better: float
    expected_loss: float


class SequentialABTest:
    """Advanced A/B testing with sequential analysis and early stopping."""

    def __init__(
        self, alpha: float = 0.05, power: float = 0.8, min_sample_size: int = 100
    ):
        self.alpha = alpha
        self.power = power
        self.min_sample_size = min_sample_size
        self.test_history = []

    def calculate_sequential_boundaries(self, max_n: int) -> Dict[str, List[float]]:
        """Calculate sequential testing boundaries (O'Brien-Fleming)."""
        # Simplified O'Brien-Fleming boundaries
        # In practice, use specialized libraries for exact calculations

        # Number of interim analyses
        k = min(5, max_n // 20)  # Up to 5 interim analyses

        # O'Brien-Fleming spending function
        boundaries = {"upper": [], "lower": [], "sample_sizes": []}

        for i in range(1, k + 1):
            # Information fraction
            t = i / k

            # O'Brien-Fleming boundary (simplified)
            z_boundary = 2.0 / math.sqrt(t)  # Simplified calculation

            boundaries["upper"].append(z_boundary)
            boundaries["lower"].append(-z_boundary)
            boundaries["sample_sizes"].append(int(max_n * t))

        return boundaries

    def sequential_test_decision(
        self, scores_a: List[float], scores_b: List[float]
    ) -> TestDecision:
        """Make sequential testing decision."""
        n_a, n_b = len(scores_a), len(scores_b)

        # Check minimum sample size
        if n_a < self.min_sample_size or n_b < self.min_sample_size:
            return TestDecision.CONTINUE

        # Calculate test statistic
        if n_a < 2 or n_b < 2:
            return TestDecision.CONTINUE

        mean_a = sum(scores_a) / len(scores_a)
        mean_b = sum(scores_b) / len(scores_b)

        # Calculate standard error
        var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)

        se = math.sqrt(var_a / n_a + var_b / n_b)

        if se == 0:
            return TestDecision.STOP_NO_DIFFERENCE

        # Z-statistic
        z_stat = (mean_b - mean_a) / se

        # Sequential boundaries (simplified)
        # In practice, use proper sequential boundaries
        current_boundary = 2.5  # Conservative boundary

        if abs(z_stat) > current_boundary:
            if z_stat > 0:
                return TestDecision.STOP_VARIANT_B_WINS
            else:
                return TestDecision.STOP_VARIANT_A_WINS

        # Check for futility (very small effect)
        if abs(z_stat) < 0.5 and n_a > self.min_sample_size * 2:
            return TestDecision.STOP_NO_DIFFERENCE

        return TestDecision.CONTINUE

    def run_sequential_test(
        self,
        variant_a: Any,
        variant_b: Any,
        test_examples: List[dspy.Example],
        metric: Any,
        max_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Run sequential A/B test with early stopping."""

        # Shuffle examples for randomization
        examples = test_examples.copy()
        random.shuffle(examples)

        scores_a = []
        scores_b = []
        decisions = []

        # Start with minimum sample size
        batch_size = max(10, self.min_sample_size // 10)

        for i in range(0, min(len(examples), max_samples), batch_size):
            batch = examples[i : i + batch_size]

            # Split batch between variants
            split_point = len(batch) // 2
            batch_a = batch[:split_point]
            batch_b = batch[split_point:]

            # Generate predictions and scores for variant A
            for example in batch_a:
                if hasattr(variant_a, "forward"):
                    inputs = example.inputs()
                    pred = variant_a(**inputs)
                else:
                    pred = variant_a(example)

                if hasattr(metric, "evaluate"):
                    result = metric.evaluate(example, pred)
                    score = result.score if hasattr(result, "score") else float(result)
                else:
                    score = float(metric(example, pred))

                scores_a.append(score)

            # Generate predictions and scores for variant B
            for example in batch_b:
                if hasattr(variant_b, "forward"):
                    inputs = example.inputs()
                    pred = variant_b(**inputs)
                else:
                    pred = variant_b(example)

                if hasattr(metric, "evaluate"):
                    result = metric.evaluate(example, pred)
                    score = result.score if hasattr(result, "score") else float(result)
                else:
                    score = float(metric(example, pred))

                scores_b.append(score)

            # Make sequential decision
            decision = self.sequential_test_decision(scores_a, scores_b)
            decisions.append(
                {
                    "sample_size_a": len(scores_a),
                    "sample_size_b": len(scores_b),
                    "decision": decision,
                    "mean_a": sum(scores_a) / len(scores_a) if scores_a else 0,
                    "mean_b": sum(scores_b) / len(scores_b) if scores_b else 0,
                }
            )

            if decision != TestDecision.CONTINUE:
                break

        # Final results
        final_decision = (
            decisions[-1]["decision"] if decisions else TestDecision.CONTINUE
        )

        return {
            "decision": final_decision,
            "scores_a": scores_a,
            "scores_b": scores_b,
            "sample_size_a": len(scores_a),
            "sample_size_b": len(scores_b),
            "mean_a": sum(scores_a) / len(scores_a) if scores_a else 0,
            "mean_b": sum(scores_b) / len(scores_b) if scores_b else 0,
            "decisions_history": decisions,
            "early_stopping": final_decision != TestDecision.CONTINUE,
            "samples_saved": max_samples - len(scores_a) - len(scores_b),
        }


class BayesianABTest:
    """Bayesian A/B testing with posterior distributions."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def update_posterior(self, successes: int, failures: int) -> Tuple[float, float]:
        """Update Beta posterior distribution."""
        # Beta(alpha + successes, beta + failures)
        posterior_alpha = self.prior_alpha + successes
        posterior_beta = self.prior_beta + failures
        return posterior_alpha, posterior_beta

    def calculate_probability_b_better(
        self, result: BayesianResult, n_samples: int = 10000
    ) -> float:
        """Calculate probability that variant B is better than A using Monte Carlo."""
        # Sample from Beta distributions
        samples_a = [
            random.betavariate(result.posterior_a_alpha, result.posterior_a_beta)
            for _ in range(n_samples)
        ]
        samples_b = [
            random.betavariate(result.posterior_b_alpha, result.posterior_b_beta)
            for _ in range(n_samples)
        ]

        # Count how often B > A
        b_better_count = sum(1 for a, b in zip(samples_a, samples_b) if b > a)

        return b_better_count / n_samples

    def calculate_expected_loss(
        self, result: BayesianResult, n_samples: int = 10000
    ) -> float:
        """Calculate expected loss of choosing wrong variant."""
        # Sample from Beta distributions
        samples_a = [
            random.betavariate(result.posterior_a_alpha, result.posterior_a_beta)
            for _ in range(n_samples)
        ]
        samples_b = [
            random.betavariate(result.posterior_b_alpha, result.posterior_b_beta)
            for _ in range(n_samples)
        ]

        # Expected loss if we choose A when B is better
        loss_choose_a = (
            sum(max(0, b - a) for a, b in zip(samples_a, samples_b)) / n_samples
        )

        # Expected loss if we choose B when A is better
        loss_choose_b = (
            sum(max(0, a - b) for a, b in zip(samples_a, samples_b)) / n_samples
        )

        # Return the minimum expected loss (optimal decision)
        return min(loss_choose_a, loss_choose_b)

    def run_bayesian_test(
        self, scores_a: List[float], scores_b: List[float]
    ) -> BayesianResult:
        """Run Bayesian A/B test."""

        # Convert scores to successes/failures (assuming scores are 0-1)
        successes_a = sum(1 for score in scores_a if score > 0.5)
        failures_a = len(scores_a) - successes_a

        successes_b = sum(1 for score in scores_b if score > 0.5)
        failures_b = len(scores_b) - successes_b

        # Update posterior distributions
        posterior_a_alpha, posterior_a_beta = self.update_posterior(
            successes_a, failures_a
        )
        posterior_b_alpha, posterior_b_beta = self.update_posterior(
            successes_b, failures_b
        )

        # Create result object
        result = BayesianResult(
            posterior_a_alpha=posterior_a_alpha,
            posterior_a_beta=posterior_a_beta,
            posterior_b_alpha=posterior_b_alpha,
            posterior_b_beta=posterior_b_beta,
            probability_b_better=0.0,  # Will be calculated
            expected_loss=0.0,  # Will be calculated
        )

        # Calculate probability B is better
        result.probability_b_better = self.calculate_probability_b_better(result)

        # Calculate expected loss
        result.expected_loss = self.calculate_expected_loss(result)

        return result


class MultiArmedBandit:
    """Multi-armed bandit testing for multiple variants."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon  # Exploration rate
        self.arm_counts = {}
        self.arm_rewards = {}
        self.arm_total_rewards = {}

    def select_arm(self, available_arms: List[str]) -> str:
        """Select arm using epsilon-greedy strategy."""

        # Initialize arms if not seen before
        for arm in available_arms:
            if arm not in self.arm_counts:
                self.arm_counts[arm] = 0
                self.arm_rewards[arm] = 0.0
                self.arm_total_rewards[arm] = 0.0

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: choose randomly
            return random.choice(available_arms)
        else:
            # Exploit: choose best arm
            best_arm = None
            best_reward = -float("inf")

            for arm in available_arms:
                if self.arm_counts[arm] == 0:
                    # Unplayed arm gets priority
                    return arm

                avg_reward = self.arm_rewards[arm]
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_arm = arm

            return best_arm if best_arm else available_arms[0]

    def update_arm(self, arm: str, reward: float) -> None:
        """Update arm statistics with new reward."""
        if arm not in self.arm_counts:
            self.arm_counts[arm] = 0
            self.arm_rewards[arm] = 0.0
            self.arm_total_rewards[arm] = 0.0

        self.arm_counts[arm] += 1
        self.arm_total_rewards[arm] += reward
        self.arm_rewards[arm] = self.arm_total_rewards[arm] / self.arm_counts[arm]

    def run_bandit_test(
        self,
        variants: Dict[str, Any],
        test_examples: List[dspy.Example],
        metric: Any,
        n_rounds: int = 1000,
    ) -> Dict[str, Any]:
        """Run multi-armed bandit test."""

        # Shuffle examples
        examples = test_examples.copy()
        random.shuffle(examples)

        # Ensure we have enough examples
        if len(examples) < n_rounds:
            # Repeat examples if needed
            examples = examples * (n_rounds // len(examples) + 1)

        arm_names = list(variants.keys())
        round_history = []

        for round_num in range(n_rounds):
            if round_num >= len(examples):
                break

            example = examples[round_num]

            # Select arm
            selected_arm = self.select_arm(arm_names)
            selected_variant = variants[selected_arm]

            # Generate prediction
            if hasattr(selected_variant, "forward"):
                inputs = example.inputs()
                pred = selected_variant(**inputs)
            else:
                pred = selected_variant(example)

            # Calculate reward
            if hasattr(metric, "evaluate"):
                result = metric.evaluate(example, pred)
                reward = result.score if hasattr(result, "score") else float(result)
            else:
                reward = float(metric(example, pred))

            # Update arm
            self.update_arm(selected_arm, reward)

            # Record round
            round_history.append(
                {
                    "round": round_num,
                    "selected_arm": selected_arm,
                    "reward": reward,
                    "arm_counts": self.arm_counts.copy(),
                    "arm_rewards": self.arm_rewards.copy(),
                }
            )

        # Calculate final statistics
        total_reward = sum(self.arm_total_rewards.values())
        best_arm = max(self.arm_rewards.items(), key=lambda x: x[1])

        return {
            "total_rounds": len(round_history),
            "total_reward": total_reward,
            "average_reward": total_reward / len(round_history) if round_history else 0,
            "best_arm": best_arm[0],
            "best_arm_reward": best_arm[1],
            "arm_statistics": {
                "counts": self.arm_counts.copy(),
                "rewards": self.arm_rewards.copy(),
                "total_rewards": self.arm_total_rewards.copy(),
            },
            "round_history": round_history,
            "exploration_rate": self.epsilon,
        }


# Sample systems for testing
class HighAccuracySystem:
    """High accuracy system."""

    def __init__(self, name: str = "High Accuracy"):
        self.name = name

    def forward(self, question):
        # Simulate high accuracy (80% success rate)
        if random.random() < 0.8:
            return dspy.Prediction(answer="correct answer")
        else:
            return dspy.Prediction(answer="incorrect answer")


class MediumAccuracySystem:
    """Medium accuracy system."""

    def __init__(self, name: str = "Medium Accuracy"):
        self.name = name

    def forward(self, question):
        # Simulate medium accuracy (60% success rate)
        if random.random() < 0.6:
            return dspy.Prediction(answer="correct answer")
        else:
            return dspy.Prediction(answer="incorrect answer")


class LowAccuracySystem:
    """Low accuracy system."""

    def __init__(self, name: str = "Low Accuracy"):
        self.name = name

    def forward(self, question):
        # Simulate low accuracy (40% success rate)
        if random.random() < 0.4:
            return dspy.Prediction(answer="correct answer")
        else:
            return dspy.Prediction(answer="incorrect answer")


class BinaryMetric:
    """Binary success/failure metric."""

    def evaluate(self, example, prediction):
        predicted = getattr(prediction, "answer", str(prediction))
        return 1.0 if "correct" in predicted else 0.0


def test_advanced_ab_testing():
    """Test the advanced A/B testing system."""
    print("Testing Advanced A/B Testing System")
    print("=" * 50)

    # Create test systems
    system_a = HighAccuracySystem("System A")
    system_b = MediumAccuracySystem("System B")
    system_c = LowAccuracySystem("System C")

    # Create test examples
    test_examples = [
        dspy.Example(question=f"Question {i}", answer="correct answer").with_inputs(
            "question"
        )
        for i in range(200)
    ]

    # Create metric
    metric = BinaryMetric()

    # Test 1: Sequential A/B Testing
    print("\n1. Testing Sequential A/B Testing:")
    print("-" * 40)

    sequential_tester = SequentialABTest(alpha=0.05, power=0.8, min_sample_size=20)

    seq_result = sequential_tester.run_sequential_test(
        system_a, system_b, test_examples, metric, max_samples=200
    )

    print(f"Decision: {seq_result['decision'].value}")
    print(
        f"Sample sizes: A={seq_result['sample_size_a']}, B={seq_result['sample_size_b']}"
    )
    print(f"Mean scores: A={seq_result['mean_a']:.3f}, B={seq_result['mean_b']:.3f}")
    print(f"Early stopping: {seq_result['early_stopping']}")
    print(f"Samples saved: {seq_result['samples_saved']}")

    # Show decision history
    print("\nDecision History:")
    for i, decision in enumerate(
        seq_result["decisions_history"][-3:]
    ):  # Last 3 decisions
        print(
            f"  Step {len(seq_result['decisions_history']) - 3 + i + 1}: "
            f"n_a={decision['sample_size_a']}, n_b={decision['sample_size_b']}, "
            f"decision={decision['decision'].value}"
        )

    # Test 2: Bayesian A/B Testing
    print("\n2. Testing Bayesian A/B Testing:")
    print("-" * 40)

    bayesian_tester = BayesianABTest(prior_alpha=1.0, prior_beta=1.0)

    # Generate some test scores
    test_scores_a = [
        metric.evaluate(None, system_a.forward(f"q{i}")) for i in range(50)
    ]
    test_scores_b = [
        metric.evaluate(None, system_b.forward(f"q{i}")) for i in range(50)
    ]

    bayesian_result = bayesian_tester.run_bayesian_test(test_scores_a, test_scores_b)

    print(
        f"Posterior A: Beta({bayesian_result.posterior_a_alpha:.1f}, {bayesian_result.posterior_a_beta:.1f})"
    )
    print(
        f"Posterior B: Beta({bayesian_result.posterior_b_alpha:.1f}, {bayesian_result.posterior_b_beta:.1f})"
    )
    print(f"P(B > A): {bayesian_result.probability_b_better:.3f}")
    print(f"Expected Loss: {bayesian_result.expected_loss:.4f}")

    # Interpretation
    if bayesian_result.probability_b_better > 0.95:
        print("Strong evidence that B is better than A")
    elif bayesian_result.probability_b_better > 0.8:
        print("Moderate evidence that B is better than A")
    elif bayesian_result.probability_b_better < 0.05:
        print("Strong evidence that A is better than B")
    elif bayesian_result.probability_b_better < 0.2:
        print("Moderate evidence that A is better than B")
    else:
        print("Inconclusive evidence")

    # Test 3: Multi-Armed Bandit Testing
    print("\n3. Testing Multi-Armed Bandit:")
    print("-" * 40)

    bandit = MultiArmedBandit(epsilon=0.1)

    variants = {"system_a": system_a, "system_b": system_b, "system_c": system_c}

    bandit_result = bandit.run_bandit_test(
        variants, test_examples[:100], metric, n_rounds=100
    )

    print(f"Total rounds: {bandit_result['total_rounds']}")
    print(f"Average reward: {bandit_result['average_reward']:.3f}")
    print(
        f"Best arm: {bandit_result['best_arm']} (reward: {bandit_result['best_arm_reward']:.3f})"
    )

    print("\nArm Statistics:")
    for arm, count in bandit_result["arm_statistics"]["counts"].items():
        reward = bandit_result["arm_statistics"]["rewards"][arm]
        percentage = (count / bandit_result["total_rounds"]) * 100
        print(f"  {arm}: {count} pulls ({percentage:.1f}%), avg reward: {reward:.3f}")

    # Show learning curve (last 20 rounds)
    print("\nLearning Progress (last 20 rounds):")
    recent_rounds = bandit_result["round_history"][-20:]
    for round_info in recent_rounds[::4]:  # Every 4th round
        round_num = round_info["round"]
        arm = round_info["selected_arm"]
        reward = round_info["reward"]
        print(f"  Round {round_num}: selected {arm}, reward: {reward:.1f}")

    # Test 4: Method Comparison
    print("\n4. Method Comparison:")
    print("-" * 40)

    print("Sequential Testing:")
    print(
        f"  - Samples used: {seq_result['sample_size_a'] + seq_result['sample_size_b']}"
    )
    print(f"  - Decision: {seq_result['decision'].value}")
    print(f"  - Efficiency: High (early stopping)")

    print("\nBayesian Testing:")
    print(f"  - Samples used: {len(test_scores_a) + len(test_scores_b)}")
    print(f"  - P(B better): {bayesian_result.probability_b_better:.3f}")
    print(f"  - Uncertainty: Quantified with posterior distributions")

    print("\nMulti-Armed Bandit:")
    print(f"  - Total rounds: {bandit_result['total_rounds']}")
    print(f"  - Best arm: {bandit_result['best_arm']}")
    print(f"  - Adaptivity: High (learns during testing)")

    # Recommendations
    print("\n5. Method Recommendations:")
    print("-" * 40)
    print("• Sequential Testing: Best for fixed sample size with early stopping")
    print("• Bayesian Testing: Best for uncertainty quantification and decision theory")
    print("• Multi-Armed Bandit: Best for online learning and multiple variants")
    print(
        "• Choose based on: sample size constraints, number of variants, and business context"
    )


if __name__ == "__main__":
    test_advanced_ab_testing()
