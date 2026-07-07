from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class CellularAutomata():
    def __init__(
        self,
        rule: int,
        width: int,
        nb_epochs: int,
        seed: int,
        dense_initialization: bool
    ) -> None:
        self.rule = rule
        self.width = width
        self.nb_epochs = nb_epochs
        self.seed = seed
        self.rng = np.random.default_rng(self.seed )
        self.dense_initialization = dense_initialization
        self.grid = np.zeros((self.nb_epochs, self.width), dtype=int)
        self.binary_rule = self.parse_rule()


    def initialize(self) -> None:
        if self.dense_initialization:
            self.grid[0, :] = self.rng.integers(low=0, high=2, size=self.width)
        else:
            mid_width = self.width // 2
            self.grid[0, mid_width] = 1


    def evolve(self) -> None:
        for epoch in range(self.nb_epochs -1):
            triplets = self.extract_triplets(epoch=epoch)
            self.grid[epoch +1, :] = self.apply_rule(triplets=triplets)


    def parse_rule(self) -> np.ndarray:
        """Wolfram convention: binary_rule[0] handles configuration 111, binary_rule[7] handles 000"""
        binary = format(self.rule, "08b")
        binary_rule = np.array(list(binary), dtype=int)
        return binary_rule[::-1]


    def extract_triplets(self, epoch: int) -> np.ndarray:
        current_generation = self.grid[epoch,:]
        left_neighbors = np.roll(current_generation, 1)
        right_neighbors = np.roll(current_generation, -1)
        triplets = np.stack([left_neighbors, current_generation, right_neighbors], axis=-1)
        return triplets


    def apply_rule(self, triplets: np.ndarray) -> np.ndarray:
        powers_of_two = np.array([4, 2, 1])
        indices = np.dot(triplets, powers_of_two)
        return self.binary_rule[indices]


    def get_config_name(self) -> str:
        shape = f"{self.nb_epochs}x{self.width}"
        init = f"_dense_initialization_{self.seed}" if self.dense_initialization else ""
        return shape + init


    def plot(self) -> None:
        output_dir = Path("images") / self.get_config_name()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
        ax.imshow(self.grid, cmap="binary", interpolation="nearest")
        ax.set_aspect("equal")
        ax.axis("off")
        plt.title(f"Rule {self.rule}")
        file_path = output_dir / f"rule_{self.rule}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def get_independent_rules() -> list[int]:

    independent_rules = set()

    nb_rules = int(2**8)
    for rule_original in range(nb_rules):

        # Original
        bits_original = [int(b) for b in format(rule_original, "08b")]

        # Mirror equivalence
        bits_mirror = [
            bits_original[0],
            bits_original[4],
            bits_original[2],
            bits_original[6],
            bits_original[1],
            bits_original[5],
            bits_original[3],
            bits_original[7]
        ]
        rule_mirror = int("".join(map(str, bits_mirror)), 2)

        # Complement equivalence
        bits_complement = [1 - b for b in bits_original[::-1]]
        rule_complement = int("".join(map(str, bits_complement)), 2)

        # Mirror-Complement equivalence
        bits_mirror_complement = [1 - b for b in bits_mirror[::-1]]
        rule_mirror_complement = int("".join(map(str, bits_mirror_complement)), 2)

        # Elect the representative for this equivalence cluster
        representative = min(rule_original, rule_mirror, rule_complement, rule_mirror_complement)
        independent_rules.add(representative)

    return sorted(list(independent_rules))


if __name__ == "__main__":
    rules = get_independent_rules()
    for rule in rules:
        ca = CellularAutomata(
            rule = rule,
            width = 512,
            nb_epochs = 300,
            seed = 7,
            dense_initialization = False,
        )
        ca.initialize()
        ca.evolve()
        ca.plot()
