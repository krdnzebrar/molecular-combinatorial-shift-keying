"""
modulation/alphabet.py
======================
Symbol alphabet generators for all modulation schemes.
Single definitions — previously duplicated across scheme cells.
"""

from itertools import permutations, combinations
import math
import numpy as np


def get_all_permutations(num_molecule_types):
    """
    All full permutations of K molecule types → K! symbols.

    Returns
    -------
    all_perms : list of tuple
        Each tuple is a permutation of range(K).
    perm_strings : list of str
        Human-readable names like 'ABCDE', 'BACED', etc.
    """
    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    all_perms = list(permutations(range(num_molecule_types)))
    perm_strings = [''.join(molecule_names[i] for i in p) for p in all_perms]
    return all_perms, perm_strings


def get_all_combinations(num_molecule_types):
    """
    All non-empty subsets of K molecule types → 2^K − 1 symbols.

    Returns
    -------
    all_combos : list of tuple
        Each tuple is a sorted subset of molecule indices.
    combo_strings : list of str
        Names like 'A', 'AB', 'BCE', 'ABCDE', etc.
    """
    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    all_combos = []
    combo_strings = []
    for size in range(1, num_molecule_types + 1):
        for combo in combinations(range(num_molecule_types), size):
            all_combos.append(combo)
            combo_strings.append(''.join(molecule_names[i] for i in combo))
    return all_combos, combo_strings


def sample_ordered_subset(num_molecule_types):
    """Uniformly sample a MoCSK symbol, including the empty symbol."""
    k = int(num_molecule_types)
    sizes = np.arange(k + 1)
    counts = np.asarray([
        math.factorial(k) // math.factorial(k - size)
        for size in sizes
    ], dtype=float)
    size = int(np.random.choice(sizes, p=counts / counts.sum()))
    if size == 0:
        return tuple()
    return tuple(int(m) for m in np.random.permutation(k)[:size])


def get_all_sparse_patterns(num_molecule_types):
    """
    All sparse patterns: partial slot fills with permutations.

    Includes the all-empty symbol, then for each active-slot count 1..K
    chooses slots and assigns distinct molecule types. Empty slots are None.

    Returns
    -------
    all_patterns : list of tuple
        Each tuple has K entries; int (molecule index) or None (empty slot).
    pattern_strings : list of str
        Names like 'A____', '_B___', 'AB___', 'A_C_E', etc.
    """
    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    all_patterns = []
    pattern_strings = []
    all_patterns.append(tuple([None] * num_molecule_types))
    pattern_strings.append('_' * num_molecule_types)
    for num_active in range(1, num_molecule_types + 1):
        for active_slots in combinations(range(num_molecule_types), num_active):
            for mol_perm in permutations(range(num_molecule_types), num_active):
                pattern = [None] * num_molecule_types
                for slot, mol in zip(active_slots, mol_perm):
                    pattern[slot] = mol
                all_patterns.append(tuple(pattern))
                pattern_strings.append(
                    ''.join(molecule_names[m] if m is not None else '_'
                            for m in pattern))
    return all_patterns, pattern_strings
