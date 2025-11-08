import numpy as np


class GreedySearchDecoder(object):
    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = ""
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        path_prob = 1.0
        uncompressed_path = []
        for t in range(y_probs.shape[1]):
            max_prob = -1
            max_idx = -1
            for s in range(y_probs.shape[0]):
                if y_probs[s, t, 0] > max_prob:
                    max_prob = y_probs[s, t, 0]
                    max_idx = s
            path_prob *= max_prob
            if max_idx != 0:  # Skip the blanks
                uncompressed_path.append(self.symbol_set[max_idx - 1])

        # Compress the path by removing blanks and repeated symbols
        for i in range(len(uncompressed_path)):
            if i == 0 or uncompressed_path[i] != uncompressed_path[i - 1]:
                decoded_path += uncompressed_path[i]

        return decoded_path, path_prob


class BeamSearchDecoder(object):
    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None

        # TODO:
        # Implement the beam search decoding algorithm. This typically involves:
        # 1. Initializing a set of paths with their probabilities.
        # 2. For each time step, extending existing paths with all possible symbols.
        # 3. Merging paths that end in the same symbol.
        # 4. Pruning the set of paths to keep only the top 'beam_width' paths.
        # 5. After iterating through all time steps, selecting the best path
        #    and its score.

        self.blank_path_score = {}
        self.path_score = {}

        (
            new_paths_with_terminal_blank,
            new_paths_with_terminal_symbol,
            new_blank_path_score,
            new_path_score,
        ) = self.initialize_paths(y_probs)

        for t in range(1, T):
            (
                paths_with_terminal_blank,
                paths_with_terminal_symbol,
                self.blank_path_score,
                self.path_score,
            ) = self.prune(
                new_paths_with_terminal_blank,
                new_paths_with_terminal_symbol,
                new_blank_path_score,
                new_path_score,
            )

            # First extend paths by a blank
            (
                new_paths_with_terminal_blank,
                new_blank_path_score,
            ) = self.extend_with_blank(
                paths_with_terminal_blank,
                paths_with_terminal_symbol,
                y_probs[:, t, 0],
            )

            # Next extend paths by a symbol
            new_paths_with_terminal_symbol, new_path_score = self.extend_with_symbol(
                paths_with_terminal_blank, paths_with_terminal_symbol, y_probs[:, t, 0]
            )

        merged_paths, merged_path_scores = self.merge_identical_paths(
            new_paths_with_terminal_blank,
            new_blank_path_score,
            new_paths_with_terminal_symbol,
            new_path_score,
        )

        bestPath = None
        FinalPathScore = -1
        for path in merged_paths:
            if merged_path_scores[path] > FinalPathScore:
                FinalPathScore = merged_path_scores[path]
                bestPath = path

        return bestPath, merged_path_scores

    def initialize_paths(self, y_probs):
        """
        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size
        """
        initial_blank_path_score = {}
        initial_path_score = {}

        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ""
        initial_blank_path_score[path] = y_probs[0, 0, 0]  # blank prob at t=0
        initial_paths_with_final_blank = {""}

        # Push rest of the symbols into a path-ending-with-symbol stack
        initial_paths_with_final_symbols = set()
        for i, c in enumerate(self.symbol_set):
            path = c
            initial_path_score[path] = y_probs[i + 1, 0, 0]  # symbol prob at t=0
            initial_paths_with_final_symbols.add(path)

        return (
            initial_paths_with_final_blank,
            initial_paths_with_final_symbols,
            initial_blank_path_score,
            initial_path_score,
        )

    def prune(
        self,
        paths_with_terminal_blank,
        paths_with_terminal_symbol,
        blank_path_score,
        path_score,
    ):
        """
        Prune paths to keep only top 'beam_width' paths.
        """

        pruned_blank_path_score = {}
        pruned_path_score = {}

        # First gather all the relevant scores
        score_list = []
        for path in paths_with_terminal_blank:
            score_list.append(blank_path_score[path])

        for path in paths_with_terminal_symbol:
            score_list.append(path_score[path])

        # Sort and find cutoff score that retains exactly BeamWidth paths
        score_list.sort(reverse=True)
        cutoff = (
            score_list[self.beam_width - 1]
            if len(score_list) >= self.beam_width
            else score_list[-1]
        )

        pruned_paths_with_terminal_blank = set()
        for p in paths_with_terminal_blank:
            if blank_path_score[p] >= cutoff:
                pruned_paths_with_terminal_blank.add(p)
                pruned_blank_path_score[p] = blank_path_score[p]

        pruned_paths_with_terminal_symbol = set()
        for p in paths_with_terminal_symbol:
            if path_score[p] >= cutoff:
                pruned_paths_with_terminal_symbol.add(p)
                pruned_path_score[p] = path_score[p]

        return (
            pruned_paths_with_terminal_blank,
            pruned_paths_with_terminal_symbol,
            pruned_blank_path_score,
            pruned_path_score,
        )

    def extend_with_blank(
        self,
        paths_with_terminal_blank,
        paths_with_terminal_symbol,
        y_probs,
    ):
        """
        Extend paths by a blank.
        """
        updated_paths_with_terminal_blank = set()
        updated_blank_path_score = {}

        # First work on paths with terminal blanks
        for path in paths_with_terminal_blank:
            updated_paths_with_terminal_blank.add(path)
            updated_blank_path_score[path] = self.blank_path_score[path] * y_probs[0]

        # Then extend paths with terminal symbols by blanks
        for path in paths_with_terminal_symbol:
            if path in updated_paths_with_terminal_blank:
                updated_blank_path_score[path] += self.path_score[path] * y_probs[0]
            else:
                updated_paths_with_terminal_blank.add(path)
                updated_blank_path_score[path] = self.path_score[path] * y_probs[0]
        return updated_paths_with_terminal_blank, updated_blank_path_score

    def extend_with_symbol(
        self,
        paths_with_terminal_blank,
        paths_with_terminal_symbol,
        y_probs,
    ):
        """
        Extend paths by a symbol.
        """
        updated_paths_with_terminal_symbol = set()
        updated_path_score = {}

        # First extend paths with terminal blanks
        for path in paths_with_terminal_blank:
            for i, c in enumerate(self.symbol_set):
                new_path = path + c
                updated_paths_with_terminal_symbol.add(new_path)
                updated_path_score[new_path] = (
                    self.blank_path_score[path] * y_probs[i + 1]
                )

        # Then extend paths with terminal symbols
        for path in paths_with_terminal_symbol:
            for i, c in enumerate(self.symbol_set):
                new_path = path if path[-1] == c else path + c
                prob = self.path_score[path] * y_probs[i + 1]
                if new_path in updated_paths_with_terminal_symbol:
                    updated_path_score[new_path] += prob
                else:
                    updated_paths_with_terminal_symbol.add(new_path)
                    updated_path_score[new_path] = prob

        return updated_paths_with_terminal_symbol, updated_path_score

    def merge_identical_paths(
        self,
        paths_with_terminal_blank,
        blank_path_score,
        paths_with_terminal_symbol,
        path_score,
    ):
        """
        Merge identical paths ending with blank and symbol.
        """
        # All paths with terminal symbols will remain
        merged_paths = set(paths_with_terminal_symbol)
        final_path_scores = path_score.copy()

        # Paths with terminal blanks will contribute scores to existing identical paths from
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in paths_with_terminal_blank:
            if p in merged_paths:
                final_path_scores[p] += blank_path_score[p]
            else:
                merged_paths.add(p)
                final_path_scores[p] = blank_path_score[p]

        return merged_paths, final_path_scores
