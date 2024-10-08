from collections import defaultdict

class Agent(object):
    def __init__(self, phoneme_table, vocabulary) -> None:
        #Your agent initialization goes here. You can also add code but don't remove the existing code.
        self.phoneme_table = phoneme_table
        self.vocabulary = vocabulary
        self.best_state = None
        self.best_cost = float('inf')

        self.threshold = 0.2
        self.threshold_vocab = 0.1
        self.max_children = 10
        self.phonemes = set()

        reverse_mapping = defaultdict(list)
        # Construct the reverse mapping
        for key, values in phoneme_table.items():
            for value in values:
                if value not in self.phonemes:
                    self.phonemes.add(value)
                reverse_mapping[value].append(key)

        self.replacements = dict(reverse_mapping)

    def asr_corrector(self, environment):
        """ASR corrector that performs local search on each word individually, computing the cost of the entire sentence."""
        self.best_state = environment.init_state
        self.best_cost = environment.compute_cost(environment.init_state)
        self.costs = {}
        self.costs[self.best_state] = self.best_cost
        self.first = 0
        self.last = 0
        self.best_states = None


        # phoneme_index, sentence
        to_process = [(self.first, self.best_state),]

        while True:
            all_neighbors = []
            # print(f"Best state: {self.best_state}")
            # print(f"Best cost: {self.best_cost}")
            # if len(to_process) > 0: print(to_process[0][0])

            for state in to_process:
                all_neighbors.extend(self.get_neighbors(state[1], state[0], environment))

            if len(all_neighbors) == 0:
                self.best_states = to_process.copy()
                break

            all_neighbors.sort()
            # print(f"All neighbours: {all_neighbors}")
            # print()
            
            to_process.clear()

            for neighbor in all_neighbors:
                if neighbor[0] < all_neighbors[0][0]*(1 + self.threshold):
                    to_process.append((neighbor[1], neighbor[2]))
                if len(to_process) >= self.max_children:
                    break

            # print(f"Selected neighbours: {to_process}")
            # print()
            
            if all_neighbors[0][0] < self.best_cost:
                self.best_state = all_neighbors[0][2]
                self.best_cost = all_neighbors[0][0]

            if all_neighbors[0][1] >= len(all_neighbors[0][2])-self.last:
                self.best_states = to_process.copy()
                break
            # print('*'*100)

        self.try_vocabulary_insertion(environment)

    def get_neighbors(self, sentence, phoneme_index, environment):
        """Generate neighboring states by replacing the passed phoneme based on the phoneme table."""
        if (phoneme_index >= len(sentence)-self.last): return []

        neighbors = []

        if sentence not in self.costs:
            self.costs[sentence] = environment.compute_cost(sentence)

        phoneme = sentence[phoneme_index]
        if phoneme_index < len(sentence)-self.last-1:
            if sentence[phoneme_index : phoneme_index+2] in self.phonemes:
                phoneme = sentence[phoneme_index : phoneme_index+2]
            if phoneme_index + len(phoneme) < len(sentence):
                neighbors = [(self.costs[sentence], (phoneme_index+len(phoneme)+1 if sentence[phoneme_index+len(phoneme)]==' ' else phoneme_index+len(phoneme)), sentence)]
            elif len(phoneme)==2:
                neighbors = [(self.costs[sentence], phoneme_index+2, sentence)]
        if len(neighbors) == 0:
            neighbors = [(self.costs[sentence], phoneme_index+1, sentence)]


        if phoneme in self.replacements.keys():
            for replacement in self.replacements[phoneme]:
                new_sentence = sentence[ : phoneme_index] + replacement + sentence[phoneme_index+len(phoneme) : ]
                new_cost = environment.compute_cost(new_sentence)
                new_index = phoneme_index+len(replacement)
                if (new_index <len(new_sentence)):
                    if new_sentence[new_index] == ' ':
                        new_index += 1
                neighbors.append((new_cost, new_index, new_sentence))

        return neighbors

    def try_vocabulary_insertion(self, environment):
        # Check insertion at the beginning
        starting_words = []
        ending_words = []

        for word in self.vocabulary:
            new_sentence = word + " " + self.best_state
            new_cost = environment.compute_cost(new_sentence)
            if new_cost < self.best_cost*(1 + self.threshold_vocab):
                starting_words.append(word)
                self.costs[new_sentence] = new_cost

        for word in self.vocabulary:
            new_sentence = self.best_state + " " + word
            new_cost = environment.compute_cost(new_sentence)
            if new_cost < self.best_cost*(1 + self.threshold_vocab):
                ending_words.append(word)
                self.costs[new_sentence] = new_cost

        # print(starting_words)
        # print(ending_words)
        for state in self.best_states:
            current_state = state[1]
            current_state_2 = current_state

            for word in starting_words:
                new_sentence = word + " " + current_state
                new_cost = environment.compute_cost(new_sentence) if new_sentence not in self.costs else self.costs[new_sentence]
                if new_cost < self.best_cost:
                    # print("Old Cost: ", self.best_cost, self.best_state)
                    # print("New Cost: ", new_cost, new_sentence)
                    self.first = len(word)+1
                    self.best_state = new_sentence
                    current_state_2 = new_sentence
                    self.best_cost = new_cost
                    self.costs[self.best_state] = self.best_cost

            # print(current_state)

            for word in ending_words:
                new_sentence = current_state_2 + " " + word
                new_cost = environment.compute_cost(new_sentence) if new_sentence not in self.costs else self.costs[new_sentence]
                if new_cost < self.best_cost:
                    # print("Old Cost: ", self.best_cost, self.best_state)
                    # print("New Cost: ", new_cost, new_sentence)
                    self.last = len(word)+1
                    self.best_state = new_sentence
                    self.best_cost = new_cost
                    self.costs[self.best_state] = self.best_cost