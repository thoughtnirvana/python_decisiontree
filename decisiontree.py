from utils import *
from xion.common import learners

class DecisionTree:
    """A DecisionTree holds an attribute that is being tested, and a
    dict of {attrval: Tree} entries.  If Tree here is not a DecisionTree
    then it is the final classification of the example."""

    def __init__(self, attr, val, attrname=None, branches=None):
        "Initialize by saying what attribute this node tests."
        update(self, val=val, attr=attr, attrname=attrname or attr,
               branches=branches or {})

    def predict(self, example):
        "Given an example, use the tree to classify the example."
        val = example[self.attr]
        branch = True
        if isnumber(val):
            if val > self.val: branch = False
        else:
            if val != self.val: branch = False
        child = self.branches[branch]
        if isinstance(child, DecisionTree):
            return child.predict(example)
        else:
            return child

    def add(self, val, subtree):
        "Add a branch.  If self.attr = val, go to the given subtree."
        self.branches[val] = subtree
        return self

    def display(self, indent=0):
        name = self.attrname
        print(' '*indent, 'Attribute :', name, 'Value :' , self.val, '?')
        indent += 4
        for branch in (True, False):
            subtree = self.branches[branch]
            if isinstance(subtree, DecisionTree):
                print(' '*indent, str(branch), '==>')
                subtree.display(indent)
            else:
                print(' '*indent, str(branch), '==>', subtree)

    def __repr__(self):
        return 'DecisionTree(%r, %r, %r, %r)' % (
            self.attr, self.val, self.attrname, self.branches)

class DecisionTreeLearner(learners.Learner):

    def predict(self, example):
        if isinstance(self.dt, DecisionTree):
            return self.dt.predict(example)
        else:
            return self.dt

    def train(self, dataset):
        self.dataset = dataset
        self.attrnames = dataset.attrnames
        self.dt = self.decision_tree_learning(dataset.examples, dataset.inputs)

    def decision_tree_learning(self, examples, attrs, default=None):
        if len(examples) == 0:
            return default
        elif self.all_same_class(examples):
            return examples[0][self.dataset.target]
        elif  len(attrs) == 0:
            return self.majority_value(examples)
        else:
            best, val, set1, set2  = self.choose_attribute(attrs, examples)
            tree = DecisionTree(best, val, self.attrnames[best])
            tb = self.decision_tree_learning(set1, removeall(best, attrs),
                                            self.majority_value(examples))
            fb = self.decision_tree_learning(set2, attrs,
                                            self.majority_value(examples))
            tree.add(True, tb)
            tree.add(False, fb)
            return tree

    def choose_attribute(self, attrs, examples):
        "Choose the attribute with the highest information gain."
        data_points = map(lambda a: self.information_gain(a, examples), attrs)
        return argmax_random_tie(data_points, lambda d: d[0])[1:]

    def all_same_class(self, examples):
        "Are all these examples in the same target class?"
        target = self.dataset.target
        class0 = examples[0][target]
        for e in examples:
           if e[target] != class0: return False
        return True

    def majority_value(self, examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        g = self.dataset.target
        return argmax_random_tie(self.dataset.values[g],
                      lambda v: self.count(g, v, examples))

    def count(self, attr, val, examples):
        return count_if(lambda e: e[attr] == val, examples)

    def information_gain(self, attr, examples, scoref=entropy):
        current_score = scoref(examples)
        N = float(len(examples))
        best_gain = 0; best_value = None; best_tb = None; best_fb = None; best_attr = None
        for (v, branches) in self.split_by(attr, examples):
            gain = 0
            tb, fb = branches
            p = len(tb) / N
            gain =  current_score - p*scoref(tb) - (1-p)*scoref(fb)
            if gain > best_gain:
                (best_gain, best_attr, best_value, best_tb, best_fb) = gain, attr, v, tb, fb
        return (best_gain, best_attr, best_value, best_tb, best_fb)

    def split_by(self, attr, examples=None):
        "Return a list of (val, examples) pairs for each val of attr."
        if examples == None:
            examples = self.dataset.examples
        return [(v, self.divide_set(attr, examples, v))
                for v in self.dataset.values[attr]]

    def divide_set(self, attr, examples, value, only_tb=False):
        """
        Divides a set on a specific column. Can handle numeric
        or nominal values
        """
        # Make a function that tells us if a row is in 
        # the first group (true) or the second group (false)
        split_fn = lambda row: row[attr] == value
        if isnumber(value):
            split_fn = lambda row: row[attr] >= value
        # Divide the rows into two sets and return them
        set1 = [row for row in examples if split_fn(row)]
        set2 = [row for row in examples if not split_fn(row)]
        if only_tb: return set1
        return (set1,set2)

