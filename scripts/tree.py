import argparse
import numpy as np
import subprocess
import ete3
from collections import deque
from math import fsum
from decimal import *

class Tree:
    def __init__(self, input_tree=None, input_format='newick'):
        self.node_parent = {}
        self.node_children = {}

        self.relabel_counter = 0
        self.node_count = 0
        self.branch_lengths = {}
        self.node_depths = {}
        self.end_nodes = []
        self.leaves = []

        if input_tree is not None:
            if input_format == 'ete3':
                # Draw random number to make unique file name
                # SCC doesn't use new RNG
                #rng = np.random.default_rng()
                #r = rng.integers(2**32 - 1)

                r = np.random.randint(0, 2**32 - 1)
                input_tree.write(format=1, outfile=f'tree_ete3_{r}.tmp')
                self.input_file = f'tree_ete3_{r}.tmp'
            else:
                self.input_file = input_tree

            with open(self.input_file, 'r') as f_in:
                str_nwk = str(f_in.readline())
                self.read_tree(str_nwk)

            if input_format == 'ete3':
                subprocess.call([f"rm -f {self.input_file}"], shell=True)


    def __repr__(self):
        return f'top-bottom tree: {self.node_children}\nbottom-top tree: {self.node_parent}\nbranch lengths: {self.branch_lengths}'

    def read_tree(self, tree_nwk):
        tree_list = self.nwk_list(tree_nwk)
        deq = deque(iter(tree_list))
        self.build_tree_bottomtop(deq)

    def build_tree_bottomtop(self, deq, root=True):
        self.node_parent = {}
        self.node_children = {}
        stack = deque()

        # Add root
        if root == True:
            root = deq.pop()
        self.node_parent['root'] = {}
        stack.append('root')

        while deq:
            nwk_entry = deq.pop()
            node = self.parse_nwk_entry(nwk_entry)
            if node['branch'] == 'open':
                if stack:
                    parent = stack.pop()
                    self.update_tree({'node':node['name'],
                                      'parent':parent,
                                      'is_end_node':False,
                                      'branch_length':node['branch_length']})
                    stack.append(parent)
                else:
                    self.update_tree({'node':node['name'],
                                      'parent':'root',
                                      'is_end_node':False,
                                      'branch_length':node['branch_length']})

                stack.append(node['name'])
            elif node['branch'] is None:
                parent = stack.pop()
                self.update_tree({'node':node['name'],
                                  'parent':parent,
                                  'is_end_node':True,
                                  'branch_length':node['branch_length']})
                stack.append(parent)
            else:
                # nwk_entry is '('
                if stack:
                    stack.pop()
        self.calculate_node_depths()
        self.find_leaves()

    def nwk_list(self, nwk_str):
        nwk_str = nwk_str.replace(' ', '')
        tree_it = iter(nwk_str)
        x = ''
        s = next(tree_it)
        tree_list = []
        while s is not None and s != ';':
            while s not in ['(', ',', ')']:
                x = x + s
                s = next(tree_it)
            if s == '(':
                tree_list.append(s)
                x = ''
                s = next(tree_it)
            elif s == ',':
                tree_list.append(x)
                x = ''
                s = next(tree_it)
            else:
                tree_list.append(x)
                x = s
                s = next(tree_it)
        tree_list.append(x)
        return tree_list

    def parse_nwk_entry(self, s):
        entry_dict = {}
        if s[0] == ')':
            entry_dict['branch'] = 'open'
            node_dict = self.parse_node_name(s[1:])
            entry_dict = {**entry_dict, **node_dict}
        elif s != '(':
            entry_dict['branch'] = None
            node_dict = self.parse_node_name(s)
            entry_dict = {**entry_dict, **node_dict}
        else:
            entry_dict['branch'] = 'close'
            entry_dict['name'] = None
            entry_dict['branch_length'] = None
        return entry_dict

    def parse_node_name(self, s):
        node_dict = {}
        if ':' in list(s):
            if s[0] == ':':
                node_dict['name'] = f'label{self.relabel_counter}'
                #node_dict['branch_length'] = float(s[1:])
                node_dict['branch_length'] = Decimal(s[1:])
                self.relabel_counter += 1
            else:
                node = s.split(':')
                node_dict['name'] = node[0]
                #node_dict['branch_length'] = float(node[1])
                node_dict['branch_length'] = Decimal(node[1])
        else:
            if s == '':
                node_dict['name'] = f'label{self.relabel_counter}'
                node_dict['branch_length'] = None
                self.relabel_counter += 1
            else:
                node_dict['name'] = s
                node_dict['branch_length'] = None
        return node_dict

    def update_tree(self, update_dict):
        self.add_child(update_dict['node'], update_dict['parent'], end_node=update_dict['is_end_node'])
        self.node_parent[update_dict['node']] = update_dict['parent']
        self.branch_lengths[update_dict['node']] = update_dict['branch_length']
        self.node_count += 1

    def add_child(self, child, parent, end_node=False):
        if parent not in self.node_children.keys():
            self.node_children[parent] = [child]
        elif child not in self.node_children[parent]:
            self.node_children[parent].append(child)
        if end_node == True:
            self.node_children[child] = []
            self.end_nodes.append(child)

    def calculate_node_depths(self, units='time', round_decimal=None, round_off=None):
        self.node_depths = {'root':0}
        for node in self.node_parent.keys():
            if node not in self.node_depths.keys():
                node_depth = 0
                parent_node = node
                while parent_node not in self.node_depths.keys():
                    if units == 'time':
                        l = self.branch_lengths[parent_node]
                        if l is not None:
                            if round_decimal is None:
                                node_depth += l
                            else:
                                node_depth = np.round(node_depth + l, round_decimal)
                    else:
                        node_depth += 1
                    parent_node = self.node_parent[parent_node]
                if round_decimal is None:
                    self.node_depths[node] = node_depth + self.node_depths[parent_node]
                else:
                    self.node_depths[node] = np.round(node_depth + self.node_depths[parent_node], round_decimal)

        if round_off is not None:
            for node, depth in self.node_depths.items():
                self.node_depths[node] = round(depth, round_off)


    def calculate_node_depths_exact(self, units='time'):
        self.node_depths = {'root':0}
        node_branch_lengths = {'root':[0]} # dict to store list of branch lengths that sum to distance to root
        for node in self.node_parent.keys():
            if node not in node_branch_lengths.keys():
                parent_node = node
                branch_lengths = []
                while parent_node not in node_branch_lengths:
                    if units == 'time':
                        l = self.branch_lengths[parent_node]
                        if l is not None:
                            branch_lengths.append(l)
                    else:
                        branch_lengths.append(l)
                    parent_node = self.node_parent[parent_node]
                node_branch_lengths[node] = branch_lengths + node_branch_lengths[parent_node]

        for node, branch_lengths in node_branch_lengths.items():
            self.node_depths[node] = fsum(branch_lengths)

    def calculate_node_depths_decimal(self, units='time'):
        self.node_depths = {'root':Decimal(0)}
        for node in self.node_parent.keys():
            if node not in self.node_depths:
                node_depth = Decimal(0)
                parent_node = node
                while parent_node not in self.node_depths:
                    if units == 'time':
                        l = self.branch_lengths[parent_node]
                        if l is not None:
                            node_depth += Decimal(l)
                    else:
                        node_depth += Decimal(1)
                    parent_node = self.node_parent[parent_node]
                self.node_depths[node] = Decimal(node_depth + self.node_depths[parent_node])

    def total_tree_length(self, ignore_none=True, exclude_root=True):
        L = 0
        for node in self.branch_lengths.keys():
            if exclude_root == True and (node == 'root' or self.node_parent[node] == 'root'):
                continue
            l = self.branch_lengths[node]
            if l is not None:
                L += l
            elif ignore_none == False:
                L = None
                break
        return L

    def find_leaves(self):
        '''
        self.leaves = []
        max_depth = np.max(list(self.node_depths.values()))
        for node in self.end_nodes:
            if self.node_depths[node] == max_depth:
                self.leaves.append(node)
                self.node_children[node] = []
        '''
        self.leaves = []
        for node, children in self.node_children.items():
            if children == []:
                self.leaves.append(node)

    def calculate_root_distances(self):
        preordered_nodes = self.preorder_nodes()
        root_distances = {'root':0}

    def sample_subtree(self, leaves):
        subtree = Tree()
        num_branches = [len(leaves)]
        current_generation = leaves
        previous_generation = []
        num_generations = 0

        while num_branches[-1] != 1 or mrca != 'root':
            mrca = current_generation[0]
            for node in current_generation:
                if node != 'root':
                    parent = self.node_parent[node]
                    is_end_node = (num_generations == 0)
                    subtree.update_tree({'node':node,
                                         'parent':parent,
                                         'is_end_node':is_end_node,
                                         'branch_length':self.branch_lengths[node]})
                    if parent not in previous_generation and parent != {}:
                        previous_generation.append(parent)
            num_branches.append(len(previous_generation))
            current_generation = previous_generation
            mrca = current_generation[0]
            previous_generation = []
            num_generations += 1
        #mrca = current_generation[0]
        #if mrca != 'root':
            #subtree.root_tree(current_generation[0])

        # TODO: calculate_node_depths sometimes doesn't work on subtrees
        # Not clear why atm
        #subtree.calculate_node_depths()
        #subtree.find_leaves()
        subtree.leaves = leaves
        return subtree

    def root_tree(self, node):
        self.update_tree({'node':node, 'parent':'root', 'is_end_node':False, 'branch_length':0})
        self.node_parent['root'] = {}
        self.branch_lengths['root'] = 0

    def save(self, output_file, output_format='newick'):
        stack = deque()
        preorder = self.preorder_nodes()
        with open(output_file, 'w') as f_out:
            self.newick(f_out, preorder, stack, remaining_children={})

    def preorder_nodes(self):
        preorder, stack = self.initialize_preorder('root')
        preorder = self.preorder_stack(preorder, stack)
        return preorder

    def initialize_preorder(self, node):
        preorder = deque([node])
        stack = deque()

        for child in self.node_children[node]:
            stack.append(child)
        return preorder, stack

    def preorder_stack(self, preorder, stack):
        while stack:
            node = stack.pop()
            preorder.append(node)
            if len(self.node_children[node]) > 0:
                for child in self.node_children[node]:
                    stack.append(child)
        return preorder

    def newick(self, f_out, preorder, stack, remaining_children={}):
        while preorder:
            node = preorder.popleft()

            if node not in remaining_children.keys():
                remaining_children[node] = len(self.node_children[node])

            if remaining_children[node] > 0:
                f_out.write('(')
                stack.append(node)
                remaining_children[node] = len(self.node_children[node])
            else:
                f_out.write(f'{self.format_node(node)}')

                if stack:
                    parent = stack.pop()
                    while remaining_children[parent] == 1 and stack:
                        remaining_children[parent] = 0
                        f_out.write(f'){self.format_node(parent)}')
                        node = parent
                        parent = stack.pop()

                    remaining_children[parent] -= 1
                    stack.append(parent)
                    # hack to stop final comma
                    if preorder:
                        f_out.write(',')
        f_out.write(');')

    def format_node(self, node):
        if self.branch_lengths[node] is not None:
            node_str = f'{node}:{self.branch_lengths[node]}'
        else:
            node_str = f'{node}'
        return node_str

    def trim_tree(self, trim_distance, units='nodes'):
        new_leaves = []

        if units == 'nodes':
            trimmed_tree_leaves = self.leaves
            node_level = 0
            while node_level < trim_distance:
                for node in trimmed_tree_leaves:
                    parent = self.node_parent[node]
                    if parent not in new_leaves and self.node_parent[parent] != 'root':
                        new_leaves.append(parent)
                trimmed_tree_leaves = new_leaves
                new_leaves = []
                node_level += 1
            subtree = self.sample_subtree(trimmed_tree_leaves)
        else:
            '''
            Trim based on branch lengths
            '''
            trim_length = self.init_trim_lengths(trim_distance)
            for node in self.leaves:
                l = trim_length.pop(node)
                new_node = node
                while self.branch_lengths[new_node] <= l:
                    l -= self.branch_lengths[new_node]
                    new_node = self.node_parent[new_node]
                    if new_node in trim_length:
                        break
                trim_length[new_node] = self.branch_lengths[new_node] - l
            subtree = self.sample_subtree(list(trim_length.keys()))
            for new_leaf in trim_length.keys():
                subtree.branch_lengths[new_leaf] = trim_length[new_leaf]

        return subtree

    def init_trim_lengths(self, t):
        trim_length = {}
        for node in self.leaves:
            if node in trim_length:
                print(node, trim_length[node])
            trim_length[node] = t
        return trim_length

    def coarse_grain_tree(self, t_cg, epsilon=1E-4, depth_round_off=None):
        '''
        Creates new tree by taking all mergers within window of approximate size ``t_cg`` to happen at the same time.
        '''
        if depth_round_off is not None:
            # Recalculate node depths using round-off
            self.calculate_node_depths(round_off=depth_round_off)
        T_tree = float(max(self.node_depths.values()))
        num_tpoints = int(T_tree // t_cg)
        t_bin_edges = np.linspace(T_tree, 0, num=num_tpoints + 1)
        for i in range(num_tpoints):
            t = (t_bin_edges[i] + t_bin_edges[i + 1]) / 2
            bin_nodes = self.find_mergers_in_window([t_bin_edges[i + 1], t_bin_edges[i]])
            if len(bin_nodes) == 1:
                self.adjust_node_time(bin_nodes[0], t)
            elif len(bin_nodes) > 1:
                grouped_nodes = self.group_consecutive_mergers(bin_nodes)
                for node_group in grouped_nodes:
                    if len(node_group) == 1:
                        self.adjust_node_time(node_group[0], t)
                    else:
                        self.merge_internal_nodes(node_group, t)

    def find_mergers_in_window(self, t_window):
        internal_nodes = []
        for node, t in self.node_depths.items():
            if t > t_window[0] and t <= t_window[1]:
                if node not in self.leaves:
                    internal_nodes.append(node)
        return internal_nodes

    def adjust_node_time(self, node, t_new):
        t_old = float(self.node_depths[node])
        dt = Decimal(t_old - t_new)
        self.branch_lengths[node] -= dt # if node moves down the tree it's branch length increases (time increases down the tree)
        children = self.node_children[node]
        for child in children:
            self.branch_lengths[child] += dt # if node moves down the tree, child branches decrease


    def group_consecutive_mergers(self, internal_node_list):
        num_nodes = len(internal_node_list)
        node_sets = []
        visited_nodes = []
        for node in internal_node_list:
            if node in visited_nodes:
                continue
            else:
                node_sets.append([node])
                parent_node = self.node_parent[node]
                while parent_node in internal_node_list:
                    node_sets[-1].append(parent_node)
                    visited_nodes.append(parent_node)
                    parent_node = self.node_parent[parent_node]
        # Merge any overlapping sets
        merged_sets = []
        visited_nodes = []
        for node in internal_node_list:
            if node in visited_nodes:
                continue

            in_group = [node in group for group in node_sets]
            if sum(in_group) == 1:
                i = np.arange(len(node_sets))[in_group][0]
                merged_sets.append(node_sets[i])
                visited_nodes += node_sets[i]
            else:
                merged_set = set()
                for i in range(len(node_sets)):
                    if in_group[i]:
                        merged_set = merged_set.union(set(node_sets[i]))
                        visited_nodes += node_sets[i]
                merged_sets.append(list(merged_set))
        return merged_sets

    def merge_internal_nodes(self, node_group, t_merger):
        group_children = []
        old_parents = set()
        for node in node_group:
            old_parents.add(self.node_parent[node])
            for child in self.node_children[node]:
                if child not in group_children and child not in node_group:
                    group_children.append(child)
            if self.node_parent[node] not in node_group:
                top_node = node
            self.adjust_node_time(node, t_merger)
        old_parents = list(old_parents)
        group_parent = self.node_parent[top_node]

        self.node_children[top_node] = group_children
        for child in group_children:
            self.node_parent[child] = top_node
        for node in node_group:
            if node != top_node:
                self.node_parent.pop(node)
                self.node_children.pop(node)
                self.branch_lengths.pop(node)
                self.node_depths.pop(node)

    def get_pair_distance(self, target_pair):
        path2mrca = set()
        current_node = target_pair[0]
        while current_node != 'root':
            path2mrca.add(current_node)
            current_node = self.node_parent[current_node]

        d = Decimal(0)
        current_node = target_pair[1]
        while current_node != 'root':
            if current_node not in path2mrca:
                d += self.branch_lengths[current_node]
                current_node = self.node_parent[current_node]
            else:
                break
        return d

    def ete_format(self):
        #rng = np.random.default_rng()
        #r = rng.integers(2**32 - 1)
        r = np.random.randint(0, 2**32 - 1)
        self.save(f'tree_conversion_{r}.tmp')
        tree_ete = ete3.Tree(f'tree_conversion_{r}.tmp', format=1)
        subprocess.call([f"rm -f tree_conversion_{r}.tmp"], shell=True)
        return tree_ete

def test_class():
    #tree_str = '(((G:2, H:1)F:1, (D:1, E:1)C:2, B:1)A);'
    tree_str = '((B:1, (E:1, F:2)C:1, (G:2)D:1)A);'
    tree = Tree()
    print('(', tree.parse_nwk_entry('('))
    print(')', tree.parse_nwk_entry(')'))
    print(')A', tree.parse_nwk_entry(')A'))
    print(')A:1', tree.parse_nwk_entry(')A:1'))
    print('):1', tree.parse_nwk_entry('):1'))
    print('B:1', tree.parse_nwk_entry('B:1'))
    print(tree)
    tree.read_tree(tree_str)
    print(tree)
    print(tree.node_children.keys())
    print(tree.node_children['root'])
    print(tree.node_children[tree.node_children['root'][0]])
    print(tree.calculate_node_depths(units=None))
    print(tree.node_depths)
    print(tree.calculate_node_depths(units='time'))
    print(tree.node_depths)
    print(tree.end_nodes)
    print(tree.leaves)
    print(tree.preorder_nodes())
    tree.save('../results/tests/tree_test.nwk')
    tree.trim_tree(1)
    print(tree.ete_format())

    print('\n')
    print(tree)
    print('\n')
    print(tree.trim_tree(0.5, units='time'))
    print('\n')
    print(tree.trim_tree(2.0, units='time'))
    print('\n')
    print(tree.trim_tree(2.5, units='time'))
    print('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default=None)
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    if args.test == True:
        test_class()
    if args.input_file is not None:
        tree = Tree(args.input_file)
        #print(tree.node_parent.keys())
        #tree.save('../results/tests/test_tree_output.nwk')

        tree.calculate_node_depths(units='time')

        test_node = tree.leaves[0]
        parent = tree.node_parent[test_node]
        grandparent = tree.node_parent[parent]
        print(test_node, parent, grandparent)
        print(tree.node_children[test_node])
        print(tree.node_children[parent])
        print(tree.node_children[grandparent])

        np.random.seed(1234)
        leaves_sample = np.random.choice(tree.leaves, 30)
        print(leaves_sample)
        subtree = tree.sample_subtree(leaves_sample)
        print(subtree.leaves)
        subtree.save('../results/tests/subtree_test.nwk')

        tree_trimmed = tree.trim_tree(1)
        tree_trimmed.save('../results/tests/test_tree_trimmed.nwk')

        tree_trimmed = tree.trim_tree(150, units='time')
        tree_trimmed.save('../results/tests/test_tree_trimmed_t150.nwk')
